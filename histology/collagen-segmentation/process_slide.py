import os
import numpy as np
from scipy import signal
import pyvips
import multiprocessing as mp
from wsi_reader import get_reader_impl
import zarr
from pathlib import Path
from tifffile import imwrite

ctx = mp.get_context('spawn')

class TileReader(mp.Process):

    def __init__(self, n_iter, barrier, queue_in, queue_out, reader_class, slide_path, series, level, tile_size, normalize, downsample_level_0, tile_masker_class, tile_masker_args):
        ctx.Process.__init__(self, name='TileReader')
        self.daemon = True
        self.n_iter = n_iter
        self.barrier = barrier
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.reader_class = reader_class
        self.slide_path = slide_path
        self.series = series
        self.level = level
        self.tile_size = tile_size
        self.normalize = normalize
        self.downsample_level_0 = downsample_level_0
        self.tile_masker_class = tile_masker_class
        self.tile_masker_args, self.tile_masker_kwargs = tile_masker_args

    def run(self):
        reader = self.reader_class(self.slide_path, series=self.series)
        tile_masker = self.tile_masker_class(reader, *self.tile_masker_args, **self.tile_masker_kwargs)

        for i in range(self.n_iter):
            done = False
            while not done:
                data_in = self.queue_in.get()
                if data_in is None:
                    done = True
                    self.queue_out.put(None)
                else:
                    x_y = data_in
                    tile, alfa_mask = reader.read_region(x_y, self.level, self.tile_size, downsample_level_0=self.downsample_level_0, normalize=self.normalize)
                    tile_mask = None
                    if tile is not None:
                        tile_mask = tile_masker(tile, alfa_mask, x_y)
                        if tile_mask is not None:
                            self.queue_out.put((tile, tile_mask, x_y))
                    self.queue_in.task_done()
            self.barrier.wait()
            self.queue_out.join()
            self.queue_in.task_done()
            
class TileProcessor(ctx.Process):

    def __init__(self, n_iter, queue_in, queue_out, n_workers, batch_size, processor, processor_args):
        ctx.Process.__init__(self, name='TileProcessor')
        self.daemon = True
        self.n_iter = n_iter
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.processor_class = processor
        self.processor_args, self.processor_kwargs = processor_args
        
    def run(self):
        processor = self.processor_class(*self.processor_args, **self.processor_kwargs)
        while True:
            for _ in range(self.n_iter):
                n = self.n_workers
                last_batch = False
                batch = []
                while not last_batch:
                    data_in = self.queue_in.get()
                    if data_in is None:
                        n -= 1
                        if n == 0:
                            last_batch = True
                        else:
                            self.queue_in.task_done()
                    else:
                        tile, tile_mask, x_y = data_in
                        batch.append(data_in)
                    
                    if len(batch) == self.batch_size or (last_batch and len(batch) > 0):
                        tiles, tiles_masks, x_y = list(zip(*batch))
                        tiles, tiles_masks = np.array(tiles), np.array(tiles_masks)
                        res_batch = processor(tiles, tiles_masks, x_y)
                        batch = []
                        for data_out in zip(res_batch, x_y):
                            self.queue_out.put(data_out)
                            self.queue_in.task_done()
                        
                for _ in range(self.n_workers):
                    self.queue_out.put(None)
                self.queue_out.join()
                self.queue_in.task_done()
    
class TileWriter(ctx.Process):

    def __init__(self, n_iter, barrier, queue_in, zarr_store, zarr_store_c, tile_size, stride, n_channels, dtype):
        ctx.Process.__init__(self, name='TileWriter')
        self.daemon = True
        self.n_iter = n_iter
        self.barrier = barrier
        self.queue_in = queue_in
        self.tile_size = tile_size
        self.stride = stride
        self.n_channels = n_channels
        self.dtype = dtype
        self.zarr_store = zarr_store
        self.zarr_store_c = zarr_store_c
        
    def _spline_window(self, window_size, stride, n_channels, power=2):
        """
        Squared spline (power=2) window function:
        https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
        """
        overlap = window_size-stride 
        interp_wind_size = overlap*2 
        i = int(interp_wind_size / 4)
        wind_outer = (abs(2 * (signal.triang(interp_wind_size))) ** power) / 2 
        wind_outer[i:-i] = 0 
        wind_inner = 1 - (abs(2 * (signal.triang(interp_wind_size) - 1)) ** power) / 2 
        wind_inner[:i] = 0
        wind_inner[-i:] = 0

        wind = wind_inner + wind_outer
        wind_l, wind_r = np.split(wind, 2)
        
        wind = np.concatenate([wind_l, np.ones(window_size-interp_wind_size), wind_r], axis=0)
        wind = wind[..., None]
        wind = wind * wind.transpose()
        
        if n_channels > 1:
            wind = wind[..., None]
            
        return wind
        
    def run(self):
        
        output = zarr.open(self.zarr_store, mode='r+')
        output_c = zarr.open(self.zarr_store_c, mode='r+')
        height, width = output.shape[:2]
        spline = self._spline_window(self.tile_size, self.stride, self.n_channels, power=2).astype(output.dtype)

        for _ in range(self.n_iter):
            done = False
            while not done:
                data_in = self.queue_in.get()
                if data_in is None:
                    done = True
                else:
                    tile, (x, y) = data_in
                    x_dst, y_dst = max(x, 0), max(y, 0)
                    x_src, y_src = -min(x, 0), -min(y, 0)
                    w = min(self.tile_size, width-x) - x_src
                    h = min(self.tile_size, height-y) - y_src
                    output[y_dst:y_dst+h, x_dst:x_dst+w] += (tile*spline)[y_src:y_src+h, x_src:x_src+w]
                    output_c[y_dst:y_dst+h, x_dst:x_dst+w] += spline[y_src:y_src+h, x_src:x_src+w]
                self.queue_in.task_done()
            self.barrier.wait()

class TileTransformer(ctx.Process):

    def __init__(self, queue_in, zarr_store, zarr_store_c, tile_size):
        ctx.Process.__init__(self, name='TileWriter')
        self.daemon = True
        self.queue_in = queue_in
        self.tile_size = tile_size
        self.zarr_store = zarr_store
        self.zarr_store_c = zarr_store_c
        
    def run(self):
        output = zarr.open(self.zarr_store, mode='r+')
        output_c = zarr.open(self.zarr_store_c, mode='r+')
        done = False
        
        while not done:
            data = self.queue_in.get()
            if data is None:
                done = True
            else:
                x, y = data
                with np.errstate(divide='ignore', invalid='ignore'):
                    c = 1/(output_c[y:y+self.tile_size, x:x+self.tile_size])
                    c[output_c[y:y+self.tile_size, x:x+self.tile_size] == 0] = 0
                    output[y:y+self.tile_size, x:x+self.tile_size] *= c
        
class SlideProcessor(object):
     
    def __init__(self, tile_size, stride, n_channels, dtype, batch_size, processor_class, processor_args, precision='single', n_workers=8):
 
        self.tile_size = tile_size
        self.stride = stride
        self.n_channels = n_channels
        self.dtype = dtype
        self.precision = {'half': np.float16, 'single': np.float32, 'double': np.float64}[precision]
        self.batch_size = batch_size
        self.processor_class = processor_class
        self.processor_args = processor_args
        self.n_workers = n_workers
        self.n_iter = 1 if self.stride == self.tile_size else 9
        
        self.queue_readers = ctx.JoinableQueue(4*self.batch_size)
        self.queue_processor = ctx.JoinableQueue(4*self.batch_size)
        self.queue_writers = ctx.JoinableQueue(4*self.batch_size)
        self.barrier_readers = ctx.Barrier(self.n_workers)
        self.barrier_writers = ctx.Barrier(self.n_workers)
        self.queue_transformers = ctx.Queue()
        
        self.processor = TileProcessor(self.n_iter, self.queue_processor, self.queue_writers, self.n_workers, self.batch_size, self.processor_class, self.processor_args)
        
        self.processor.start()

    def __call__(self, filename, level, tile_masker_class, tile_masker_args, series=0, normalize=True, downsample_level_0=False):
                
        reader_class = get_reader_impl(filename)
        reader = reader_class(filename, series=series)
        width, height = reader.level_dimensions[level]
        reader.close()
        
        if self.stride == self.tile_size:
            coords_batches = [
                ((x,y) for y in np.arange(0, height, self.stride)
                           for x in np.arange(0, width, self.stride))
            ] 
        else:
            coords_batches = [
                ((x, y) for y in np.arange(0, height, 2*self.stride)
                            for x in np.arange(0, width, 2*self.stride)),
                ((x, y) for y in np.arange(-self.stride, height, 4*self.stride)
                            for x in np.arange(0, width, 2*self.stride)),
                ((x, y) for y in np.arange(self.stride, height, 4*self.stride)
                            for x in np.arange(0, width, 2*self.stride)),
                ((x, y) for y in np.arange(0, height, 2*self.stride)
                            for x in np.arange(-self.stride, width, 4*self.stride)),
                ((x, y) for y in np.arange(0, height, 2*self.stride)
                            for x in np.arange(self.stride, width, 4*self.stride)),
                ((x, y) for y in np.arange(-self.stride, height, 4*self.stride)
                            for x in np.arange(-self.stride, width, 4*self.stride)),
                ((x, y) for y in np.arange(-self.stride, height, 4*self.stride)
                            for x in np.arange(self.stride, width, 4*self.stride)),
                ((x, y) for y in np.arange(self.stride, height, 4*self.stride)
                            for x in np.arange(-self.stride, width, 4*self.stride)),
                ((x, y) for y in np.arange(self.stride, height, 4*self.stride)
                            for x in np.arange(self.stride, width, 4*self.stride))
            ]
        
        if self.n_channels > 1:
            shape = (height, width, self.n_channels)
            chunk_shape = (self.tile_size, self.tile_size, self.n_channels)
        else:
            shape = (height, width)
            chunk_shape = (self.tile_size, self.tile_size)
            
        output = zarr.open(zarr.storage.TempStore(), mode='a', shape=shape, chunks=chunk_shape, dtype=self.precision)
        
        output_c = zarr.open(zarr.storage.TempStore(), mode='a', shape=shape, chunks=chunk_shape[:2], dtype=self.precision)

        readers = [TileReader(self.n_iter, self.barrier_readers, self.queue_readers, self.queue_processor, reader_class, filename, series, level, self.tile_size, normalize, downsample_level_0, tile_masker_class, tile_masker_args) for _ in range(self.n_workers)]
        
        for reader in readers:
            reader.start()
        
        writers = [TileWriter(self.n_iter, self.barrier_writers, self.queue_writers, output.chunk_store.path, output_c.chunk_store.path, self.tile_size, self.stride, self.n_channels, self.precision) for _ in range(self.n_workers)]
        
        for writer in writers:
            writer.start()
        
        for coords in coords_batches:
            for x_y in coords:
                self.queue_readers.put(x_y)
            for _ in range(self.n_workers):
                self.queue_readers.put(None)
            self.queue_readers.join()
        
        for reader in readers:
            reader.join()
        
        for writer in writers:
            writer.join()
        
        if self.stride < self.tile_size:
            transformers = [TileTransformer(self.queue_transformers, output.chunk_store.path, output_c.chunk_store.path, self.tile_size) for _ in range(self.n_workers)]
        
            for transformer in transformers:
                transformer.start()
        
            for y in np.arange(0, height, self.tile_size):
                for x in np.arange(0, width, self.tile_size):
                    self.queue_transformers.put((x, y))
        
            for _ in range(self.n_workers):
                self.queue_transformers.put(None)
                
            for transformer in transformers:
                transformer.join()

        return output.astype(self.dtype)
        
    def close(self):
        self.processor.terminate()
        self.processor.join()
        
    @staticmethod
    def write_to_tiff(z, filename, channel=None, **tiffsave_kwargs):
        # See https://jcupitt.github.io/libvips/API/current/VipsForeignSave.html#vips-tiffsave for tiffsave_kwargs        
        n_y, n_x = z.cdata_shape[:2]
        tile_h, tile_w = z.chunks[:2]
        bands = z.chunks[2] if channel is None and len(z.chunks) == 3 else 1
        shape = z.shape[:2] if bands == 1 else z.shape
        if len(z.shape) == 2:
            tiles = ((z[y * tile_h:y * tile_h + tile_h, x * tile_w:x * tile_w + tile_w] for y in range(n_y) for x in
             range(n_x)))
        else:
            tiles = ((z[y*tile_h:y*tile_h+tile_h,x*tile_w:x*tile_w+tile_w,channel] for y in range(n_y) for x in range(n_x)))
        img_tmp = str(Path(z.chunk_store.path) / 'tmp.tif')
        imwrite(img_tmp, tiles, bigtiff=True, shape=shape, dtype=z.dtype, tile=z.chunks[:2])#, compression='deflate')
        pyvips.Image.new_from_file(img_tmp).tiffsave(str(filename), **tiffsave_kwargs)
        os.remove(img_tmp)
