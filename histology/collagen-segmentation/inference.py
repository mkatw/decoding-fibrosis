from argparse import ArgumentParser
from pathlib import Path
import os
from os.path import splitext
import sys
import time
import numpy as np
import cv2
import csv
from openslide.lowlevel import OpenSlideUnsupportedFormatError, OpenSlideError
from process_slide import SlideProcessor


class Predictor:

    def __init__(self, model_path):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}  # changing tf verbosity level to minimal
        import tensorflow as tf

        self.tf = tf
        self.collagen_segmenter = self.tf.keras.models.load_model(model_path)

    def __call__(self, tiles, tiles_masks, x_y):
        y = np.squeeze(self.collagen_segmenter.predict(tiles, verbose=0), 3)
        return (y * tiles_masks * 255).astype(np.uint8)  # raw prediction

class OtsuTissueFilter:
    def __init__(self, reader, downsample, filter_colors, luminance_range, tissue_threshold=0.05, color_tolerance=5):
        self.reader = reader
        self.downsample = downsample
        self.filter_colors = [np.array(color) / 255 for color in filter_colors] if filter_colors else []
        self.luminance_range = luminance_range
        self.tissue_threshold = tissue_threshold
        self.color_tolerance = color_tolerance / 255

    def __call__(self, tile, alfa_mask, x_y):
        tile_mask = alfa_mask
        if len(tile.shape) == 3:
            tile_gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
            for color in self.filter_colors:
                color_mask = ~cv2.inRange(tile, color - self.color_tolerance, color + self.color_tolerance).astype(bool)
                tile_mask &= color_mask
        else:
            tile_gray = tile

        gray_mask = (tile_gray <= self.luminance_threshold) & \
                    (tile_gray >= self.luminance_range[0]) & \
                    (tile_gray <= self.luminance_range[1])

        tile_mask &= gray_mask

        if tile_mask.sum() / tile_mask.size >= self.tissue_threshold:
            return tile_mask
        else:
            return None

    @property
    def luminance_threshold(self):
        if not hasattr(self, '_luminance_threshold'):
            level = self.reader.get_best_level_for_downsample(self.downsample)
            slide, alfa_mask = self.reader.get_downsampled_slide(self.reader.level_dimensions[level])
            slide_mask = alfa_mask

            if len(slide.shape) == 3:
                for color in self.filter_colors:
                    color_mask = ~cv2.inRange(slide, color - self.color_tolerance, color + self.color_tolerance).astype(
                        bool)
                    slide_mask &= color_mask
                slide = cv2.cvtColor(slide, cv2.COLOR_RGB2GRAY)

            gray_mask = (slide >= self.luminance_range[0]) & \
                        (slide <= self.luminance_range[1])

            slide_mask &= gray_mask

            slide = (slide * 255).astype(np.uint8)
            threshold, _ = cv2.threshold(slide[slide_mask].ravel(), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            threshold /= 255

            self._luminance_threshold = threshold

        return self._luminance_threshold


def get_main_image_series(slide_path):

    slide_format = splitext(slide_path)[1]
    if slide_format in ['.ndpi', '.svs', '.tiff', '.tif']:
        series = 0
    elif slide_format in ['.scn']:
        series = 1

    return series


def main(args):

    processor = SlideProcessor(args.tile_size, args.stride, 1, np.uint8, 64, Predictor, ((args.model_path,), dict()), n_workers=args.n_workers)

    for input_file in args.input_image:
        
        slide_id = Path(input_file.stem).stem  # this is in case of .ome.tif files which have a double extension
        #slide_id = input_file.stem
        print(slide_id)
        output_file = args.output_dir.joinpath(slide_id + '.tiff')
        series = get_main_image_series(input_file)
        
        if output_file.exists() and output_file.is_file() and args.ignore_existing:
            print('{} already exists, skipping'.format(output_file), file=sys.stderr)
            continue
        try:
            print('Processing {}'.format(output_file))
            start = int(time.time())
            slide_segmented = processor(input_file, 0, OtsuTissueFilter, ((8, [], (0.05, 0.95)), dict(tissue_threshold=0.15)), series=series)  # level
        except (OpenSlideError, OpenSlideUnsupportedFormatError, RuntimeError) as e:
            print('{}: {}'.format(input_file, str(e)), file=sys.stderr)
            continue

        print('Writing to file.')
        processor.write_to_tiff(slide_segmented, str(output_file), tile=True, tile_width=256, tile_height=256, squash=False, pyramid=True, bigtiff=True, compression='VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE', properties=False)

        end = int(time.time())
        print('Done ({} seconds)'.format(end-start))


if __name__ == '__main__':

    parser = ArgumentParser(description='collagen segmentation inference')

    parser.add_argument('input_image', type=Path, nargs='+')
    parser.add_argument('-o', dest='output_dir', type=Path, required=True)
    parser.add_argument('--model', dest='model_path', type=Path, required=True)
    parser.add_argument('--tile-size', dest='tile_size', type=int, default=512)
    parser.add_argument('--stride', dest='stride', type=int, default=256)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--ignore-existing', dest='ignore_existing', action='store_true')
    parser.add_argument('--n-workers', dest='n_workers', type=int, default=8)


    args = parser.parse_args()

    main(args)
