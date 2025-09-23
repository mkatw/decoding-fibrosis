# Decoding Fibrosis: Transcriptomic and Clinical Insights via AI-Derived Collagen Deposition Phenotypes in MASLD

This repository contains the code accompanying our preprint:

👉 [Decoding Fibrosis: Transcriptomic and Clinical Insights via AI-Derived Collagen Deposition Phenotypes in MASLD](https://www.medrxiv.org/content/10.1101/2025.08.29.25334719v1)  

A test case with expected results are archived on Zenodo:  
👉 [Zenodo record 16967316](https://zenodo.org/records/16967316)

---

## Repository layout

```
.
├── histology/       # pipelines for segmentation and CDP clustering
│   ├── CDPs/        # paper-version CDP inference (main entry point)
│   └── collagen-segmentation/  # U-Net segmentation (for QA / CPA)
└── omics/           # downstream omics analyses (RNA-seq, SomaScan)
```

- **CDPs**: start here to reproduce collagen deposition phenotypes from PSR slides.  
- **collagen-segmentation**: run segmentation independently, e.g. to inspect quality or compute CPA.  
- **omics**: scripts for bulk RNA-seq and proteomics analyses described in the paper.

---

## Getting started

Clone the repo and create the environment:

```bash
git clone https://github.com/<your-org>/decoding-fibrosis.git
cd decoding-fibrosis
conda env create -f environment.yml
conda activate decoding-fibrosis
```

Download example case from Zenodo:

1. Go to the Zenodo record: https://zenodo.org/records/16967316
2. Download the file named `example_PSR_slide.ndpi`
3. Place the file into your local repository under: 

```
decoding-fibrosis/histology/data/
```

Run CDP inference on PSR slides:

```bash
python histology/CDPs/batch_process_clustering.py
```

Outputs will be saved under `histology/results/`.
We cannot guarantee that the pipeline will perform reliably on PSR slides stained using protocols different from those reported in the paper.
Please note that the pipeline was implemented for slides acquired at 40x. We haven't yet tested it at 20x.
The code for training both the segmentation models and the CDP classifiers will be released in due course.

---

## Acknowledgements

This repository makes use of [wsi-reader](https://github.com/stefano-malacrino/wsi-reader)  
by Stefano Malacrino for efficient whole-slide image handling.

---

## Citation

If you use this code or data, please cite:

- Wojciechowska MK, et al.  
  *Decoding Fibrosis: Transcriptomic and Clinical Insights via AI-Derived Collagen Deposition Phenotypes in MASLD.*  
  medRxiv, 2025. doi: [10.1101/2025.08.29.25334719v1](https://www.medrxiv.org/content/10.1101/2025.08.29.25334719v1)

- Zenodo record: [https://zenodo.org/records/16967316](https://zenodo.org/records/16967316)

This project is licensed under the MIT License.
