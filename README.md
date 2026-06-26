# Decoding Fibrosis: Transcriptomic and Clinical Insights via AI-Derived Collagen Deposition Phenotypes in MASLD

This repository contains the code accompanying our paper:

👉 [Decoding Fibrosis: Transcriptomic and Clinical Insights via AI-Derived Collagen Deposition Phenotypes in MASLD](https://journals.lww.com/hep/abstract/9900/decoding_fibrosis__transcriptomic_and_clinical.1633.aspx)  

A test case with expected results and training data are archived on Zenodo:  
👉 [Zenodo record](https://zenodo.org/records/16967315)

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

- Wojciechowska MK, Thing M, et al.  
  *Decoding Fibrosis: Transcriptomic and Clinical Insights via AI-Derived Collagen Deposition Phenotypes in MASLD.*  
  Hepatology, 2026. doi: [10.1097/HEP.0000000000001811](https://journals.lww.com/hep/abstract/9900/decoding_fibrosis__transcriptomic_and_clinical.1633.aspx)

- Zenodo record: [https://zenodo.org/records/16967316](https://zenodo.org/records/16967315)

This project is licensed under the MIT License.
