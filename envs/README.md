# Environment setup (omics)

This project uses conda for environment management.  
All environment definitions live in envs/.

### 1. Create the environment
```bash
conda env create -f envs/environment-omics.yml
```
### 2. Activate the environment
```bash
conda activate omics-env
```
### 3. Register the Jupyter kernel (one time only)
```bash
python -m ipykernel install --user --name=omics-env --display-name "Python (omics)"
```
### 4. Open notebooks
Launch Jupyter.
```bash
jupyter notebook
```
All notebooks in omics/ are configured to use this kernel by default.
