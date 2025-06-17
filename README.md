# Multi-Task Learning (MTL)

## Setup Environment

It is recommended to use `conda` or `venv`.

### Using `conda` (recommended)

```bash
conda create -n MTL python=3.9 -y
conda activate MTL
pip install -r requirements.txt
```

### Using `venv`
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
## Run
### MDMTN
```bash
cd MDMTN
```
#### MGDA
```bash
#training
python mdmtn_mgda.py

#infer
python infer_mgda.py
```
#### Adam
```bash
#training
python mdmtn.py

#infer
python infer.py
```
#### Plot 2D Pareto
```bash
python twoDpf_study_mdmtn.py
```
### PMTL
```bash
cd PMTL
```
```bash
#training
python main.py

#infer
python infer.py
```