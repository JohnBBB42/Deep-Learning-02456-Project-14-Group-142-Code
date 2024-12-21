# Graph Representation learning on Molecular Datasets
The initial aim of this project was to train the Polarizable Atom Interaction Neural Network (PaiNN) model(https://arxiv.org/pdf/2102.03150) on the Quantum Machines 9 (QM9) data set, which is a database of around 134 thousands stable organic molecules with nine heavy atoms(https://arxiv.org/pdf/1703.00564).

The PaiNN model used in this project can be found under models as `/models/painn.py`

The script to running the simple PaiNN model can be found as `/scripts/run_painn.ipynb`

Furthermore, there is a range of extensions has been applied as extensions to the original PaiNN model which are implemented in the trainer class in `/scripts/run_painn.ipynb` including:

•	[5] SWA: https://arxiv.org/pdf/1803.05407.pdf

•	[6] SWAG: https://arxiv.org/pdf/1902.02476.pdf

•	[7] SAM: https://arxiv.org/pdf/2010.01412

•	[8] ASAM: https://arxiv.org/pdf/2102.11600

•	[9] Gaussian Negetive Log Likelihood: https://arxiv.org/abs/2006.04910, https://arxiv.org/pdf/2212.09184

•	[10] Laplacian Approximation: https://openreview.net/pdf?id=A6EquH0enk

The dataset used was the QM9 database which can be downloaded as an ASE database from `/data/`
The splits used was 110k for training 10k for validation and the rest for testing and can be downloaded from `/data/` 
