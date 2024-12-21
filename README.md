# Graph Representation learning on Molecular Datasets
The initial aim of this project was to train the Polarizable Atom Interaction Neural Network (PaiNN) model(https://arxiv.org/pdf/2102.03150) on the Quantum Machines 9 (QM9) data set, which is a database of around 134 thousands stable organic molecules with nine heavy atoms(https://arxiv.org/pdf/1703.00564).

A notebook example of the PaiNN model used in this project and the training script to running the simple PaiNN model can be found as `/scripts/run_painn.ipynb`

Furthermore, there is a range of modfications has been applied as extensions to the original PaiNN model which are implemented in the trainer class and can be turned on and off.
The modifications:

•	[5] SWA: https://arxiv.org/pdf/1803.05407.pdf

•	[6] SWAG: https://arxiv.org/pdf/1902.02476.pdf

•	[7] SAM: https://arxiv.org/pdf/2010.01412

•	[8] ASAM: https://arxiv.org/pdf/2102.11600

•	[9] Gaussian Negetive Log Likelihood: https://arxiv.org/abs/2006.04910

•	[10] Laplacian Approximation: https://openreview.net/pdf?id=A6EquH0enk

The dataset used was the QM9 database, which can be downloaded as an ASE database from Kaggle.
The splits used was 110k for training 10k for validation and the rest for testing and can be downloaded from `/data/`. 
The code should be relatively straigt forward to use, the number of steps and validation interval can be controlled directly in the training loop and so can the hyperparameters.
Note: the SWAG number of samples are controlled directly in the get_swag_predictions.
