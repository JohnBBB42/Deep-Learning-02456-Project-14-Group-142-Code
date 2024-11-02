# Graph Representation learning on Molecular Datasets
The initial aim of this project was to train the Polarizable Atom Interaction Neural Network (PaiNN) model(https://arxiv.org/pdf/2102.03150) on the Quantum Machines 9(QM9) data set, which is a database of around 134 thousands stable organic molecules with nine heavy atoms(https://arxiv.org/pdf/1703.00564).

The PaiNN model used in this project can be found under models as `models/painn.py`

The Pytorch Lightning script to running the simple PaiNN model can be found as `/scripts/run_painn.py`

Furthermore, there is a range of extensions has been applied as extensions to the original PaiNN model which can also be found in `/scripts` including:

•	[5] SWA: https://arxiv.org/pdf/1803.05407.pdf

•	[6] SWAG: https://arxiv.org/pdf/1902.02476.pdf

•	[7] SAM: https://arxiv.org/pdf/2010.01412

•	[8] ASAM: https://arxiv.org/pdf/2102.11600

•	[9] Heteroscedastic Regression https://arxiv.org/abs/2006.04910, https://arxiv.org/pdf/2212.09184, https://openreview.net/pdf?id=A6EquH0enk

compare the results focusing mainly on energy and or force prediction. compare them with a benchmark being the plain PaiNN model trained on QM9,
combine some of the methods to see if that can provide even better results.

more challenging Transition1x data set (https://arxiv.org/pdf/2207.12858, https://pubs.rsc.org/en/content/articlepdf/2023/cp/d3cp02143b)
MACE(https://arxiv.org/pdf/2206.07697), 
NEQUIP(https://arxiv.org/pdf/2101.03164)
