# Deep_Learning_02456_Project_14_Group_142_Code
Title: Graph Representation learning on Molecular Datasets
The initial aim of the project is to train the Polarizable Atom Interaction Neural Network(PaiNN) model(https://arxiv.org/pdf/2102.03150) on the Quantum Machines 9(QM9) data set, which is a database of around 134 thousands stable organic molecules with nine heavy atoms(https://arxiv.org/pdf/1703.00564).
Furthermore, there is a range of extensions that could be applied to the model including:
•	[5] SWA: https://arxiv.org/pdf/1803.05407.pdf
•	[6] SWAG: https://arxiv.org/pdf/1902.02476.pdf
•	[7] SAM: https://arxiv.org/pdf/2010.01412
•	[8] ASAM: https://arxiv.org/pdf/2102.11600
•	[9] Heteroscedastic Regression https://arxiv.org/abs/2006.04910, https://arxiv.org/pdf/2212.09184, https://openreview.net/pdf?id=A6EquH0enk
As I have experience with this model and can skip much of the introduction and start training models therefore I will aim after implementing all these approaches and compare the results focusing mainly on energy and or force prediction.
I will train a model for each approach and then compare them with a benchmark being the plain PaiNN model trained on QM9, I will then if time permits try to combine some of the methods to see if that can provide even better results.
I would also like to try out this hopefully improved model on the more challenging Transition1x data set and see if I can surpass the state-of-the-art results on this data set or other data sets if we can find suitable sets (https://arxiv.org/pdf/2207.12858, https://pubs.rsc.org/en/content/articlepdf/2023/cp/d3cp02143b)
Then it could be fun to try out other models as MACE(https://arxiv.org/pdf/2206.07697), NEQUIP(https://arxiv.org/pdf/2101.03164) or others to see if they perform better than an improved PaiNN model.
If I succeed in all of this and have time I would be very interested in discussing further ideas for improvement.
