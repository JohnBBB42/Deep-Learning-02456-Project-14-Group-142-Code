from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch

from ase.io import read, write
from ase.mep.neb import NEB, NEBOptimizer, NEBTools
from ase.optimize.bfgs import BFGS
#from neuralneb import painn, utils

#New imports
from ase import Atoms
import ase.db
from ase.visualize import view
import json
import numpy as np
import yaml
import sys
from pathlib import Path

from atomgnn.data.transforms import AddEdgesWithinCutoffDistanceTransform
from atomgnn.calculator import AseCalculator
from scripts.run_painn import LitPaiNNModel
from scripts.run_mace import LitMACEModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = AddEdgesWithinCutoffDistanceTransform(config["data"]["cutoff"])

calc = AseCalculator(
    model,
    transform,
    implemented_properties=["energy", "forces"],
    # properties_map={"energy": "U0"},
    device="cpu",
)

reactant = ase.io.read(reactant_path)
product = ase.io.read(product_path)
assert str(product.symbols) == str(reactant.symbols)

images = [reactant.copy() for _ in range(10)] + [product.copy()]

# Attach new calculator to each atoms object
for image in images:
    image.calc = AseCalculator(
        model,
        transform,
        implemented_properties=["energy", "forces"],
        # properties_map={"energy": "U0"},
        device="cpu",
)

# Optimize the initial and final images
BFGS(images[0]).run(fmax=0.05, steps=1000)
BFGS(images[-1]).run(fmax=0.05, steps=1000)

neb = NEB(images)
neb.interpolate(method="idpp")
relax_neb = NEBOptimizer(neb)
relax_neb.run()

nebtools = NEBTools(images)
fit = nebtools.get_fit()

print(nebtools.get_barrier())

energies = fit.fit_energies.tolist()
path = fit.fit_path.tolist()

def main(args):  # pylint: disable=redefined-outer-name
    statedict = torch.load(args.model)
    model = painn.PaiNN(3, 256, 5)
    model.load_state_dict(statedict)
    model.eval()

    product = read(args.product)
    reactant = read(args.reactant)
    assert str(product.symbols) == str(reactant.symbols), "product and reactant must have same formula. Product: {product.symbols}, Reactant: {reactant.symbols}"
    atom_configs = [reactant.copy() for _ in range(10)] + [product]

    for atom_config in atom_configs:
        atom_config.calc = utils.MLCalculator(model)

    BFGS(atom_configs[0]).run(fmax=0.05, steps=1000)
    BFGS(atom_configs[-1]).run(fmax=0.05, steps=1000)

    neb = NEB(atom_configs)
    neb.interpolate(method="idpp")
    relax_neb = NEBOptimizer(neb)
    relax_neb.run()

    nebtools = NEBTools(atom_configs)
    fit = nebtools.get_fit()

    energies = fit.fit_energies.tolist()
    path = fit.fit_path.tolist()

    mep_fig(path, energies)
    plt.show()
    write("/tmp/rxn.gif", images=atom_configs, format="gif")
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("product", nargs="?", default="data/test_reaction/p.xyz")
    parser.add_argument("reactant", nargs="?", default="data/test_reaction/r.xyz")
    parser.add_argument("--model", nargs="?", default="data/painn.sd")
    args = parser.parse_args()

    main(args)
