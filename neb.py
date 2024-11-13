import argparse
from argparse import ArgumentParser
from pathlib import Path
import torch
import ase
from ase.io import read, write
from ase.mep.neb import NEB, NEBOptimizer, NEBTools
from ase.optimize.bfgs import BFGS
import matplotlib.pyplot as plt
import yaml

from atomgnn.data.transforms import AddEdgesWithinCutoffDistanceTransform
from atomgnn.calculator import AseCalculator
from scripts.run_painn import LitPaiNNModel
from scripts.run_mace import LitMACEModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main(args):
    # Load configuration and model
    config_path = Path(args.config)
    model_checkpoint = Path(args.model_checkpoint)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load the model
    if args.model_type.lower() == "painn":
        lit_model = LitPaiNNModel.load_from_checkpoint(str(model_checkpoint))
        model = lit_model.model  # Extract the PyTorch model from the Lightning wrapper
    elif args.model_type.lower() == "mace":
        lit_model = LitMACEModel.load_from_checkpoint(str(model_checkpoint))
        model = lit_model.model  # Extract the PyTorch model from the Lightning wrapper
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    model.eval()  # Set the model to evaluation mode

    # Load reactant and product structures
    reactant_path = Path(args.reactant)
    product_path = Path(args.product)
    
    reactant = read(reactant_path)
    product = read(product_path)
    assert str(product.symbols) == str(reactant.symbols), "Product and reactant must have the same formula"
    
    # Set up the calculator
    transform = AddEdgesWithinCutoffDistanceTransform(config["data"]["cutoff"])
    
    # Prepare images for NEB
    images = [reactant.copy() for _ in range(10)] + [product.copy()]
    for image in images:
        image.calc = AseCalculator(
            model,
            transform,
            implemented_properties=["energy", "forces"],
            device=DEVICE,
        )

    
    # Optimize initial and final images
    BFGS(images[0]).run(fmax=0.05, steps=1000)
    BFGS(images[-1]).run(fmax=0.05, steps=1000)
    
    # Set up and run NEB
    neb = NEB(images)
    neb.interpolate(method="idpp")
    relax_neb = NEBOptimizer(neb)
    relax_neb.run()
    
    # Analyze NEB results
    nebtools = NEBTools(images)
    fit = nebtools.get_fit()
    print("Energy barrier:", nebtools.get_barrier())
    
    energies = fit.fit_energies.tolist()
    path = fit.fit_path.tolist()
    
    # Plot the Minimum Energy Path (MEP)
    mep_fig(path, energies)
    plt.show()
    write("/tmp/rxn.gif", images=images, format="gif")
    plt.show()

def mep_fig(path, energies):
    plt.figure()
    plt.plot(path, energies)
    plt.xlabel("Reaction Coordinate")
    plt.ylabel("Energy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NEB calculation using a PaINN or MACE model.")
    parser.add_argument("product", help="Path to the product structure file (e.g., p.xyz)")
    parser.add_argument("reactant", help="Path to the reactant structure file (e.g., r.xyz)")
    parser.add_argument("--config", required=True, help="Path to the config file")
    parser.add_argument("--model_checkpoint", required=True, help="Path to the model checkpoint")
    parser.add_argument("--model_type", choices=["painn", "mace"], required=True, help="Type of model to use (painn or mace)")
    args = parser.parse_args()
    main(args)
