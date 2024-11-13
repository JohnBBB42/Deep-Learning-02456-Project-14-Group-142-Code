import argparse
import csv
import os
from pathlib import Path
import torch
from ase.io import read, write
from ase.mep.neb import NEB, NEBOptimizer, NEBTools
from ase.optimize.bfgs import BFGS
import yaml

from atomgnn.data.transforms import AddEdgesWithinCutoffDistanceTransform
from atomgnn.calculator import AseCalculator
from scripts.run_painn import LitPaiNNModel
from scripts.run_mace import LitMACEModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_neb(reaction_dir, config_path, model_checkpoint, model_type):
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load the model
    if model_type.lower() == "painn":
        lit_model = LitPaiNNModel.load_from_checkpoint(str(model_checkpoint))
        model = lit_model.model  # Extract the PyTorch model from the Lightning wrapper
    elif model_type.lower() == "mace":
        lit_model = LitMACEModel.load_from_checkpoint(str(model_checkpoint))
        model = lit_model.model  # Extract the PyTorch model from the Lightning wrapper
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.eval()  # Set the model to evaluation mode

    # Define paths to reactant, product, and transition state files
    reactant_path = Path(reaction_dir) / "r.xyz"
    product_path = Path(reaction_dir) / "p.xyz"
    ts_path = Path(reaction_dir) / "ts.xyz"

    # Check if all files exist
    if not reactant_path.exists() or not product_path.exists() or not ts_path.exists():
        raise FileNotFoundError("One of the required files (r.xyz, p.xyz, ts.xyz) is missing.")

    # Load structures
    reactant = read(reactant_path)
    product = read(product_path)
    transition_state = read(ts_path)

    # Ensure the atomic symbols are the same
    assert str(product.symbols) == str(reactant.symbols) == str(transition_state.symbols), \
        "Reactant, product, and transition state must have the same formula."

    # Set up the calculator transform
    transform = AddEdgesWithinCutoffDistanceTransform(config["data"]["cutoff"])

    # Prepare images for NEB
    images = [reactant.copy() for _ in range(10)] + [product.copy()]
    # Attach a new calculator to each atoms object
    for image in images:
        image.calc = AseCalculator(
            model,
            transform,
            implemented_properties=["energy", "forces"],
            device=DEVICE,
        )

    # Optimize initial and final images
    BFGS(images[0], logfile=None).run(fmax=0.05, steps=500)
    BFGS(images[-1], logfile=None).run(fmax=0.05, steps=500)

    # Set up and run NEB
    neb = NEB(images)
    neb.interpolate(method="idpp")
    relax_neb = NEBOptimizer(neb, logfile=None)
    relax_neb.run(fmax=0.05, steps=500)

    # Analyze NEB results
    nebtools = NEBTools(images)
    fit = nebtools.get_fit()
    barrier_energy, barrier_id = nebtools.get_barrier()
    energies = fit.fit_energies.tolist()
    path = fit.fit_path.tolist()

    # Calculate the true barrier energy using the transition state
    # Set up the calculator for the transition state
    transition_state.calc = AseCalculator(
        model,
        transform,
        implemented_properties=["energy", "forces"],
        device=DEVICE,
    )
    # Compute energies
    reference_energy = images[0].get_potential_energy()
    ts_energy = transition_state.get_potential_energy()
    true_barrier_energy = ts_energy - reference_energy
    # Calculate the barrier error
    energy_difference = abs(true_barrier_energy - barrier_energy)

    # Return results
    return {
        "barrier_energy": barrier_energy,
        "true_barrier_energy": true_barrier_energy,
        "energy_difference": energy_difference,
        "path": path,
        "energies": energies
    }

def write_csv(file_path, data):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Reaction Folder", "Calculated Barrier [eV]", "True Barrier [eV]", "Barrier Error [eV]"])
        for entry in data:
            reaction_folder = entry['reaction_folder']
            barrier_energy = entry['barrier_energy']
            true_barrier_energy = entry['true_barrier_energy']
            energy_difference = entry['energy_difference']
            writer.writerow([reaction_folder, barrier_energy, true_barrier_energy, energy_difference])
    print(f"CSV file saved to {file_path}")

def main(args):
    base_directory = args.base_directory
    config_path = Path(args.config)
    model_checkpoint = Path(args.model_checkpoint)
    model_type = args.model_type

    all_data = []

    # Iterate through the specified split directory (e.g., 'test')
    split_path = Path(base_directory) / args.split
    if split_path.exists() and split_path.is_dir():
        # Iterate through reaction folders in the split directory
        reaction_folders = [f for f in split_path.iterdir() if f.is_dir()]
        total_reactions = len(reaction_folders)
        print(f"Total reactions to process: {total_reactions}")

        for index, reaction_dir in enumerate(reaction_folders):
            print(f"Processing reaction: {reaction_dir.name} ({index + 1}/{total_reactions})")
            try:
                result = run_neb(reaction_dir, config_path, model_checkpoint, model_type)
                all_data.append({
                    "reaction_folder": reaction_dir.name,
                    "barrier_energy": result["barrier_energy"],
                    "true_barrier_energy": result["true_barrier_energy"],
                    "energy_difference": result["energy_difference"],
                })
            except Exception as e:
                print(f"Failed to process NEB for {reaction_dir.name}: {e}")
    else:
        print(f"Split directory '{args.split}' does not exist in base directory.")

    output_csv = os.path.join(args.output_dir, "all_reactions_barriers.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    write_csv(output_csv, all_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NEB calculations over multiple reactions.")
    parser.add_argument("base_directory", help="Base directory containing the split subdirectories (e.g., 'test')")
    parser.add_argument("--split", default="test", help="Name of the split directory (e.g., 'test', 'train', 'validation')")
    parser.add_argument("--config", required=True, help="Path to the config file")
    parser.add_argument("--model_checkpoint", required=True, help="Path to the model checkpoint")
    parser.add_argument("--model_type", choices=["painn", "mace"], required=True, help="Type of model to use (painn or mace)")
    parser.add_argument("--output_dir", default="./outputs", help="Directory to save the output CSV file")
    args = parser.parse_args()
    main(args)
