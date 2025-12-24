from __future__ import annotations
from typing import Dict, List
import random
import os
from pathlib import Path
import warnings
from collections import defaultdict


from lightning.fabric import seed_everything

from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine, RFD3Output


from mpnn.inference_engines.mpnn import MPNNInferenceEngine
from mpnn.utils.inference import MPNNInferenceOutput


from rf3.inference_engines.rf3 import RF3InferenceEngine
from rf3.utils.inference import InferenceInput

from biotite.structure import rmsd, superimpose
import biotite.structure as struc
import numpy as np
import pandas as pd

os.environ['CCD_MIRROR_PATH'] = ''
os.environ['PDB_MIRROR_PATH'] = ''

warnings.filterwarnings('ignore', module='atomworks')

# Set seed for reproducibility
seed = random.randint(0, 1000000)
seed_everything(seed, workers=True)


def run_rfd3():
    # Configure RFD3 inference
    engine_config = RFD3InferenceConfig(
        diffusion_batch_size=2,  # Generate 2 structures per batch
    )

    config = {
        "inputs": 'enzyme_design.json',      # None for unconditional generation
        "out_dir": None,     # None to return in memory (no file output)
        "n_batches": 2,      # Generate 2 batches
    }

    # Initialize engine and run generation
    model = RFD3InferenceEngine(**engine_config)
    outputs = model.run(**config)

    out_dir = Path("rfd3_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for output_list in outputs.values():
        for output in output_list:
            output.dump(out_dir=out_dir, verbose=True)

    return outputs


def run_mpnn(rfd3_outputs: Dict[str, List[RFD3Output]]):
    # Configure MPNN inference engine
    # See mpnn.utils.inference.MPNN_GLOBAL_INFERENCE_DEFAULTS for all options

    engine_config = {
        "model_type": "ligand_mpnn",     # or "protein_mpnn" for vanilla ProteinMPNN
        "is_legacy_weights": True,       # Required for now for ligand_mpnn and protein_mpnn
        # out_directory must be a string when provided.
        "out_directory": "mpnn_outputs",
        "write_structures": True,
        "write_fasta": True,
    }

    # Configure per-input inference options
    # See mpnn.utils.inference.MPNN_PER_INPUT_INFERENCE_DEFAULTS for all options
    config = defaultdict(list)
    for rfd3_example_id, rfd3_output_list in rfd3_outputs.items():
        for rfd3_output in rfd3_output_list:
            config['input_dicts'].append({
                "name": f"{rfd3_output.example_id}",
                # Example fixed residues
                "fixed_residues": ["A5", "A15", "A20"],
                "batch_size": 10,           # Generate 10 sequences per structure
                "remove_waters": True,
            })
            config['atom_arrays'].append(rfd3_output.atom_array)

    # Run sequence design on the RFD3-generated backbone
    model = MPNNInferenceEngine(**engine_config)
    outputs = model.run(**config)

    return outputs


def run_rf3(mpnn_outputs: list[MPNNInferenceOutput]):
    # Initialize RF3 inference engine
    engine_config = {
        "ckpt_path": 'rf3',
        "verbose": False,
    }

    # Create input from the MPNN-designed structure (first design)
    # This re-folds the sequence to validate it adopts the intended structure
    inputs = [
        InferenceInput.from_atom_array(mpnn_output.atom_array,
                                       example_id=f"{mpnn_output.input_dict['name']}_b{mpnn_output.output_dict['batch_idx']}_d{mpnn_output.output_dict['design_idx']}")
        for mpnn_output in mpnn_outputs
    ]

    config = {
        "inputs": inputs,
        "out_dir": Path("rf3_outputs"),
    }

    model = RF3InferenceEngine(**engine_config)
    outputs = model.run(**config)

    return outputs




def validate(mpnn_outputs: list[MPNNInferenceOutput], rf3_outputs: Dict[str, List]):
    results = []
    for mpnn_output in mpnn_outputs:
        example_id=f"{mpnn_output.input_dict['name']}_b{mpnn_output.output_dict['batch_idx']}_d{mpnn_output.output_dict['design_idx']}"

        # id = f"{mpnn_output.input_dict['name']}_{mpnn_output.output_dict['design_idx']}"
        # Extract the top-ranked prediction
        rf3_output = rf3_outputs[example_id][0]

        aa_generated = mpnn_output.atom_array
        aa_refolded = rf3_output.atom_array

        # Filter to backbone atoms (N, CA, C, O)
        bb_generated = aa_generated[struc.filter_amino_acids(aa_generated) & np.isin(aa_generated.atom_name, ('N', 'CA', 'C', 'O'))]
        bb_refolded = aa_refolded[struc.filter_amino_acids(aa_refolded) & np.isin(aa_refolded.atom_name, ('N', 'CA', 'C', 'O'))]

        # Superimpose structures and calculate RMSD
        bb_refolded_fitted, transform = superimpose(bb_generated, bb_refolded)
        rmsd_value = rmsd(bb_generated, bb_refolded_fitted)

        results.append({
            "example_id": example_id,
            "designed_sequence": mpnn_output.output_dict['designed_sequence'],
            "rmsd": rmsd_value,
            # Include additional RF3 confidence metrics
            **{k: v for k, v in rf3_output.summary_confidences.items() if not isinstance(v, list)},
            "seed": rf3_output.seed # Include seed for reproducibility, seed_everything was used
        })


    return results


def main():
    rfd3_outputs = run_rfd3()
    mpnn_outputs = run_mpnn(rfd3_outputs)
    rf3_outputs = run_rf3(mpnn_outputs)
    validation_results = validate(mpnn_outputs, rf3_outputs)

    # for result in validation_results:
    #     print(result)
    df = pd.DataFrame(validation_results)
    # df = pd.DataFrame(validation_results,
    #                   columns=["design_id", "designed_sequence", "ptm", "iptm", "rmsd"])
    df.to_csv("validation_results.csv", index=False)
    print("Validation results saved to validation_results.csv")

if __name__ == "__main__":
    main()