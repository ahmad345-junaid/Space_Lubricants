from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank candidate molecules with desirability and uncertainty-aware acquisition.")
    parser.add_argument("--config", default="configs/active_learning.yaml", help="Path to the active-learning configuration file.")
    parser.add_argument("--candidates", default="data/raw/sample_candidates.csv", help="Path to the candidate CSV file.")
    parser.add_argument("--checkpoint", default="results/trained_models/best_model.pt", help="Path to the trained model checkpoint.")
    parser.add_argument("--gpr", default="results/trained_models/gpr_uncertainty.joblib", help="Path to the fitted GPR artifact.")
    parser.add_argument("--output-dir", default="results/candidate_rankings", help="Directory for ranking outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from leo_lubricants.active_learning.acquisition import AcquisitionConfig
    from leo_lubricants.active_learning.desirability import DesirabilityConfig, DesirabilityTarget
    from leo_lubricants.active_learning.select_candidates import CandidateSelectionConfig, rank_candidates, select_top_batch
    from leo_lubricants.data.io import read_table, write_table
    from leo_lubricants.data.preprocess import preprocess_candidates
    from leo_lubricants.data.schema import DatasetSchema, TARGET_COLUMNS
    from leo_lubricants.features.descriptors import DESCRIPTOR_COLUMNS
    from leo_lubricants.models.gpr_uncertainty import MultiOutputGPRUncertainty
    from leo_lubricants.utils.config import load_config
    from leo_lubricants.utils.logging import get_logger
    from leo_lubricants.workflows import load_checkpoint, prepare_batch

    logger = get_logger("active_learning")
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    candidate_frame = read_table(args.candidates)
    processed_candidates = preprocess_candidates(candidate_frame, DatasetSchema())
    candidate_batch = prepare_batch(processed_candidates, device=device, include_targets=False)
    model, payload = load_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(candidate_batch.batch).detach().cpu().numpy()
    gpr_model = MultiOutputGPRUncertainty.load(args.gpr)
    gpr_mean_frame, gpr_std_frame = gpr_model.predict(embeddings)
    prediction_frame = pd.DataFrame(
        {
            target_name: gpr_mean_frame[target_name].to_numpy(dtype=float) * payload["target_stds"][target_name]
            + payload["target_means"][target_name]
            for target_name in TARGET_COLUMNS
        }
    )
    std_frame = pd.DataFrame(
        {
            target_name: gpr_std_frame[target_name].to_numpy(dtype=float) * payload["target_stds"][target_name]
            for target_name in TARGET_COLUMNS
        }
    )
    desirability_targets = tuple(
        DesirabilityTarget(
            name=target_name,
            mode=target_config["mode"],
            weight=float(target_config["weight"]),
            lower=target_config.get("lower"),
            target=target_config.get("target"),
            target_low=target_config.get("target_low"),
            target_high=target_config.get("target_high"),
            upper=target_config.get("upper"),
            curvature=float(target_config.get("curvature", 1.0)),
        )
        for target_name, target_config in config["desirability"].items()
    )
    selection_config = CandidateSelectionConfig(
        desirability_config=DesirabilityConfig(targets=desirability_targets),
        acquisition_config=AcquisitionConfig(
            desirability_weight=float(config["acquisition_weights"]["desirability"]),
            uncertainty_weight=float(config["acquisition_weights"]["uncertainty"]),
            diversity_weight=float(config["acquisition_weights"].get("diversity_penalty", 0.0)),
            vapor_pressure_cutoff=float(config["vapor_pressure_cutoff"]),
            desirability_cutoff=float(config["desirability_cutoff"]),
            uncertainty_cutoff=float(config["uncertainty_cutoff"]),
        ),
    )
    ranked = rank_candidates(
        candidate_df=processed_candidates,
        predictions=prediction_frame,
        stds=std_frame,
        descriptors=processed_candidates[DESCRIPTOR_COLUMNS].to_numpy(dtype=float),
        config=selection_config,
    )
    selected = select_top_batch(ranked, batch_size=int(config["batch_size"]))
    write_table(ranked, output_dir / "full_ranking.csv")
    write_table(selected, output_dir / "selected_batch.csv")
    summary = {
        "total_candidates": int(len(ranked)),
        "eligible_candidates": int(ranked["eligible"].sum()),
        "selected_candidates": int(len(selected)),
        "top_candidate": None if selected.empty else selected.iloc[0]["molecule_id"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("saved active-learning outputs to %s", output_dir)


if __name__ == "__main__":
    main()
