from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess raw lubricant data into train, validation, and test splits.")
    parser.add_argument("--config", default="configs/preprocessing.yaml", help="Path to the preprocessing YAML config.")
    parser.add_argument("--input", default=None, help="Optional override for the raw input table path.")
    parser.add_argument("--output-dir", default=None, help="Optional override for the preprocessing output directory.")
    parser.add_argument(
        "--split-strategy",
        default=None,
        choices=["scaffold", "molecule_disjoint"],
        help="Optional override for the split strategy.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from leo_lubricants.data.preprocess import run_preprocessing
    from leo_lubricants.utils.config import load_config

    config = load_config(args.config)
    run_preprocessing(
        input_path=args.input or config["input_path"],
        output_dir=args.output_dir or config["output_dir"],
        split_strategy=args.split_strategy or config["split_strategy"],
        train_size=float(config["train_size"]),
        val_size=float(config["val_size"]),
        test_size=float(config["test_size"]),
        seed=int(config["seed"]),
        group_column=str(config.get("group_column", "inchikey")),
    )


if __name__ == "__main__":
    main()
