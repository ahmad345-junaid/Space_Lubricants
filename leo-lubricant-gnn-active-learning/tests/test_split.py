import pandas as pd

from leo_lubricants.data.split import molecule_disjoint_split


def test_molecule_disjoint_split_prevents_leakage() -> None:
    frame = pd.DataFrame(
        {
            "molecule_id": ["A1", "A2", "B1", "B2", "C1", "C2"],
            "smiles": ["CCO", "CCO", "CCC", "CCC", "c1ccccc1", "c1ccccc1"],
            "inchikey": ["KA", "KA", "KB", "KB", "KC", "KC"],
        }
    )
    splits = molecule_disjoint_split(
        frame,
        train_size=0.5,
        val_size=0.25,
        test_size=0.25,
        seed=5,
        group_column="inchikey",
    )
    split_keys = {name: set(split["inchikey"]) for name, split in splits.items()}
    assert split_keys["train"].isdisjoint(split_keys["val"])
    assert split_keys["train"].isdisjoint(split_keys["test"])
    assert split_keys["val"].isdisjoint(split_keys["test"])

