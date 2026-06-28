import pandas as pd

from leo_lubricants.features.descriptors import add_descriptors, compute_descriptors


def test_descriptor_generation_produces_named_fields() -> None:
    descriptor_row = compute_descriptors("CCO")
    assert descriptor_row["canonical_smiles"] == "CCO"
    assert descriptor_row["molecular_weight"] > 0.0
    assert len(descriptor_row["inchikey"]) > 0


def test_add_descriptors_appends_columns() -> None:
    frame = pd.DataFrame({"smiles": ["CCO"], "molecule_id": ["A"]})
    enriched = add_descriptors(frame)
    assert "logp" in enriched.columns
    assert "formal_charge" in enriched.columns

