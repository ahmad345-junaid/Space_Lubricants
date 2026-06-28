from __future__ import annotations

from typing import Any

import pandas as pd


def _require_rdkit() -> Any:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen
        from rdkit.Chem import Descriptors
        from rdkit.Chem import rdMolDescriptors
    except ImportError as error:
        raise ImportError(
            "RDKit is required for descriptor generation. Install it with "
            "`conda install -c conda-forge rdkit` or `pip install rdkit`."
        ) from error
    return Chem, Crippen, Descriptors, rdMolDescriptors


def canonicalize_smiles(smiles: str) -> str:
    Chem, _, _, _ = _require_rdkit()
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return str(Chem.MolToSmiles(molecule, canonical=True))


def compute_descriptors(smiles: str) -> dict[str, Any]:
    Chem, Crippen, Descriptors, rdMolDescriptors = _require_rdkit()
    canonical_smiles = canonicalize_smiles(smiles)
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        raise ValueError(f"Unable to parse canonical SMILES string: {smiles}")
    return {
        "canonical_smiles": canonical_smiles,
        "inchikey": str(Chem.MolToInchiKey(molecule)),
        "molecular_weight": float(Descriptors.MolWt(molecule)),
        "logp": float(Crippen.MolLogP(molecule)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
        "rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(molecule)),
        "heavy_atoms": float(rdMolDescriptors.CalcNumHeavyAtoms(molecule)),
        "h_bond_donors": float(rdMolDescriptors.CalcNumHBD(molecule)),
        "h_bond_acceptors": float(rdMolDescriptors.CalcNumHBA(molecule)),
        "ring_count": float(rdMolDescriptors.CalcNumRings(molecule)),
        "aromatic_ring_count": float(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(molecule)),
        "fluorine_count": float(sum(atom.GetAtomicNum() == 9 for atom in molecule.GetAtoms())),
        "silicon_count": float(sum(atom.GetAtomicNum() == 14 for atom in molecule.GetAtoms())),
        "heteroatom_count": float(rdMolDescriptors.CalcNumHeteroatoms(molecule)),
        "formal_charge": float(Chem.GetFormalCharge(molecule)),
    }


def add_descriptors(df: pd.DataFrame, smiles_column: str = "smiles") -> pd.DataFrame:
    if smiles_column not in df.columns:
        raise ValueError(f"Missing SMILES column: {smiles_column}")
    descriptor_rows = [compute_descriptors(smiles) for smiles in df[smiles_column].astype(str)]
    descriptor_frame = pd.DataFrame(descriptor_rows, index=df.index)
    augmented = df.copy()
    for column in descriptor_frame.columns:
        augmented[column] = descriptor_frame[column]
    return augmented


def add_descriptor_columns(frame: pd.DataFrame, smiles_column: str = "smiles") -> pd.DataFrame:
    return add_descriptors(frame, smiles_column=smiles_column)

