from __future__ import annotations

from dataclasses import dataclass

import torch


def _require_rdkit():
    try:
        from rdkit import Chem
    except ImportError as error:
        raise ImportError(
            "RDKit is required for molecular graph construction. Install it with "
            "`conda install -c conda-forge rdkit` or `pip install rdkit`."
        ) from error
    return Chem


@dataclass
class MolecularGraph:
    smiles: str
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


def atom_features(atom) -> list[float]:
    Chem = _require_rdkit()
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    hybridization_one_hot = [1.0 if atom.GetHybridization() == value else 0.0 for value in hybridizations]
    hybridization_one_hot.append(float(sum(hybridization_one_hot) == 0.0))
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        *hybridization_one_hot,
        float(atom.GetIsAromatic()),
        float(atom.GetTotalNumHs(includeNeighbors=True)),
        float(atom.IsInRing()),
        float(atom.GetMass()),
    ]


def bond_features(bond) -> list[float]:
    Chem = _require_rdkit()
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    type_one_hot = [1.0 if bond.GetBondType() == value else 0.0 for value in bond_types]
    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    stereo_one_hot = [1.0 if bond.GetStereo() == value else 0.0 for value in stereo_types]
    return [
        *type_one_hot,
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        *stereo_one_hot,
    ]


def smiles_to_graph(smiles: str) -> MolecularGraph:
    Chem = _require_rdkit()
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    canonical = str(Chem.MolToSmiles(molecule, canonical=True))
    node_feature_tensor = torch.tensor(
        [atom_features(atom) for atom in molecule.GetAtoms()],
        dtype=torch.float32,
    )
    edge_pairs: list[list[int]] = []
    edge_feature_rows: list[list[float]] = []
    for bond in molecule.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_features(bond)
        edge_pairs.append([start, end])
        edge_pairs.append([end, start])
        edge_feature_rows.append(features)
        edge_feature_rows.append(features)
    if edge_pairs:
        edge_index_tensor = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_feature_tensor = torch.tensor(edge_feature_rows, dtype=torch.float32)
    else:
        edge_index_tensor = torch.empty((2, 0), dtype=torch.long)
        edge_feature_tensor = torch.empty((0, 10), dtype=torch.float32)
    return MolecularGraph(
        smiles=canonical,
        node_features=node_feature_tensor,
        edge_index=edge_index_tensor,
        edge_features=edge_feature_tensor,
    )


def batch_graphs(graphs: list[MolecularGraph]) -> dict[str, torch.Tensor]:
    if not graphs:
        raise ValueError("At least one graph is required for batching.")
    node_blocks: list[torch.Tensor] = []
    edge_blocks: list[torch.Tensor] = []
    edge_index_blocks: list[torch.Tensor] = []
    graph_index_blocks: list[torch.Tensor] = []
    graph_ptr = [0]
    node_offset = 0
    for graph_id, graph in enumerate(graphs):
        node_blocks.append(graph.node_features)
        edge_blocks.append(graph.edge_features)
        edge_index_blocks.append(graph.edge_index + node_offset)
        graph_index_blocks.append(torch.full((graph.num_nodes,), graph_id, dtype=torch.long))
        node_offset += graph.num_nodes
        graph_ptr.append(node_offset)
    return {
        "node_features": torch.cat(node_blocks, dim=0),
        "edge_index": torch.cat(edge_index_blocks, dim=1) if edge_index_blocks else torch.empty((2, 0), dtype=torch.long),
        "edge_features": torch.cat(edge_blocks, dim=0) if edge_blocks else torch.empty((0, 10), dtype=torch.float32),
        "graph_index": torch.cat(graph_index_blocks, dim=0),
        "graph_ptr": torch.tensor(graph_ptr, dtype=torch.long),
    }

