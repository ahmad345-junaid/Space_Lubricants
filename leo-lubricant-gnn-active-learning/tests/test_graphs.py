from leo_lubricants.data.molecular_graphs import batch_graphs, smiles_to_graph


def test_graph_tensor_shapes_are_consistent() -> None:
    graph_a = smiles_to_graph("CCO")
    graph_b = smiles_to_graph("c1ccccc1")
    batch = batch_graphs([graph_a, graph_b])
    assert graph_a.node_features.ndim == 2
    assert graph_a.edge_index.shape[0] == 2
    assert batch["node_features"].shape[0] == graph_a.num_nodes + graph_b.num_nodes
    assert batch["graph_ptr"].shape[0] == 3
