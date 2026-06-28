# Experimental Protocol

The demonstration protocol is designed for software validation rather than scientific benchmarking.

1. preprocess the raw table
2. split into train, validation, and test subsets with a fixed seed
3. train the GNN with a small default epoch count
4. fit Gaussian-process regressors on training embeddings
5. evaluate on the held-out split
6. rank candidate molecules with desirability and uncertainty

For real studies, replace the demonstration dataset with experimentally validated data and define domain-appropriate split strategies.

