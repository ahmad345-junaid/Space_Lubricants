# Methodology

The repository uses a multi-stage workflow:

1. preprocess molecular and environment-conditioned data
2. train a dense message-passing graph neural network
3. extract latent embeddings from the trained model
4. fit per-target Gaussian-process regressors on latent embeddings
5. use predictive means and uncertainties for active learning

The GNN consumes molecular graph structure together with engineered environment features so predictions can depend jointly on chemistry and operating context.

