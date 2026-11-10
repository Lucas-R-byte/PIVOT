# PIVOT selected algorithms

This supplement provides a small JAX reference excerpt for the gated residual (Eq. 4), empirical quantile and masked mean used in sample CVaR (Eqs. 8–10), and bounded sigmoid head (Eq. 12). The fusion and bounded-head functions extract the relevant branch of the research implementation. The CVaR helpers are direct excerpts. The objective wrapper expresses Eq. 10.

Run with Python 3.11, JAX and NumPy. Functions accept arrays. The bounded head expects broadcast-compatible bounds in the same normalized price space and lower <= upper.

The package covers these operators only. The full training and benchmark pipeline, dataset, model checkpoints, and experiment logs are outside this release. The files cannot regenerate the reported tables or figures.
