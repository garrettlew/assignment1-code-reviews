"""CSE 242 Assignment 1 — submit as Assignment1.py.

Fill # YOUR ANSWER HERE blocks with commented reasoning; fill function bodies
at # YOUR CODE HERE. Keep signatures. See README.md and INTERFACE.md.
Functions must not mutate inputs, prompt for input, or perform network I/O.
Only code inside the main guard should save/show figures.
"""

import numpy as np
import matplotlib.pyplot as plt

# P3 Q2: Describe the Gaussian generator and observed correlation.
# YOUR ANSWER HERE
#  With 0 correlation, the dataset is an straight oval centered around the mean (because the deviation
#  for feature 2 is twice that of 1). As the correlation increases negatively or positively,
#  the oval skews diagonally and becomes thinner. This happens because as the two features become
#  more correlated, they pin down each others values to a smaller range. See the pdf for the plots.

def generate_data(n_samples=500, seed=0, mean=None, covariance=None):
    """Return reproducible Gaussian samples (n_samples,2) using NumPy and seed.
    """
    # YOUR CODE HERE
    rng = np.random.default_rng(1)

    # Center the mean of the two features at the origin
    mean = np.array([0, 0])

    # standard deviation of the two features, feature 2 is twice as spread out
    sigma1, sigma2 = 1.0, 2.0

    # Correlation coefficient, positive means when one goes up the other goes up
    rho = 0.8

    # Covariance matrix
    cov = np.array([
        [sigma1 ** 2, rho * sigma1 * sigma2],
        [rho * sigma1 * sigma2, sigma2 ** 2]
    ])

    n = 200

    datasets = {}
    for rho in [-0.8, 0.0, 0.5, 0.8, 0.95]:
        cov = np.array([[sigma1 ** 2, rho * sigma1 * sigma2],
                        [rho * sigma1 * sigma2, sigma2 ** 2]])
        datasets[rho] = rng.multivariate_normal(mean, cov, size=n)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)

    lim = 8

    for ax, (rho, X) in zip(axes, datasets.items()):
        ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
        ax.set_title(f"ρ = {rho}")
        ax.set_xlabel("Feature 1")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")

    axes[0].set_ylabel("Feature 2")
    plt.tight_layout()
    plt.show()

# P3 Q3: Link centering/covariance/eigendecomposition to the derivation and explain the plot.
# YOUR ANSWER HERE
def pca(X=None):
    """
    Links to the Q1 derivation:
        Var(u^T X) = u^T Sigma u
        maximize u^T Sigma u  subject to  u^T u = 1
        -> Lagrangian gives  Sigma u = lambda u
        -> optimal u are eigenvectors of Sigma, and the variance along each equals its eigenvalue.
    """

    # Dataset
    if not X:
        rng = np.random.default_rng(42)
        true_cov = np.array([[3.0, 2.0],
                             [2.0, 2.0]])  # positive covariance -> tilted ellipse
        X = rng.multivariate_normal(mean=[5.0, 3.0], cov=true_cov, size=300)  # shape (n, p)
    n, p = X.shape

    # ---------------------------------------------------------------
    # STEP 1: Center the data (subtract the mean of each column)
    # Q1 link: the derivation works with (X - mu). Centering makes the
    # data cloud sit at the origin so projections measure deviation from the mean.
    # ---------------------------------------------------------------
    mu = X.mean(axis=0)
    X_centered = X - mu

    # ---------------------------------------------------------------
    # STEP 2: Covariance matrix
    # Q1 link: Sigma = E[(X - mu)(X - mu)^T]. With n samples (rows) this is
    # the average of outer products, written compactly as Xc^T Xc / (n - 1).
    # ---------------------------------------------------------------
    cov = (X_centered.T @ X_centered) / (n - 1)

    # ---------------------------------------------------------------
    # STEP 3: Eigenvalues and eigenvectors
    # Q1 link: maximizing u^T Sigma u with u^T u = 1 gives Sigma u = lambda u.
    # eigh is for symmetric matrices (covariance is symmetric); it returns
    # eigenvalues in ascending order and eigenvectors as COLUMNS.
    # ---------------------------------------------------------------
    eigvals, eigvecs = np.linalg.eigh(cov)

    # ---------------------------------------------------------------
    # STEP 4: Sort by eigenvalue (largest first) and select top k components
    # Q1 link: variance along eigenvector v_i is v_i^T Sigma v_i = lambda_i,
    # so the biggest eigenvalue = direction with the most spread = PC1.
    # ---------------------------------------------------------------
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    k = 1
    components = eigvecs[:, :k]  # shape (p, k)
    explained_ratio = eigvals / eigvals.sum()

    # Project the centered data onto the chosen components
    X_projected = X_centered @ components  # shape (n, k)

    # ---------------------------------------------------------------
    # Verify the Q1 result numerically: Var(u^T X) = u^T Sigma u = lambda
    # ---------------------------------------------------------------
    for i in range(p):
        u = eigvecs[:, i]
        proj = X_centered @ u
        print(f"PC{i + 1}: eigenvalue = {eigvals[i]:.4f} | "
              f"u^T Sigma u = {u @ cov @ u:.4f} | "
              f"sample var of projection = {proj.var(ddof=1):.4f} | "
              f"explained = {explained_ratio[i]:.1%}")
    print("\nEigenvectors (columns = PC1, PC2):\n", eigvecs)

    # ---------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) original data with principal component directions
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], alpha=0.35, s=14, label="data")
    ax.scatter(*mu, color="black", zorder=5, label="mean")
    colors = ["crimson", "darkorange"]
    for i in range(p):
        # arrow length = 2 * sqrt(eigenvalue) = 2 standard deviations along that PC
        vec = 2 * np.sqrt(eigvals[i]) * eigvecs[:, i]
        ax.annotate("", xy=mu + vec, xytext=mu,
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=2.5))
        ax.plot([], [], color=colors[i], lw=2.5,
                label=f"PC{i + 1} ({explained_ratio[i]:.0%} of variance)")
    ax.set_title("Original data with principal component directions")
    ax.set_xlabel("Measurement 1")
    ax.set_ylabel("Measurement 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) data expressed in PC coordinates (rotated so PCs are the axes)
    ax = axes[1]
    scores = X_centered @ eigvecs  # all components, for plotting
    ax.scatter(scores[:, 0], scores[:, 1], alpha=0.35, s=14)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_title("Data in principal component coordinates")
    ax.set_xlabel("PC1 score")
    ax.set_ylabel("PC2 score")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# P3 Q4:  Compute the proportion of total variance explained by each principal component.
# YOUR ANSWER HERE

# PC1: eigenvalue = 4.5485 | u^T Sigma u = 4.5485 | sample var of projection = 4.5485 | explained = 92.2%
# PC2: eigenvalue = 0.3873 | u^T Sigma u = 0.3873 | sample var of projection = 0.3873 | explained = 7.8%
#
# Eigenvectors (columns = PC1, PC2):
#  [[-0.81654063  0.57728796]
#  [-0.57728796 -0.81654063]]
#
### --------


# P3 Q4: Interpret the variance fractions and dimensionality reduction tradeoff.
# YOUR ANSWER HERE
# PC1 explains 92.2% of the total variance and PC2 only 7.8%, because the strong
# correlation between the two measurements makes the data a thin, tilted ellipse
# that one direction describes well. Dimensionality reduction exploits this redundancy
# to compress data, reduce noise, and simplify visualization and modeling.
# The trade-off is that each dropped component saves a dimension but loses variance
# equal to its eigenvalue: keeping only PC1 halves the dimensions at a cost of 7.8% of
# the variance, which is worthwhile if that loss is tolerable.

def explained_variance(eigenvalues):
    """Return same-shape fractions for nonnegative eigenvalues with positive sum."""
    # YOUR CODE HERE
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if not np.all(eigenvalues >= 0):
        raise ValueError("eigenvalues must be nonnegative")
    total = eigenvalues.sum()
    if total <= 0:
        raise ValueError("eigenvalues must have a positive sum")
    return eigenvalues / total


if __name__ == '__main__':
    # YOUR CODE HERE: call your functions and save/display the requested figures.
    # This block does not run when the grader imports your functions.
    generate_data()
    pca()
