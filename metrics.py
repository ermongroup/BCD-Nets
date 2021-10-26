import jax.numpy as jnp
from dag_utils import SyntheticDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import jax.numpy as jnp
from scipy.stats import wasserstein_distance
from typing import cast


def intervention_distance(
    W_true,
    W_est,
    sigmas_true,
    sigmas_est,
    sem_type="linear-gauss",
    threshold=0.3,
    n_repeats=10,
):
    """Calculate the intervention distance between two estimated graphs.

    We define the intervention distance by taking a random edge (X_i -> X_j) in the true
    graph, then forming P(X_j|do(X_i)). We then form P(X_j|do(X_i)) from an estimated graph
    W_est and estimated noises sigmas_est

    Args:
        W_true: True causal graph structure
        W_est: Estimated causal graph structure
        sigmas_true: True process noise on the nodes
        sigmas_est: Estimated process noise on the nodes
    Returns:
        KL divergence between P(X_j|do(X_i)) and P_est(X_j|do(X_i))
    """
    if sigmas_est.shape == ():
        noise_dim = 1
    else:
        noise_dim = len(sigmas_est)
    if noise_dim == 1:
        dim = len(W_true)
        sigmas_est = sigmas_est * jnp.ones(dim)
    total_kl = 0.0
    total_wass = 0.0
    for _ in range(n_repeats):
        W_est = np.array(jnp.where(jnp.abs(W_est) > threshold, W_est, 0.0))
        true_n = 1_000
        est_n = 1_000
        intervention = SyntheticDataset.intervene_sem
        i_idxs, j_idxs = np.where(W_true != 0)
        idx = np.random.randint(low=0, high=len(i_idxs))
        from_idx = i_idxs[idx]
        to_idx = j_idxs[idx]
        fix_val = np.random.choice([-1, 1]) * np.random.uniform(low=0.5, high=2)
        out_xs = intervention(
            W_true,
            true_n,
            sem_type,
            sigmas_true,
            idx_to_fix=from_idx,
            value_to_fix=fix_val,
        )
        true_x_j_dist = out_xs[:, to_idx]
        est_xs = intervention(
            W_est,
            est_n,
            sem_type,
            sigmas_est,
            idx_to_fix=from_idx,
            value_to_fix=fix_val,
        )
        est_x_j_dist = est_xs[:, to_idx]

        # First we compute p(u)
        kde_true, kde_est = KernelDensity(), KernelDensity()
        kde_true.fit(true_x_j_dist.reshape(-1, 1))
        true_samples = kde_true.sample(1_000)
        true_samples = cast(np.ndarray, true_samples)
        true_logprobs = kde_true.score_samples(true_samples.reshape((-1, 1)))
        kde_est.fit(est_x_j_dist.reshape(-1, 1))
        est_logprobs = kde_est.score_samples(true_samples.reshape((-1, 1)))
        est_kl = (true_logprobs - est_logprobs).mean()
        total_kl += est_kl
        total_wass += wasserstein_distance(est_x_j_dist, true_x_j_dist)
    return total_wass / n_repeats


def ensemble_intervention_distance(
    W_true,
    W_ests,
    sigmas_true,
    sigmas_ests,
    sem_type="linear-gauss",
    threshold=0.3,
    n_repeats=10,
):
    """Calculate the intervention distance between two estimated graphs.

    We define the intervention distance by taking a random edge (X_i -> X_j) in the true
    graph, then forming P(X_j|do(X_i)). We then form P(X_j|do(X_i)) from an estimated graph
    W_est and estimated noises sigmas_est

    Args:
        W_true: True causal graph structure
        W_est: Estimated causal graph structure
        sigmas_true: True process noise on the nodes
        sigmas_est: Estimated process noise on the nodes
    Returns:
        KL divergence between P(X_j|do(X_i)) and P_est(X_j|do(X_i))
    """
    total_kl = 0.0
    total_wass = 0.0
    num_Ws, dim, _ = W_ests.shape
    if sigmas_ests.shape == () or len(sigmas_ests.shape) == 1:
        noise_dim = 1
        sigmas_ests = sigmas_ests[..., None]
    else:
        _, noise_dim = sigmas_ests.shape
    if noise_dim == 1:
        sigmas_ests = sigmas_ests * jnp.ones((num_Ws, dim))
    for _ in range(n_repeats):
        true_n = 10_000
        num_per_W = true_n // num_Ws

        intervention = SyntheticDataset.intervene_sem
        i_idxs, j_idxs = np.where(W_true != 0)
        idx = np.random.randint(low=0, high=len(i_idxs))
        from_idx = i_idxs[idx]
        to_idx = j_idxs[idx]
        fix_val = np.random.choice([-1, 1]) * np.random.uniform(low=0.5, high=2)
        out_xs = intervention(
            W_true,
            true_n,
            sem_type,
            sigmas_true,
            idx_to_fix=from_idx,
            value_to_fix=fix_val,
        )
        true_x_j_dist = out_xs[:, to_idx]
        est_x_j_dist = []
        for i in range(len(W_ests)):
            W_est = W_ests[i]
            W_est = np.array(jnp.where(jnp.abs(W_est) > threshold, W_est, 0.0))
            sigmas_est = sigmas_ests[i]

            _est_xs = intervention(
                W_est,
                num_per_W,
                sem_type,
                sigmas_est,
                idx_to_fix=from_idx,
                value_to_fix=fix_val,
            )
            _est_x_j_dist = _est_xs[:, to_idx]
            est_x_j_dist.append(_est_x_j_dist)
        est_x_j_dist = np.stack(est_x_j_dist).flatten()

        # First we compute p(u)
        kde_true, kde_est = KernelDensity(), KernelDensity()
        kde_true.fit(true_x_j_dist.reshape(-1, 1))
        true_samples = kde_true.sample(1_000)
        true_samples = cast(np.ndarray, true_samples)
        true_logprobs = kde_true.score_samples(true_samples.reshape((-1, 1)))
        kde_est.fit(est_x_j_dist.reshape(-1, 1))
        est_logprobs = kde_est.score_samples(true_samples.reshape((-1, 1)))
        est_kl = (true_logprobs - est_logprobs).mean()
        total_kl += est_kl
        total_wass += wasserstein_distance(est_x_j_dist, true_x_j_dist)

    return total_wass / n_repeats
