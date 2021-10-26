import os
import jax.numpy as np
from tensorflow_probability.substrates.jax.distributions import (
    MultivariateNormalFullCovariance as normal,
    kl_divergence,
)


def gaussian_precision_kl(theta, theta_hat):
    """Computes KL divergence between Gaussians with precisions theta, theta_hat

    Assumes mean zero for the Gaussians.

    Args:
        theta: d x d precision matrix
        theta_hat: d x d precision matrix
    Returns:
        kl: KL divergence
    """
    dim = np.shape(theta)[0]
    theta_0_logdet = np.linalg.slogdet(theta)[1]
    theta_hat_logdet = np.linalg.slogdet(theta_hat)[1]

    # Not sure how stable this term is going to be :/
    trace_term = np.trace(theta_hat @ np.linalg.inv(theta))
    return 0.5 * (theta_0_logdet - theta_hat_logdet - dim + trace_term)


def gaussian_precision_wasserstein(theta, theta_hat):
    """Computes Wasserstein distance between Gaussians with precisions theta, theta_hat

    Assumes mean zero for the Gaussians.
    Horribly hacky implementation at the moment...

    Args:
        theta: d x d precision matrix
        theta_hat: d x d precision matrix
    Returns:
        kl: KL divergence
    """
    return gaussian_square_wasserstein(np.linalg.inv(theta), np.linalg.inv(theta_hat))


def precision_wasserstein_loss(true_noise, true_W, est_noise, est_W):
    """Computes the wasserstein loss to the true parameters
    Args:
        true_noise: (d,)-shape vector of noise variances
        true_W: (d,d)-shape adjacency matrix
        est_noise: (d,)-shape vector of estimated noise variances
        est_W: (d,d)-shape adjacency matrix
    Returns:
        w_square: induced squared Wasserstein distance
    """
    dim = len(true_noise)
    true_cov = (
        (np.eye(dim) - true_W) @ (np.diag(1.0 / true_noise)) @ (np.eye(dim) - true_W).T
    )
    est_cov = (
        (np.eye(dim) - est_W) @ (np.diag(1.0 / est_noise)) @ (np.eye(dim) - est_W).T
    )
    return gaussian_precision_wasserstein(true_cov, est_cov)


def precision_wasserstein_sample_loss(x_prec, est_noise, est_W):
    """Computes the wasserstein loss to the true parameters


    Args:
        true_noise: (d,)-shape vector of noise variances
        true_W: (d,d)-shape adjacency matrix
        est_noise: (d,)-shape vector of estimated noise variances
    Returns:
        w_square: induced squared Wasserstein distance
    """
    dim = len(est_noise)
    est_prec = (
        (np.eye(dim) - est_W) @ (np.diag(1.0 / est_noise)) @ (np.eye(dim) - est_W).T
    )
    return gaussian_precision_wasserstein(x_prec, est_prec)


def gaussian_kl(theta, theta_hat):
    """Computes KL divergence between Gaussians with covariances theta, theta_hat

    Assumes mean zero for the Gaussians.

    Args:
        theta: d x d covariance matrix
        theta_hat: d x d covariance matrix
    Returns:
        kl: KL divergence
    """
    dim, _ = np.shape(theta)
    theta_dist = normal(loc=np.zeros(dim), covariance_matrix=theta)
    theta_hat_dist = normal(loc=np.zeros(dim), covariance_matrix=theta_hat)
    kl = kl_divergence(theta_dist, theta_hat_dist)
    return kl


def precision_kl_loss(true_noise, true_W, est_noise, est_W):
    """Computes the KL divergence to the true parameters


    Args:
        true_noise: (d,)-shape vector of noise variances
        true_W: (d,d)-shape adjacency matrix
        est_noise: (d,)-shape vector of estimated noise variances
        est_W: (d,d)-shape adjacency matrix
    Returns:
        w_square: induced squared Wasserstein distance
    """
    dim = len(true_noise)
    true_prec = (
        (np.eye(dim) - true_W) @ (np.diag(1.0 / true_noise)) @ (np.eye(dim) - true_W).T
    )
    est_prec = (
        (np.eye(dim) - est_W) @ (np.diag(1.0 / est_noise)) @ (np.eye(dim) - est_W).T
    )
    return gaussian_precision_kl(true_prec, est_prec)


def precision_kl_sample_loss(x_prec, est_noise, est_W):
    """Computes the KL divergence to the sample distribution

    Args:
        x_cov: (d,d)-shape sample covariance matrix
        est_noise: (d,)-shape vector of estimated noise variances
        est_W: (d,d)-shape estimated adjacency matrix
    Returns:
        w_square: induced KL divergence
    """
    dim = len(est_noise)
    est_prec = (
        (np.eye(dim) - est_W) @ (np.diag(1.0 / est_noise)) @ (np.eye(dim) - est_W).T
    )
    return gaussian_precision_kl(x_prec, est_prec)


def my_sqrtm(X):
    """Computes the matrix square root of X

    Does this by diagonalizing X.
    The method in scipy is probably more stable, not
    requiring a matrix inverse, but not implemented at the
    moment. Can probably also not use the matrix inverse
    and use solve instead.
    Args:
        X: An n x n symmetric and PSD matrix
    Returns:
        sqrt_X: An n x n matrix such that sqrt_X @ sqrt_X = X
    """
    vals, vectors = np.linalg.eigh(X)

    # We have vectors.inv @ X @ vectors = sqrt(vals)^2
    # so X = vectors @ sqrt(vals)^2 @ vectors.inv
    # X = (vectors @ sqrt(vals) @ vectors.inv)^2

    return vectors @ np.diag(np.sqrt(vals)) @ np.linalg.inv(vectors)


def gaussian_square_wasserstein(theta, theta_hat):
    """Computes square of Wasserstein distance between Gaussians

    Assumes mean zero for the Gaussians.

    Args:
        theta: d x d covariance matrix
        theta_hat: d x d covariance matrix
    Returns:
        dist: Wasserstein_2 distance
    """
    T_sqrt = my_sqrtm(theta)
    inner_sqrt = my_sqrtm(T_sqrt @ theta_hat @ T_sqrt)
    return np.trace(theta + theta_hat - 2 * inner_sqrt)


def wasserstein_loss(true_noise, true_W, est_noise, est_W):
    """Computes the wasserstein loss to the true parameters


    Args:
        true_noise: (d,)-shape vector of noise variances
        true_W: (d,d)-shape adjacency matrix
        est_noise: (d,)-shape vector of estimated noise variances
        est_W: (d,d)-shape adjacency matrix
    Returns:
        w_square: induced squared Wasserstein distance
    """
    dim = len(true_noise)
    true_cov = (np.eye(dim) + true_W) @ np.diag(true_noise) @ (np.eye(dim) + true_W).T
    est_cov = (np.eye(dim) + est_W) @ np.diag(est_noise) @ (np.eye(dim) + est_W).T
    return gaussian_square_wasserstein(true_cov, est_cov)


def wasserstein_sample_loss(x_cov, est_noise, est_W):
    """Computes the wasserstein loss to the true parameters


    Args:
        true_noise: (d,)-shape vector of noise variances
        true_W: (d,d)-shape adjacency matrix
        est_noise: (d,)-shape vector of estimated noise variances
    Returns:
        w_square: induced squared Wasserstein distance
    """
    dim = len(est_noise)
    est_cov = (np.eye(dim) + est_W) @ np.diag(est_noise) @ (np.eye(dim) + est_W).T
    return gaussian_square_wasserstein(x_cov, est_cov)


def kl_loss(true_noise, true_W, est_noise, est_W):
    """Computes the KL divergence to the true parameters


    Args:
        true_noise: (d,)-shape vector of noise variances
        true_W: (d,d)-shape adjacency matrix
        est_noise: (d,)-shape vector of estimated noise variances
        est_W: (d,d)-shape adjacency matrix
    Returns:
        w_square: induced squared Wasserstein distance
    """
    dim = len(true_noise)
    true_cov = (np.eye(dim) + true_W) @ np.diag(true_noise) @ (np.eye(dim) + true_W).T
    est_cov = (np.eye(dim) + est_W) @ np.diag(est_noise) @ (np.eye(dim) + est_W).T
    return gaussian_kl(true_cov, est_cov)


def kl_sample_loss(x_cov, est_noise, est_W):
    """Computes the KL divergence to the sample distribution

    Args:
        x_cov: (d,d)-shape sample covariance matrix
        est_noise: (d,)-shape vector of estimated noise variances
        est_W: (d,d)-shape estimated adjacency matrix
    Returns:
        w_square: induced KL divergence
    """
    dim = len(est_noise)
    est_cov = (np.eye(dim) + est_W) @ np.diag(est_noise) @ (np.eye(dim) + est_W).T
    return gaussian_kl(x_cov, est_cov)
