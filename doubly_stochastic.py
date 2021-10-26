import jax.numpy as np
import jax.random as rnd
from jax import lax, vmap
from jax.scipy.special import logsumexp
from utils import npperm

import numpy as onp
from typing import Type, Union
from hungarian_callback import hungarian, batched_hungarian
from implicit_sinkhorn import while_uv_sinkhorn, while_uv_sinkhorn_debug

Tensor = Union[onp.ndarray, np.ndarray]
PRNGKey = Type[np.ndarray]


class GumbelSinkhorn:
    def __init__(self, dim, noise_type="gumbel", tol=1e-2):
        """Makes a class which gives a doubly stochastic matrix
        and exposes methods for returning the matrix"""
        self.dim = dim
        self.param_shape = (dim * dim,)
        if noise_type == "gumbel":
            self.noise = rnd.gumbel
        elif noise_type == "normal":
            self.noise = rnd.normal
        else:
            raise NotImplementedError(f"{noise_type} noise not implemented")
        self.euler_mascheroni = 0.577215664901532
        self.while_sinkhorn = lambda x: while_uv_sinkhorn(x, tol)
        self.while_sinkhorn_debug = lambda x: while_uv_sinkhorn_debug(x, tol)

    def bethe_permanent_gamma(self, M, n_iters=1, eps=1e-20):
        N = M.shape[0]
        logV1 = onp.log((1 / N) * onp.ones((N, N)))
        logV2 = M - logsumexp(M, axis=1, keepdims=True)

        def scan_inner(carry, _):
            logV1, logV2 = carry
            logexpV2 = np.log(-np.expm1(logV2) + eps)
            HelpMat = logV2 + logexpV2
            HelpMat = HelpMat - np.log(-np.expm1(logV2) + eps)
            logV1 = HelpMat - logsumexp(HelpMat, 0, keepdims=True)
            HelpMat = logV1 + logexpV2
            HelpMat = HelpMat - np.log(-np.expm1(logV1) + eps)
            logV2 = HelpMat - logsumexp(HelpMat, 1, keepdims=True)
            return (logV1, logV2), None

        (logV1, _), _ = lax.scan(scan_inner, (logV1, logV2), None, length=n_iters)
        return np.exp(logV1)

    def sample_hard(self, params, tau, rng_key):
        X = params.reshape(self.dim, self.dim)
        gumbels = self.noise(rng_key, shape=X.shape)
        perturbed_X = X + gumbels
        # soft_samples = pure_sinkhorn(perturbed_X, tau, n_steps)
        soft_samples = self.while_sinkhorn(perturbed_X / tau)
        # Convenient way to get it to use the straight-through gradient estimator
        return lax.stop_gradient(hungarian(-soft_samples) - soft_samples) + soft_samples

    def sample_hard_batched(self, params, tau, rng_key, n_batch):
        # Samples a big batch of samples so that we can use the scipy
        # hungarian implementation
        X = np.tile(params.reshape(self.dim, self.dim), (n_batch, 1, 1))
        gumbels = self.noise(rng_key, shape=X.shape)
        perturbed_X = X + gumbels
        # soft_samples = vmap(pure_sinkhorn, in_axes=(0, None, None))(
        #     perturbed_X, tau, n_steps
        # )
        soft_samples = vmap(self.while_sinkhorn)(perturbed_X / tau)
        hard_samples = batched_hungarian(lax.stop_gradient(-soft_samples))
        return lax.stop_gradient(hard_samples - soft_samples) + soft_samples

    def sample_hard_batched_logits(self, params, tau, rng_key):
        """Takes batch of different logits, returns hard batch"""
        gumbels = self.noise(rng_key, shape=params.shape)
        perturbed_X = params + gumbels
        soft_samples = vmap(self.while_sinkhorn)(perturbed_X / tau)
        hard_samples = batched_hungarian(lax.stop_gradient(-soft_samples))
        return lax.stop_gradient(hard_samples - soft_samples) + soft_samples

    def sample_hard_batched_logits_debug(self, params, tau, rng_key):
        """Takes batch of different logits, returns hard batch"""
        gumbels = self.noise(rng_key, shape=params.shape)
        perturbed_X = params + gumbels
        soft_samples, errors = vmap(self.while_sinkhorn_debug)(perturbed_X / tau)
        hard_samples = batched_hungarian(lax.stop_gradient(-soft_samples))
        return lax.stop_gradient(hard_samples - soft_samples) + soft_samples, errors

    def sample_soft_batched_logits(self, params, tau, rng_key):
        """Takes batch of different logits, returns hard batch"""
        gumbels = self.noise(rng_key, shape=params.shape)
        perturbed_X = params + gumbels
        # soft_samples = vmap(pure_sinkhorn, in_axes=(0, None, None))(
        #     perturbed_X, tau, n_steps
        # )
        soft_samples = vmap(self.while_sinkhorn)(perturbed_X / tau)
        return soft_samples

    def sample_soft(self, params, tau, rng_key):
        X = params.reshape(self.dim, self.dim)
        gumbels = self.noise(rng_key, shape=X.shape)
        perturbed_X = X + gumbels
        # return pure_sinkhorn(perturbed_X, tau, n_steps)
        return self.while_sinkhorn(perturbed_X / tau)

    def rand_init(self, rng_key, init_std):
        return rnd.normal(rng_key, shape=self.param_shape) * init_std

    def logprob(self, sample, params, n_iters):
        """Return the (approximate) log probability of the (approximate) permutation we get.
        The probability p(sample|params) propto exp(frobenius_norm(sample @ params)).
        We get the normalization factor using the matrix permanent."""
        eps = 1e-20
        params = params.reshape(self.dim, self.dim)
        unnormalized_logprob = np.sum(sample * params)

        # Approximate the matrix permanent of exp(params) with Bethe permanent of exp (params)
        # In the indigo code there is an additional division by the temperature in here...
        gamma_matrix = self.bethe_permanent_gamma(params, n_iters, 1e-20)
        term_1 = np.sum(params * gamma_matrix)
        term_2 = -np.sum(gamma_matrix * np.log(gamma_matrix + eps))
        term_3 = np.sum((1 - gamma_matrix) * np.log(1 - gamma_matrix + eps))
        log_approx_perm = term_1 + term_2 + term_3
        return unnormalized_logprob - log_approx_perm

    def exact_logprob(self, sample, params):
        params = params.reshape(self.dim, self.dim)
        unnormalized_logprob = np.sum(sample * params)

        # Compute the exact partition function, equal to perm(exp(X))
        # Note this has complexity ~ n^2 2^n, so completely intractable for
        # n > 16, also might have some overflow/underflow issues...
        perm = npperm(np.exp(params))
        return unnormalized_logprob - np.log(perm)

    def kl(self, params, tau, tau_prior):
        """Get the KL between GS(X, tau), GS(0, tau_prior)"""
        N_2 = len(params)  # N squared
        S_1 = np.sum(params) * tau_prior / tau
        S_2 = np.sum(np.exp(-params * (tau_prior / tau)))
        return (
            N_2
            * (
                np.log(tau / tau_prior)
                - 1
                + self.euler_mascheroni * (tau_prior / tau - 1)
            )
            + S_1
            + np.exp(lax.lgamma(1 + tau_prior / tau)) * S_2
        )

    def pretrain_hard(self, sample, separation):
        return (sample - 0.5) * separation

    def entropy(self, params, rng_key, tau, n, num_bethe_iters):
        samples = self.sample_hard_batched(params, tau, rng_key, n)
        log_probs = vmap(self.logprob, in_axes=(0, None, None))(
            samples, params, num_bethe_iters
        )
        return -np.mean(log_probs)

    def exact_entropy(self, params, rng_key, tau, n):
        samples = self.sample_hard_batched(params, tau, rng_key, n)
        log_probs = 0
        for sample in samples:
            log_probs += self.exact_logprob(sample, params)
        return -log_probs / n

    def get_uniform_params(self):
        return np.zeros((self.dim, self.dim))


def get_doubly_stochastic(parameterisation, dim, noise_type="gumbel", tol=1e-4):
    """Makes a class which gives a doubly stochastic
    matrix and exposes methods for returning the matrix"""
    supported_parameterisations = [
        "gumbel-sinkhorn",
    ]

    if parameterisation not in supported_parameterisations:
        raise NotImplementedError(
            f"Supported Parameterisations are {supported_parameterisations}"
        )

    if parameterisation == "gumbel-sinkhorn":
        return GumbelSinkhorn(dim, noise_type, tol)
