from typing import Tuple, Callable, Any
import jax.numpy as jnp
import haiku as hk
import jax
import jax.random as rnd
import numpy as np


Tensor = Any
PRNGKey = Any
Network = Callable[[hk.Params, PRNGKey, Tensor, bool], Tensor]


def get_model(
    dim: int, batch_size: int, num_layers: int, hidden_size: int = 32, do_ev_noise=True,
) -> Tuple[hk.Params, Network]:
    if do_ev_noise:
        noise_dim = 1
    else:
        noise_dim = dim
    l_dim = dim * (dim - 1) // 2
    input_dim = l_dim + noise_dim
    rng_key = rnd.PRNGKey(0)

    def forward_fn(in_data: jnp.ndarray) -> jnp.ndarray:
        # Must have num_heads * key_size (=64) = embedding_size
        x = hk.Linear(hidden_size)(hk.Flatten()(in_data))
        x = jax.nn.gelu(x)
        for _ in range(num_layers - 2):
            x = hk.Linear(hidden_size)(x)
            x = jax.nn.gelu(x)
        x = hk.Linear(hidden_size)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(dim * dim)(x)

    # out_stats = eval_mean(params, Xs, np.zeros(dim))
    forward_fn_init, forward_fn_apply = hk.transform(forward_fn)
    blank_data = np.zeros((batch_size, input_dim))
    laplace_params = forward_fn_init(rng_key, blank_data)
    return laplace_params, forward_fn_apply


def get_model_arrays(
    dim: int,
    batch_size: int,
    num_layers: int,
    rng_key: PRNGKey,
    hidden_size: int = 32,
    do_ev_noise=True,
) -> hk.Params:
    """Only returns parameters so that it can be used in pmap"""
    if do_ev_noise:
        noise_dim = 1
    else:
        noise_dim = dim
    l_dim = dim * (dim - 1) // 2
    input_dim = l_dim + noise_dim

    def forward_fn(in_data: jnp.ndarray) -> jnp.ndarray:
        # Must have num_heads * key_size (=64) = embedding_size
        x = hk.Linear(hidden_size)(hk.Flatten()(in_data))
        x = jax.nn.gelu(x)
        for _ in range(num_layers - 2):
            x = hk.Linear(hidden_size)(x)
            x = jax.nn.gelu(x)
        x = hk.Linear(hidden_size)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(dim * dim)(x)

    # out_stats = eval_mean(params, Xs, np.zeros(dim))
    forward_fn_init, _ = hk.transform(forward_fn)
    blank_data = np.zeros((batch_size, input_dim))
    laplace_params = forward_fn_init(rng_key, blank_data)
    return laplace_params
