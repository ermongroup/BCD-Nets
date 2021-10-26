from jax.experimental import host_callback
from scipy.optimize import linear_sum_assignment
import numpy as np
import jax
from jax.nn import one_hot
from jax import vmap
from c_modules.mine import compute_parallel
from jax import numpy as jnp


def multiple_linear_assignment(X):
    # X has shape (batch_size, k, k)
    X_out = np.zeros(X.shape[:-1], dtype=int)
    for i, x in enumerate(X):
        X_out[i] = linear_sum_assignment(x)[1]
    return X_out


def parlinear_assignment(X):
    return compute_parallel(X)


# def hungarian_callback_loop(X):
#     # This will send a batch of matrices of size (n, k, k)
#     # to the host and returns an (n, k) set of indices
#     # of the locations of assignments in the rows of the matrices
#     return host_callback.call(
#         parlinear_assignment,
#         X,
#         result_shape=jax.ShapeDtypeStruct(X.shape[:-1], jnp.int32),
#     )


def hungarian_callback_loop(X):
    # This will send a batch of matrices of size (n, k, k)
    # to the host and returns an (n, k) set of indices
    # of the locations of assignments in the rows of the matrices
    return host_callback.call(
        multiple_linear_assignment,
        # parlinear_assignment,
        X,
        result_shape=jax.ShapeDtypeStruct(X.shape[:-1], jnp.int64),
    )


# def batched_hungarian(X):
#     # Input is (n, k, k)
#     n = X.shape[-1]
#     indices = hungarian_callback_loop(X)
#     return vmap(one_hot, in_axes=(0, None))(indices, n)


def batched_hungarian(X):
    # Input is (n, k, k)
    n = X.shape[-1]
    # indices = hungarian_callback(X)
    indices = hungarian_callback_loop(X)
    return vmap(one_hot, in_axes=(0, None))(indices, n)


def linear_sum_assignment_wrapper(X):
    return linear_sum_assignment(X)[1]


def hungarian(X):
    # Probably more efficient to use the batched version of this generally
    indices = host_callback.call(
        linear_sum_assignment_wrapper,
        X,
        result_shape=jax.ShapeDtypeStruct(X.shape[:-1], int),
    )
    return one_hot(indices, X.shape[0])
