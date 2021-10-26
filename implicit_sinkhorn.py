import jax.numpy as jnp
from ott.geometry.geometry import Geometry
from ott.core.sinkhorn import sinkhorn


def while_uv_sinkhorn(X, tol):
    gg = Geometry(-X, epsilon=1)
    dim = len(X)
    # linear_solve_kwargs = {'ridge_identity': 1e-2, 'ridge_kernel': 1e-2}
    log_u, log_v, _, _, _ = sinkhorn(
        gg,
        a=jnp.ones(dim),
        b=jnp.ones(dim),
        threshold=tol,
        parallel_dual_updates=False,
        jit=False,
        use_danskin=True,
        implicit_differentiation=True,
        max_iterations=20_000,
    )
    u = gg.scaling_from_potential(log_u)
    v = gg.scaling_from_potential(log_v)
    return gg.transport_from_scalings(u, v)


def while_uv_sinkhorn_debug(X, tol):
    gg = Geometry(-X, epsilon=1)
    dim = len(X)
    # linear_solve_kwargs = {'ridge_identity': 1e-2, 'ridge_kernel': 1e-2}
    log_u, log_v, _, errors, _ = sinkhorn(
        gg,
        a=jnp.ones(dim),
        b=jnp.ones(dim),
        threshold=tol,
        parallel_dual_updates=False,
        jit=False,
        use_danskin=True,
        implicit_differentiation=True,
        max_iterations=20_000,
    )
    u = gg.scaling_from_potential(log_u)
    v = gg.scaling_from_potential(log_v)
    return gg.transport_from_scalings(u, v), errors
