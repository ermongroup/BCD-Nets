import jax.numpy as jnp

# from jax.scipy.linalg import expm
from our_linalg import expm
import numpy as np
import optax
import jax.random as rnd
from jax import lax, grad, jit, partial, vmap
from sklearn.utils import resample

# expm = lambda x: _expm()


def solve_golem(
    Xs,
    lambda_1=2e-2,
    lambda_2=5,
    lr=1e-3,
    max_iters=50_000,
    ev=True,
    key=rnd.PRNGKey(0),
):
    _, dim = Xs.shape
    init_scale = 1.0

    def ev_loss(W):
        return (dim / 2) * jnp.log(jnp.sum((Xs.T - W.T @ Xs.T) ** 2))

    def non_ev_loss(W):
        return (dim / 2) * jnp.sum(
            jnp.log(jnp.sum((Xs.T - W.T @ Xs.T) ** 2, axis=-1)), axis=0
        )

    if ev:
        inner_loss = ev_loss
    else:
        inner_loss = non_ev_loss

    mask = 1 - np.eye(dim)

    def loss(W_params):
        """Computes the loss in the Frobenius form"""
        W = W_params * mask
        # W = to_W(W_params)
        likelihood_loss = inner_loss(W) - jnp.linalg.slogdet(jnp.eye(dim) - W)[1]
        L_1_loss = lambda_1 * jnp.linalg.norm(W.flatten(), 1)
        DAG_loss = lambda_2 * (jnp.trace(expm(W * W)) - dim)
        return likelihood_loss + L_1_loss + DAG_loss

    grad_tolerance = 1e-3
    # W_params_init = rnd.normal(key, shape=(dim * (dim - 1),)) * init_scale
    W_params_init = rnd.normal(key, shape=(dim, dim)) * init_scale
    opt_W = optax.adam(lr, eps=1e-8)
    opt_W_params = opt_W.init(W_params_init)

    def step(carry):
        W_params, opt_W_params, i, _ = carry
        W_grad = grad(loss)(W_params)
        W_updates, opt_W_params = opt_W.update(W_grad, opt_W_params, W_params)
        W_params = optax.apply_updates(W_params, W_updates)
        return W_params, opt_W_params, i + 1, jnp.linalg.norm(W_grad)

    def cond_fn(carry):
        _, _, i, grad_norm = carry
        break_condition = jnp.bitwise_or(i > max_iters, grad_norm < grad_tolerance)
        return jnp.bitwise_not(break_condition)

    init_val = (W_params_init, opt_W_params, 0.0, jnp.inf)
    out_vals = lax.while_loop(cond_fn, step, init_val)
    W_params, _, i, grad_norm = out_vals
    out = W_params * mask
    # out = to_W(W_params)
    return out, i, grad_norm


solve_golem_jit = jit(solve_golem, static_argnums=(5,))


def solve_golem_cv_nonjit(
    Xs,
    lambda_1_list,
    lambda_2=5,
    lr=1e-3,
    max_iters=50_000,
    ev=True,
    holdout_frac=0.2,
    threshold=0.3,
):
    """Does GOLEM for all the lambda_1 in the list, returns best held-out
    accuracy"""
    n, d = Xs.shape
    train_Xs = Xs[: int(holdout_frac * n)]
    best_val_mse = jnp.inf
    best_lambda = None
    best_W = jnp.ones((d, d)) * jnp.nan
    rng_key = rnd.PRNGKey(0)

    for lambda_1 in lambda_1_list:
        rng_key, _ = rnd.split(rng_key, 2)
        W_out, i, grad_norm = solve_golem_jit(
            train_Xs, lambda_1, lambda_2, lr, max_iters, ev, rng_key
        )
        print(f"After {i} steps, final gradient norm was {grad_norm}")
        est_W_clipped = np.where(np.abs(W_out) > threshold, W_out, 0)
        val_mse = jnp.mean((Xs.T - est_W_clipped.T @ Xs.T) ** 2)
        print(f"For lambda_1 {lambda_1}, val_mse {val_mse}")
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_lambda = lambda_1
            best_W = W_out

    print(f"Best lambda was {best_lambda}, with MSE {best_val_mse}")
    return best_W


def solve_golem_cv(
    Xs,
    lambda_1_list,
    lambda_2=5,
    lr=1e-3,
    max_iters=50_000,
    ev=True,
    holdout_frac=0.2,
    threshold=0.3,
):

    holdout_num = int(holdout_frac * len(Xs))
    (_, best_W), (val_mses, i_s, grad_norms) = jit(
        solve_golem_cv_for_jit, static_argnames=("holdout_num", "ev", "max_iters")
    )(
        Xs,
        jnp.array(lambda_1_list),
        lambda_2,
        lr=lr,
        max_iters=max_iters,
        ev=ev,
        holdout_num=holdout_num,
        threshold=threshold,
        rng_key=rnd.PRNGKey(0),
    )
    print(
        f"For lambdas {lambda_1_list}, got val MSEs {val_mses}, iterations {i_s}, grad norms {grad_norms}"
    )
    return best_W


def solve_golem_cv_for_jit(
    Xs,
    lambda_1_list,
    lambda_2=5,
    lr=1e-3,
    max_iters=50_000,
    ev=True,
    holdout_num=10,
    threshold=0.3,
    rng_key=rnd.PRNGKey(0),
):
    """Does GOLEM for all the lambda_1 in the list, returns best held-out
    accuracy"""
    _, d = Xs.shape
    train_Xs = Xs[:holdout_num]
    best_W = jnp.ones((d, d)) * jnp.nan
    rng_keys = rnd.split(rng_key, len(lambda_1_list))

    def inner_solve(lambda_1, rng_key):
        W_out, i, grad_norm = solve_golem(
            train_Xs, lambda_1, lambda_2, lr, max_iters, ev, rng_key
        )
        est_W_clipped = jnp.where(jnp.abs(W_out) > threshold, W_out, 0)
        val_mse = jnp.mean((Xs.T - est_W_clipped.T @ Xs.T) ** 2)
        return W_out, val_mse, i, grad_norm

    all_Ws, all_val_mses, i_s, grad_norms = vmap(inner_solve)(lambda_1_list, rng_keys)
    smallest_val_mse = jnp.argmin(all_val_mses)
    best_W = all_Ws[smallest_val_mse]
    return (smallest_val_mse, best_W), (all_val_mses, i_s, grad_norms)


def bootstrapped_golem_cv(
    Xs,
    lambda_1_list,
    lambda_2=5,
    lr=1e-3,
    max_iters=50_000,
    ev=True,
    holdout_frac=0.2,
    bootstrap_frac=0.95,
    bootstrap_iters=20,
    rng_key=rnd.PRNGKey(0),
) -> np.ndarray:
    """Performs a bootstrap with golem, i.e. resampling the dataset with 
    replacement, fitting the model on this new dataset, and repeating this
    many times. """
    data_size = int(bootstrap_frac * len(Xs))
    random_states = rnd.split(rng_key, bootstrap_iters)[:, 0]
    rng_keys = rnd.split(rng_key, bootstrap_iters)

    data_inputs = np.ones((bootstrap_iters, data_size, Xs.shape[-1])) * np.nan
    for k in range(bootstrap_iters):
        data = resample(
            Xs, replace=True, n_samples=data_size, random_state=int(random_states[k])
        )
        data_inputs[k] = data

    golem_cv_fn = lambda xs, r: solve_golem_cv_for_jit(
        xs,
        jnp.array(lambda_1_list),
        lambda_2,
        lr,
        max_iters,
        ev,
        holdout_num=int(holdout_frac * len(Xs)),
        rng_key=r,
    )

    # jit(golem_cv_fn)(data_inputs[0], rng_keys[0])
    # vmap(golem_cv_fn)(data_inputs, rng_keys)
    # breakpoint()
    (best_mses, best_Ws), _ = jit(vmap(golem_cv_fn))(data_inputs, rng_keys)
    return best_Ws
