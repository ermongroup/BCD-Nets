import nux
import jax
import jax.numpy as jnp
import jax.random as rnd
import nux.networks as net
from nux.internal.layer import InvertibleLayer
from nux.flows.bijective import Reverse, MAF
from nux.flows.stochastic import ContinuouslyIndexed
from jax.scipy.stats import norm
import optax
from jax import jit, value_and_grad
import haiku as hk
from typing import Optional, Mapping, Tuple


class BatchNorm(InvertibleLayer):
    def __init__(self, axis=0, name: str = "batch_norm"):
        """ Elementwise shift + scale
    Args:
      axis: Axes to apply to
      name: Optional name for this module.
    """
        super().__init__(name=name)
        self.axes = (axis,) if isinstance(axis, int) else axis

    def call(
        self, inputs, rng: jnp.ndarray = None, sample: Optional[bool] = False, **kwargs,
    ) -> Mapping[str, jnp.ndarray]:
        eps = 1e-5
        outputs = {}
        x = inputs["x"]
        x_shape = self.get_unbatched_shapes(sample)["x"]
        # x_shape = x.shape[1:]
        m = jnp.mean(x, axis=0)
        v = jnp.std(x, axis=0) ** 2

        gamma = hk.get_parameter(
            "gamma",
            shape=x_shape,
            dtype=x.dtype,
            init=lambda *args: jnp.ones(*args) * -2,
        )
        beta = hk.get_parameter("beta", shape=x_shape, dtype=x.dtype, init=jnp.zeros)

        if sample == False:
            outputs["x"] = ((x - m) / jnp.sqrt((v + eps))) * jnp.exp(gamma) + beta
        else:
            raise NotImplementedError

        log_det = jnp.sum(gamma - 0.5 * jnp.log(v + eps))
        outputs["log_det"] = log_det

        return outputs


def get_flow_CIF(
    rand_key,
    input_shape,
    num_layers,
    batch_size,
    num_components=8,
    threshold=-1_000,
    init_std=3.0,
    hidden_sizes=None,
    pretrain=True,
    noise_dim=1,
):
    def create_network(out_shape):
        return net.MLP(
            out_dim=out_shape[-1],
            layer_sizes=[32] * 2,
            nonlinearity="relu",
            parameter_norm=None,
            dropout_rate=None,
        )

    def get_CIF_MAF():
        return ContinuouslyIndexed(MAF(hidden_layer_sizes=[32, 32]))

    def create_flow():
        layers = []
        for _ in range(num_layers):
            layers += [get_CIF_MAF(), Reverse(), BatchNorm()]

        return nux.sequential(*layers)

    # Perform data-dependent initialization
    train_inputs = {
        "x": rnd.normal(rand_key, shape=((batch_size, input_shape))) * init_std
    }
    # flow = nux.Flow(create_flow, rand_key, train_inputs, batch_axes=(0,))
    flow = nux.Flow(create_flow, rand_key, train_inputs, batch_axes=(0,))
    # outputs = flow.apply(rnd.PRNGKey(0), train_inputs)
    # flow = nux.Flow(create_flow, rand_key, train_inputs)

    def sample_flow(
        params: hk.Params, state: hk.State, rng_key, n: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, hk.State]:
        # Sample from base distribution, i.e. normal
        # We get Z -> L
        samples = rnd.normal(rng_key, shape=(n, input_shape))
        # samples = rnd.normal(rng_key, shape=(batch_size, input_shape))

        logprob = jnp.sum(norm.logpdf(samples), axis=-1)
        # samples = rnd.sample(...)

        out, state = flow.stateful_apply(rng_key, {"x": samples}, params, state)
        # Need to add a minus to the log det term since we want p(L),
        # not P(Z) which is what log det would normally be
        sample_log_probs = jnp.clip(
            -out["log_det"] + logprob, a_min=threshold, a_max=None
        )
        return (
            out["x"],  # type: ignore
            sample_log_probs,
            state,
        )  # I *think* this is right # type: ignore

    if pretrain:
        layers = [
            nux.util.scale_by_belief(eps=1e-8),
            optax.scale(3e-3),
            optax.clip(15.0),
        ]
        opt = optax.chain(*layers)
        opt_state = opt.init(flow.params)
        n_steps = 200
        key = jax.random.split(rand_key, 2)[0]
        noise = rnd.normal(rand_key, shape=(batch_size, input_shape)) * init_std
        print("Pretraining Flow")
        p, state = flow.params, flow.state

        def loss_fn(p, state):
            outputs, state = flow.stateful_apply(key, train_inputs, p, state)
            return jnp.mean(outputs["log_px"]), state

        @jit
        def step(p, state, opt_p, key):
            key, data_key = rnd.split(key, 2)
            noise = rnd.normal(data_key, shape=(batch_size, input_shape)) * init_std
            train_inputs = {"x": noise}
            (loss, state), flow_grad = value_and_grad(loss_fn, has_aux=True)(p, state)
            p_updates, opt_state = opt.update(flow_grad, opt_p, p)
            p = optax.apply_updates(p, p_updates)
            return p, state, opt_state, key

        for i in range(n_steps):
            p, state, opt_state, key = step(p, state, opt_state, key)
        params = p
        print("Finished pretraining flow")
    else:
        params = flow.params

    def get_flow_arrays():
        # We need this so that we can pmap the function to get the params
        return flow.params, flow.state

    def get_density(params, state, samples, rng_key):
        logprob = jnp.sum(norm.logpdf(samples), axis=-1)
        output = flow.stateful_apply(rng_key, {"x": samples}, params, state)

        # Need to do - since the way it's set up, we'd get
        logprobs = -output[0]["log_det"] + logprob  # I *think* this is right
        # Deal with occasional issues with point somehow outside the support
        return jnp.clip(logprobs, a_min=threshold, a_max=None)

    return flow.params, sample_flow, get_flow_arrays, get_density
