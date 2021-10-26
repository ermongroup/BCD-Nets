from typing import Optional, Union
import jax.numpy as jnp
import haiku as hk

LStateType = Optional[hk.State]
PParamType = Union[hk.Params, jnp.ndarray]
