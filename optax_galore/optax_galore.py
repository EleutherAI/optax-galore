"""
optax_galore.py

This module implements the Gradient Low-Rank Projection (GaLore) algorithm for memory-efficient
training of Large Language Models (LLMs). GaLore allows for full-parameter learning while being
more memory-efficient than common low-rank adaptation methods like LoRA.

The key idea of GaLore is to project the high-dimensional gradients into a lower-dimensional
space, perform optimization in this compact space, and then project the updates back to the
original high-dimensional space. This approach significantly reduces the memory required for
optimizer states while maintaining performance.

The main components of this implementation are:

1. ProjectionSpec: A class to specify dimensions for projection.
2. reproject: Function to create low-rank projection matrices from parameters.
3. project_gradients: Function to project gradients into the low-rank space.
4. project_back: Function to project updates back to the original space.
5. galore: The main optimizer function implementing GaLore as an optax GradientTransformation.

The GaLore algorithm can be summarized with the following pseudocode:

BEGIN ALGORITHM: Adam with GaLore
Require: A layer weight matrix W ∈ R^(m×n) with m ≤ n. Step size η, scale factor α,
         decay rates β_1, β_2, rank r, subspace change frequency T.
1: Initialize first-order moment M_0 ∈ R^(n×r) ← 0
2: Initialize second-order moment V_0 ∈ R^(n×r) ← 0
3: Initialize step t ← 0
4: Repeat
5:     G_t ∈ R^(m×n) ← -∇W φ_t(W_t)
6:     If t mod T = 0 then
7:         U, S, V ← SVD(G_t)
8:         P_t ← U[:, :r]  # Initialize left projector as m ≤ n
9:     Else
10:        P_t ← P_(t-1)  # Reuse the previous projector
11:    End If
12:    R_t ← P_t^T G_t  # Project gradient into compact space
13:    UPDATE(R_t) by Adam:
14:        M_t ← β_1 · M_(t-1) + (1 - β_1) · R_t
15:        V_t ← β_2 · V_(t-1) + (1 - β_2) · R_t^2
16:        M_t ← M_t / (1 - β_1^t)
17:        V_t ← V_t / (1 - β_2^t)
18:        N_t ← M_t / (√V_t + ε)
19:    G̃_t ← α · P N_t  # Project back to original space
20:    W_t ← W_(t-1) + η · G̃_t
21:    t ← t + 1
22: Until convergence criteria met
23: Return W_t
END ALGORITHM

This implementation allows for efficient training of large models on limited memory hardware,
such as pre-training a 7B parameter model on consumer GPUs with 24GB memory without requiring
model parallelism, checkpointing, or offloading strategies.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Optional, Tuple, Union, Any, NamedTuple, Callable

class ProjectionSpec:
    """
    A class to specify the dimensions for projection operations.

    Attributes:
        first_dim (int): The first dimension for projection.
        second_dim (int): The second dimension for projection.
    """
    def __init__(self, first_dim: int, second_dim: int):
        self.first_dim = first_dim
        self.second_dim = second_dim
    
def create_rank_pytree(params: Any, rank: Union[int, Callable]) -> Any:
    """
    Create a rank pytree from a single value or a function.

    Args:
        params: The parameter pytree to match the structure of.
        rank: Either a single integer value to use for all leaves,
              or a function that takes (leaf, path) and returns an integer.

    Returns:
        A pytree with the same structure as params, where each leaf is an integer rank.
    """
    if isinstance(rank, int):
        return jax.tree_map(lambda _: rank, params)
    elif callable(rank):
        return jax.tree_map_with_path(lambda path, leaf: rank(leaf, path), params)
    else:
        raise ValueError("rank must be either an integer or a callable")

def reproject(parameters, rank_pytree, dimension_pytree=None):
    """
    Compute a projection matrix pytree from a parameter pytree.

    This function performs low-rank approximation on the parameter matrices
    using Singular Value Decomposition (SVD). It's a key component of the
    GaLore (Gradient Low-Rank Projection) optimization algorithm.

    The function processes each leaf of the parameter pytree independently:
    1. It reshapes the parameter tensor to a 2D matrix (or keeps it as is if already 2D).
    2. Applies SVD to this matrix.
    3. Returns the first 'r' left singular vectors as the projection matrix.

    Args:
        parameters: A pytree of parameter tensors to be projected.
        r (int): The rank of the projection, i.e., the number of singular vectors to keep.
        dimension_pytree: Optional pytree specifying custom dimensions for projection.
            If None, the function uses the last two dimensions of each parameter tensor.
            If provided, it should have the same structure as 'parameters', with each
            leaf being either None or a ProjectionSpec object specifying which dimensions
            to use for projection.

    Returns:
        A pytree with the same structure as 'parameters', where each leaf is replaced
        by its corresponding projection matrix (or None if projection is not applicable).

    The dimension_pytree allows for flexible specification of which dimensions to use
    for projection in each parameter tensor. This is particularly useful for tensors
    with more than 2 dimensions, where the default behavior of using the last two
    dimensions might not be appropriate.

    For example, in a 4D tensor representing a convolutional layer's weights,
    you might want to project over the input and output channel dimensions,
    rather than the spatial dimensions. The dimension_pytree allows you to
    specify this custom behavior.
    """
    def project_leaf(leaf, r, spec):
        if spec is None or leaf.ndim < 2:
            return None
        
        if dimension_pytree is None:
            dims = (-2, -1)
        else:
            dims = (spec.first_dim, spec.second_dim)
        
        shape = leaf.shape
        leaf_2d = leaf.reshape(-1, shape[dims[0]], shape[dims[1]])
        
        U, _, _ = jax.vmap(jnp.linalg.svd, in_axes=(0, None))(leaf_2d, full_matrices=False)
        return U[..., :r]
    
    if dimension_pytree is None:
        dimension_pytree = jax.tree_map(lambda _: None, parameters)
    
    return jax.tree_map(project_leaf, parameters, rank_pytree, dimension_pytree)

def project_gradients(parameter_pytree, projection_pytree, dimension_pytree=None):
    """
    Project gradients using the computed projection matrices.

    Args:
        parameter_pytree: The parameter pytree.
        projection_pytree: The projection matrix pytree.
        dimension_pytree: Optional pytree specifying dimensions for projection.

    Returns:
        The projected gradient pytree.
    """
    def project_leaf(param, proj, spec):
        if proj is None:
            return param
        
        if dimension_pytree is None:
            dims = (-2, -1)
        else:
            dims = (spec.first_dim, spec.second_dim)
        
        param_shape = param.shape
        proj_shape = proj.shape
        
        param_2d = param.reshape(-1, param_shape[dims[0]], param_shape[dims[1]])
        proj_2d = proj.reshape(-1, proj_shape[-2], proj_shape[-1])
        
        result = jax.vmap(jnp.matmul, in_axes=(0, 0))(jnp.transpose(proj_2d, (0, 2, 1)), param_2d)
        return result.reshape(param_shape[:-2] + (-1,))
    
    if dimension_pytree is None:
        dimension_pytree = jax.tree_map(lambda _: None, parameter_pytree)
    
    return jax.tree_map(project_leaf, parameter_pytree, projection_pytree, dimension_pytree)


def project_back(update_pytree, projection_pytree, dimension_pytree):
    """
    Project the updates back to the original parameter space.

    Args:
        update_pytree: The update pytree in the projected space.
        projection_pytree: The projection matrix pytree.
        dimension_pytree: Optional pytree specifying dimensions for projection.

    Returns:
        The update pytree in the original parameter space.
    """
    def project_back_leaf(update, proj, spec):
        if proj is None:
            return update
        
        if spec is None:
            dims = (-2, -1)
        else:
            dims = (spec.first_dim, spec.second_dim)
        
        update_shape = update.shape
        proj_shape = proj.shape
        
        update_2d = update.reshape(-1, update_shape[-1])
        proj_2d = proj.reshape(-1, proj_shape[-2], proj_shape[-1])
        
        result = jax.vmap(jnp.matmul, in_axes=(0, 0))(proj_2d, update_2d)
        
        if spec is None:
            new_shape = update_shape[:-1] + (proj_shape[-2],)
        else:
            new_shape = update_shape[:dims[0]] + (proj_shape[-2],) + update_shape[dims[0]+1:]
        
        return result.reshape(new_shape)
    
    if dimension_pytree is None:
        dimension_pytree = jax.tree_map(lambda _: None, update_pytree)
    
    return jax.tree_map(project_back_leaf, update_pytree, projection_pytree, dimension_pytree)



class GaLoreState(NamedTuple):
    """State for the GaLore wrapper."""
    count: jnp.ndarray
    inner_state: Any
    projections: Any

def galore_wrapper(
    base_optimizer: optax.GradientTransformation,
    rank: int = 64,
    subspace_change_freq: int = 1000,
    dimension_pytree: Optional[Any] = None,
) -> optax.GradientTransformation:
    """
    A generic GaLore wrapper that can be applied to any Optax optimizer.

    Args:
        base_optimizer: The Optax optimizer to wrap with GaLore.
        rank: The rank of the projection.
        subspace_change_freq: How often to update the projection subspace.
        dimension_pytree: Optional pytree specifying dimensions for projection.

    Returns:
        A new GradientTransformation that applies GaLore to the base optimizer.
    """
    
    def init_fn(params):
        rank_pytree = rank if isinstance(rank, jax.tree_util.PyTreeDef) else create_rank_pytree(params, rank)
        projections = reproject(params, rank_pytree, dimension_pytree)
        return GaLoreState(
            count=jnp.zeros([], jnp.int32),
            inner_state=base_optimizer.init(params),
            projections=projections,
        )

    def update_fn(updates, state, params):
        count = optax.safe_int32_increment(state.count)
        should_update_projections = count % subspace_change_freq == 0
        rank_pytree = rank if isinstance(rank, jax.tree_util.PyTreeDef) else create_rank_pytree(params, rank)
        projections = jax.lax.cond(
            should_update_projections,
            lambda: reproject(updates, rank_pytree, dimension_pytree),
            lambda: state.projections
        )
        
        projected_updates = project_gradients(updates, projections, dimension_pytree)
        inner_updates, new_inner_state = base_optimizer.update(projected_updates, state.inner_state, params)
        
        final_updates = project_back(inner_updates, projections, dimension_pytree)
        
        return final_updates, GaLoreState(count, new_inner_state, projections)

    return optax.GradientTransformation(init_fn, update_fn)


def galore(
    learning_rate: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    rank: int = 64,
    subspace_change_freq: int = 1000,
    dimension_pytree: Optional[Any] = None,
) -> optax.GradientTransformation:
    """GaLore optimizer (maintained for backwards compatibility)."""
    base_optimizer = optax.adam(learning_rate, b1, b2, eps, eps_root)
    return galore_wrapper(base_optimizer, rank, subspace_change_freq, dimension_pytree)

# Example usage of the generic GaLore wrapper
def galore_sgd(
    learning_rate: float,
    momentum: float = 0.9,
    nesterov: bool = False,
    rank: int = 64,
    subspace_change_freq: int = 1000,
    dimension_pytree: Optional[Any] = None,
) -> optax.GradientTransformation:
    """GaLore applied to SGD optimizer."""
    base_optimizer = optax.sgd(learning_rate, momentum, nesterov)
    return galore_wrapper(base_optimizer, rank, subspace_change_freq, dimension_pytree)