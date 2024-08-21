"""
optax_galore.py

This module implements the Gradient Low-Rank Projection (GaLore) algorithm for memory-efficient
training of Large Language Models (LLMs). GaLore allows for full-parameter learning while being
more memory-efficient than common low-rank adaptation methods like LoRA.

You can find the original GaLore paper here: https://arxiv.org/abs/2403.03507
and the pytorch repository (probably) here: https://github.com/jiaweizzhao/GaLore

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

This implementation allows for efficient fine-tuneing of large models on limited memory hardware.

Take a look at the examples for how to combine optax-galore with 
jax paralleism to train large models on individual nodes.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Optional, Tuple, Union, Any, NamedTuple, Callable

class ProjectionSpec:
    """
    A class to specify the dimensions for projection operations.

    Attributes:
        first_dim (int): The first dimension for projection: will be downsized to <rank>
        second_dim (int): The second dimension for projection: will be used to compute U matrix, but not downsized 
    
    For example to shrink the last dimension in a n by m matrix, ProjectionSpec(1,0) would be appropriate
    Default behavior when a projection spec is to assume ProjectionSpec(-2,-1) or to not project down at all
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
        return jax.tree.map(lambda _: rank, params)
    elif callable(rank):
        return jax.tree_util.tree_map_with_path(lambda path, leaf: rank(leaf, path), params)
    else:
        raise ValueError("rank must be either an integer or a callable")

def dim_permutation(dims, ndims):
    # Create the permutation
    perm = list(range(ndims))
    
    #move relevant dimensions to the end
    dim0,dim1 = perm[dims[0]],perm[dims[1]]
    l1,l2 = perm[-2],perm[-1]
    perm[dims[0]],perm[dims[1]] = l1,l2 
    perm[-2],perm[-1] = dim0,dim1
    
    assert dim0 != dim1, "identical dimensions provided in projection spec"
    
    return perm

def dim_inv_permutation(dims, ndims):
    perm = dim_permutation(dims, ndims)
    # Create the inverse permutation
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return inv_perm 

def reproject(parameters, rank_pytree, dimension_pytree=None):
    """
    Compute a projection matrix pytree from a parameter pytree.

    This function performs low-rank approximation on the parameter matrices
    using Singular Value Decomposition (SVD).

    Args:
        parameters: A pytree of parameter tensors to be projected.
        rank_pytree: A pytree with the same structure as 'parameters', where each leaf
                     specifies the rank to use for the corresponding parameter.
        dimension_pytree: Optional pytree specifying custom dimensions for projection.
            If None, the function uses the last two dimensions of each parameter tensor.
            If provided, it should have the same structure as 'parameters', with each
            leaf being either None or a ProjectionSpec object specifying which dimensions
            to use for projection.

    Returns:
        A pytree with the same structure as 'parameters', where each leaf is replaced
        by its corresponding projection matrix (or None if projection is not applicable).
    """
    def project_leaf(leaf, r, spec):
        if spec is None and leaf.ndim >= 2:
            dims = (-2, -1)
        elif spec is not None:
            dims = (spec.first_dim, spec.second_dim)
        else:
            return None
        
        perm = dim_permutation(dims,leaf.ndim)
        inv_perm = dim_inv_permutation(dims, leaf.ndim)
        
        # Transpose the tensor
        leaf_transposed = jnp.transpose(leaf, perm)
        
        # Perform SVD on the last two dimensions
        U, _, _ = jnp.linalg.svd(leaf_transposed, full_matrices=False)
        
        # Take the first r columns
        U_r = U[..., :r]
        
        # Apply the inverse permutation to U_r
        U_r_original_order = jnp.transpose(U_r, inv_perm)
        
        return U_r_original_order
    
    if dimension_pytree is None:
        dimension_pytree = jax.tree.map(lambda _: None, parameters)
    
    return jax.tree.map(project_leaf, parameters, rank_pytree, dimension_pytree)


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
    def project_leaf(grad, proj_mat, spec):
        if spec is None and grad.ndim >= 2:
            dims = (-2, -1)
        elif spec is not None:
            dims = (spec.first_dim, spec.second_dim)
        else:
            return grad  # Copy the gradient if no projection is available
        
        perm = dim_permutation(dims,grad.ndim)
        inv_perm = dim_inv_permutation(dims, grad.ndim)
        
        transpose_grad = jnp.transpose(grad, perm)
        transpose_proj_mat = jnp.transpose(proj_mat, perm)

        transpose_projection = jnp.einsum('...ij,...ik->...kj', transpose_grad, transpose_proj_mat)
        projection = jnp.transpose(transpose_projection, inv_perm)
        
        return projection
    
    if dimension_pytree is None:
        dimension_pytree = jax.tree.map(lambda _: None, parameter_pytree)
    
    return jax.tree.map(project_leaf, parameter_pytree, projection_pytree, dimension_pytree)

def project_back(update_pytree, projection_pytree, dimension_pytree=None):
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
        
        if spec is None and update.ndim >= 2:
            dims = (-2, -1)
        elif spec is not None:
            dims = (spec.first_dim, spec.second_dim)
        else:
            return update
        
        perm = dim_permutation(dims, update.ndim)
        inv_perm = dim_inv_permutation(dims, update.ndim)
        
        # Transpose the update and projection matrices
        update_transposed = jnp.transpose(update, perm)
        proj_transposed = jnp.transpose(proj, perm)
        
        # Perform the back-projection
        back_projected = jnp.einsum('...ij,...ki->...kj', update_transposed, proj_transposed)
        
        # Apply the inverse permutation to get back to the original shape
        result = jnp.transpose(back_projected, inv_perm)
        
        return result
    
    if dimension_pytree is None:
        dimension_pytree = jax.tree.map(lambda _: None, update_pytree)
    
    return jax.tree.map(project_back_leaf, update_pytree, projection_pytree, dimension_pytree)

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
        projected_params = project_gradients(params, projections, dimension_pytree)
        return GaLoreState(
            count=jnp.zeros([], jnp.int32),
            inner_state=base_optimizer.init(projected_params),
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
    """GaLore optimizer."""
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