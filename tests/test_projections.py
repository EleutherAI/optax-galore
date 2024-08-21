# tests/test_projections.py
import os
print("Current working directory:", os.getcwd())

import jax
import jax.numpy as jnp
import pytest
import optax_galore.optax_galore as og
from optax_galore.optax_galore import ProjectionSpec, reproject, project_gradients, project_back, create_rank_pytree

@pytest.fixture
def sample_params():
    key = jax.random.PRNGKey(0)
    return {
        'layer1': {
            'w': jax.random.normal(key, shape=(10, 20)),
            'b': jax.random.normal(key, shape=(20,))
        },
        'layer2': {
            'w': jax.random.normal(key, shape=(20, 30)),
            'b': jax.random.normal(key, shape=(30,))
        }
    }

@pytest.fixture
def conv_params():
    key = jax.random.PRNGKey(0)
    return {
        'conv1': {
            'w': jax.random.normal(key, shape=(3, 3, 64, 128)),
            'b': jax.random.normal(key, shape=(128,))
        },
        'conv2': {
            'w': jax.random.normal(key, shape=(5, 5, 128, 256)),
            'b': jax.random.normal(key, shape=(256,))
        }
    }

def test_create_rank_pytree():
    # Test case 1: Constant rank
    params = {
        'layer1': {'w': jnp.ones((10, 20)), 'b': jnp.ones(20)},
        'layer2': {'w': jnp.ones((20, 30)), 'b': jnp.ones(30)}
    }
    constant_rank = 5
    rank_pytree = create_rank_pytree(params, constant_rank)
    
    assert jax.tree.structure(rank_pytree) == jax.tree.structure(params)
    assert all(jax.tree.leaves(jax.tree.map(lambda x: x == constant_rank, rank_pytree)))
    assert all(jax.tree.leaves(jax.tree.map(lambda x: x == constant_rank, rank_pytree)))
    # Test case 2: Rank function
    def rank_fn(leaf, path):
        if any(key.key == 'w' for key in path):
            return min(leaf.shape) // 2
        else:
            return 1
    rank_pytree_func = create_rank_pytree(params, rank_fn)

    assert rank_pytree_func['layer1']['w'] == 5  # min(10, 20) // 2
    assert rank_pytree_func['layer1']['b'] == 1
    assert rank_pytree_func['layer2']['w'] == 10  # min(20, 30) // 2
    assert rank_pytree_func['layer2']['b'] == 1
    # Test case 3: Invalid input
    with pytest.raises(ValueError):
        create_rank_pytree(params, "invalid_input")

    # Test case 4: Empty pytree
    empty_params = {}
    empty_rank_pytree = create_rank_pytree(empty_params, constant_rank)
    assert empty_rank_pytree == {}

    # Test case 5: Nested pytree
    nested_params = {
        'outer': {
            'inner1': {'w': jnp.ones((5, 5)), 'b': jnp.ones(5)},
            'inner2': {'w': jnp.ones((10, 10)), 'b': jnp.ones(10)}
        }
    }
    nested_rank_pytree = create_rank_pytree(nested_params, constant_rank)
    assert jax.tree.structure(nested_rank_pytree) == jax.tree.structure(nested_params)
    assert all(jax.tree.leaves(jax.tree.map(lambda x: x == constant_rank, nested_rank_pytree)))


def test_reproject_convolution(conv_params):
    rank = create_rank_pytree(conv_params, 32)
    dimension_pytree = {
        'conv1': {'w': ProjectionSpec(2, 3), 'b': None},
        'conv2': {'w': ProjectionSpec(2, 3), 'b': None}
    }
    
    projections = reproject(conv_params, rank, dimension_pytree)
    
    assert projections['conv1']['w'].shape == (3, 3, 64, 32)
    assert projections['conv2']['w'].shape == (5, 5, 128, 32)
    assert projections['conv1']['b'] is None
    assert projections['conv2']['b'] is None

    # Check if the projections can be used for the intended matrix multiplication
    assert jnp.einsum('abij,abik->abkj', conv_params['conv1']['w'], projections['conv1']['w']).shape == (3, 3, 32, 128)
    assert jnp.einsum('abij,abik->abkj', conv_params['conv2']['w'], projections['conv2']['w']).shape == (5, 5, 32, 256)

def test_project_gradients_convolution(conv_params):
    rank = create_rank_pytree(conv_params, 32)
    dimension_pytree = {
        'conv1': {'w': ProjectionSpec(2, 3), 'b': None},
        'conv2': {'w': ProjectionSpec(2, 3), 'b': None}
    }
    
    projections = reproject(conv_params, rank, dimension_pytree)
    gradients = jax.tree.map(jnp.ones_like, conv_params)
    
    projected_grads = project_gradients(gradients, projections, dimension_pytree)
    
    assert projected_grads['conv1']['w'].shape == (3, 3, 32, 128)
    assert projected_grads['conv2']['w'].shape == (5, 5, 32, 256)
    assert projected_grads['conv1']['b'].shape == conv_params['conv1']['b'].shape
    assert projected_grads['conv2']['b'].shape == conv_params['conv2']['b'].shape

def test_project_back_convolution(conv_params):
    rank = create_rank_pytree(conv_params, 32)
    dimension_pytree = {
        'conv1': {'w': ProjectionSpec(2, 3), 'b': None},
        'conv2': {'w': ProjectionSpec(2, 3), 'b': None}
    }
    
    projections = reproject(conv_params, rank, dimension_pytree)
    updates = {
        'conv1': {
            'w': jnp.ones((3, 3, 32, 128)),
            'b': jnp.ones_like(conv_params['conv1']['b'])
        },
        'conv2': {
            'w': jnp.ones((5, 5, 32, 256)),
            'b': jnp.ones_like(conv_params['conv2']['b'])
        }
    }
    
    projected_back = project_back(updates, projections, dimension_pytree)
    
    assert projected_back['conv1']['w'].shape == conv_params['conv1']['w'].shape
    assert projected_back['conv2']['w'].shape == conv_params['conv2']['w'].shape
    assert projected_back['conv1']['b'].shape == conv_params['conv1']['b'].shape
    assert projected_back['conv2']['b'].shape == conv_params['conv2']['b'].shape

def test_projection_roundtrip_convolution(conv_params):
    rank = create_rank_pytree(conv_params, 32)
    dimension_pytree = {
        'conv1': {'w': ProjectionSpec(2, 3), 'b': None},
        'conv2': {'w': ProjectionSpec(2, 3), 'b': None}
    }
    
    projections = reproject(conv_params, rank, dimension_pytree)
    gradients = jax.tree.map(jnp.ones_like, conv_params)
    
    projected_grads = project_gradients(gradients, projections, dimension_pytree)
    roundtrip_grads = project_back(projected_grads, projections, dimension_pytree)
    
    for layer in roundtrip_grads:
        assert roundtrip_grads[layer]['w'].shape == conv_params[layer]['w'].shape
        assert roundtrip_grads[layer]['b'].shape == conv_params[layer]['b'].shape

def test_different_projection_dimensions(conv_params):
    def rank_fn(leaf, path):
        if any(key.key == 'w' for key in path):
            return 32
        else:
            return 1 

    rank = create_rank_pytree(conv_params, rank_fn)
    dimension_pytree = {
        'conv1': {'w': ProjectionSpec(2, 3), 'b': None},  # Project over input and output channels
        'conv2': {'w': ProjectionSpec(3, 2), 'b': None}   # Project over spatial dimensions
    }
    
    projections = reproject(conv_params, rank, dimension_pytree)
    
    assert projections['conv1']['w'].shape == (3, 3, 64, 32)
    assert projections['conv2']['w'].shape == (5, 5, 32, 256)
    
    gradients = jax.tree.map(jnp.ones_like, conv_params)
    projected_grads = project_gradients(gradients, projections, dimension_pytree)
    
    assert projected_grads['conv1']['w'].shape == (3, 3, 32, 128)
    assert projected_grads['conv2']['w'].shape == (5, 5, 128, 32)
    
    roundtrip_grads = project_back(projected_grads, projections, dimension_pytree)
    
    assert roundtrip_grads['conv1']['w'].shape == conv_params['conv1']['w'].shape
    assert roundtrip_grads['conv2']['w'].shape == conv_params['conv2']['w'].shape

def test_reproject(sample_params):
    rank = create_rank_pytree(sample_params, 5)
    projections = reproject(sample_params, rank)
    
    assert set(projections.keys()) == set(sample_params.keys())
    for layer in projections:
        assert set(projections[layer].keys()) == set(sample_params[layer].keys())
        
        # Check the shape of the projection for 'w'
        original_shape = sample_params[layer]['w'].shape
        expected_shape = (original_shape[-2], 5)  # The projection should be (input_dim, rank)
        assert projections[layer]['w'].shape == expected_shape, f"Expected shape {expected_shape}, got {projections[layer]['w'].shape}"
        
        # Verify that the projection can be used for the intended matrix multiplication
        projected_output = jnp.einsum('ij,ik->jk', projections[layer]['w'], sample_params[layer]['w'])
        assert projected_output.shape == (5, original_shape[-1]), f"Expected shape {(5, original_shape[-1])}, got {projected_output.shape}"
        
        # Check orthogonality of the projection
        proj_2d = projections[layer]['w']
        assert jnp.allclose(jnp.dot(proj_2d.T, proj_2d), jnp.eye(5), atol=1e-5)
        
        # Check that 'b' is not projected
        assert projections[layer]['b'] is None

def test_reproject_with_dimension_pytree(sample_params):
    rank = create_rank_pytree(sample_params, 5)
    dimension_pytree = {
        'layer1': {'w': ProjectionSpec(0, 1), 'b': None},
        'layer2': {'w': ProjectionSpec(1, 0), 'b': None}
    }
    projections = reproject(sample_params, rank, dimension_pytree)
    
    assert projections['layer1']['w'].shape == (10, 5)
    assert projections['layer2']['w'].shape == (5, 30)

def test_project_gradients(sample_params):
    def rank_fn(leaf, path):
        if any(key.key == 'w' for key in path):
            return min(leaf.shape) // 2
        else:
            return 1

    rank = create_rank_pytree(sample_params, rank_fn)
    projections = reproject(sample_params, rank)
    gradients = jax.tree.map(jnp.ones_like, sample_params)
    
    projected_grads = project_gradients(gradients, projections)
    
    assert jax.tree.structure(projected_grads) == jax.tree.structure(gradients)
    for layer in projected_grads:
        original_shape = sample_params[layer]['w'].shape
        projected_rank = min(original_shape) // 2
        assert projected_grads[layer]['w'].shape == (projected_rank, original_shape[1]), f"Expected shape {(projected_rank ,original_shape[1])}, got {projected_grads[layer]['w'].shape}"
        assert projected_grads[layer]['b'].shape == gradients[layer]['b'].shape

    # Specifically check layer1 and layer2
    assert projected_grads['layer1']['w'].shape == (5, 20)  # (min(10, 20) // 2, 20)
    assert projected_grads['layer2']['w'].shape == (10, 30)  # (min(20, 30) // 2, 30)

def test_project_back(sample_params):
    rank = create_rank_pytree(sample_params, 5)
    dimension_pytree = {
        'layer1': {'w': ProjectionSpec(0, 1), 'b': None},
        'layer2': {'w': ProjectionSpec(0, 1), 'b': None}
    }
    projections = reproject(sample_params, rank, dimension_pytree)
    
    # Create updates with the correct shape (as if they were projected gradients)
    updates = {
        'layer1': {
            'w': jnp.ones((5, 20)),  # (rank, output_dim)
            'b': jnp.ones_like(sample_params['layer1']['b'])
        },
        'layer2': {
            'w': jnp.ones((5, 30)),  # (rank, output_dim)
            'b': jnp.ones_like(sample_params['layer2']['b'])
        }
    }
    
    projected_back = project_back(updates, projections, dimension_pytree)
    
    assert jax.tree.structure(projected_back) == jax.tree.structure(sample_params)
    for layer in projected_back:
        assert projected_back[layer]['w'].shape == sample_params[layer]['w'].shape
        assert projected_back[layer]['b'].shape == sample_params[layer]['b'].shape

    # Additional checks
    assert projected_back['layer1']['w'].shape == (10, 20)
    assert projected_back['layer2']['w'].shape == (20, 30)

    # Check that the projection was actually applied
    for layer in projected_back:
        if 'w' in projected_back[layer]:
            assert projected_back[layer]['w'].shape != updates[layer]['w'].shape
        if 'b' in projected_back[layer]:
            assert jnp.allclose(projected_back[layer]['b'], updates[layer]['b'])

    # Check that the result is the product of the update and the projection
    for layer in projected_back:
        if 'w' in projected_back[layer]:
            expected = jnp.dot(projections[layer]['w'], updates[layer]['w'])
            assert jnp.allclose(projected_back[layer]['w'], expected, atol=1e-5)

def test_projection_roundtrip(sample_params):
    rank = create_rank_pytree(sample_params, 5)
    projections = reproject(sample_params, rank)
    gradients = jax.tree.map(jnp.ones_like, sample_params)
    
    projected_grads = project_gradients(gradients, projections)
    roundtrip_grads = project_back(projected_grads, projections, None)
    
    assert jax.tree.structure(roundtrip_grads) == jax.tree.structure(gradients)
    for layer in roundtrip_grads:
        assert roundtrip_grads[layer]['w'].shape == gradients[layer]['w'].shape
        assert roundtrip_grads[layer]['b'].shape == gradients[layer]['b'].shape

def test_projection_with_4d_tensor():
    key = jax.random.PRNGKey(0)  # Initialize a random key
    params = {
        'conv': {
            'w': jax.random.normal(key, shape=(3, 3, 64, 128)),
            'b': jax.random.normal(key, shape=(128,))
        }
    }
    rank = create_rank_pytree(params, 32)
    dimension_pytree = {
        'conv': {'w': ProjectionSpec(2, 3), 'b': None}
    }
    
    projections = reproject(params, rank, dimension_pytree)
    assert projections['conv']['w'].shape == (3, 3, 64, 32)
    
    gradients = jax.tree.map(jnp.ones_like, params)
    projected_grads = project_gradients(gradients, projections, dimension_pytree)
    assert projected_grads['conv']['w'].shape == (3, 3, 32, 128)

def test_variable_rank_constant():
    params = {
        'layer1': {'w': jnp.ones((10, 20)), 'b': jnp.ones(20)},
        'layer2': {'w': jnp.ones((20, 30)), 'b': jnp.ones(30)}
    }
    rank = 5
    rank_pytree = create_rank_pytree(params, rank)
    
    assert jax.tree.structure(rank_pytree) == jax.tree.structure(params)
    assert rank_pytree['layer1']['w'] == 5
    assert rank_pytree['layer1']['b'] == 5
    assert rank_pytree['layer2']['w'] == 5
    assert rank_pytree['layer2']['b'] == 5 

def test_variable_rank_function():
    params = {
        'layer1': {'w': jnp.ones((10, 20)), 'b': jnp.ones(20)},
        'layer2': {'w': jnp.ones((20, 30)), 'b': jnp.ones(30)}
    }
    
    def rank_fn(leaf, path):
        if any(key.key == 'w' for key in path):
            return min(leaf.shape) // 2
        else:
            return 1
    
    rank_pytree = create_rank_pytree(params, rank_fn)
    
    assert jax.tree.structure(rank_pytree) == jax.tree.structure(params)
    assert rank_pytree['layer1']['w'] == 5  # min(10, 20) // 2
    assert rank_pytree['layer1']['b'] == 1
    assert rank_pytree['layer2']['w'] == 10  # min(20, 30) // 2
    assert rank_pytree['layer2']['b'] == 1

    # Additional checks
    for layer in rank_pytree:
        for param in rank_pytree[layer]:
            if param == 'w':
                assert rank_pytree[layer][param] == min(params[layer][param].shape) // 2
            else:
                assert rank_pytree[layer][param] == 1

    # Test with a more complex nested structure
    complex_params = {
        'conv1': {'filters': jnp.ones((3, 3, 64, 128)), 'bias': jnp.ones(128)},
        'dense': {
            'layer1': {'w': jnp.ones((256, 128)), 'b': jnp.ones(128)},
            'layer2': {'w': jnp.ones((128, 64)), 'b': jnp.ones(64)}
        }
    }
    
    complex_rank_pytree = create_rank_pytree(complex_params, rank_fn)
    
    assert complex_rank_pytree['conv1']['filters'] == 1
    assert complex_rank_pytree['conv1']['bias'] == 1
    assert complex_rank_pytree['dense']['layer1']['w'] == 64  # min(256, 128) // 2
    assert complex_rank_pytree['dense']['layer1']['b'] == 1
    assert complex_rank_pytree['dense']['layer2']['w'] == 32  # min(128, 64) // 2
    assert complex_rank_pytree['dense']['layer2']['b'] == 1

def test_reproject_with_variable_rank():
    key = jax.random.PRNGKey(0)
    params = {
        'layer1': {'w': jax.random.normal(key, shape=(10, 20)), 'b': jax.random.normal(key, shape=(20,))},
        'layer2': {'w': jax.random.normal(key, shape=(20, 30)), 'b': jax.random.normal(key, shape=(30,))}
    }
    
    def rank_fn(leaf, path):
        if any(key.key == 'w' for key in path):
            return min(leaf.shape) // 2
        else:
            return 1

    rank = create_rank_pytree(params, rank_fn)
    projections = reproject(params, rank)
    
    assert projections['layer1']['w'].shape == (10, 5)  # (10, min(10, 20) // 2)
    assert projections['layer2']['w'].shape == (20, 10)  # (20, min(20, 30) // 2)
    assert projections['layer1']['b'] is None
    assert projections['layer2']['b'] is None

    # Check orthogonality of the projections
    assert jnp.allclose(jnp.dot(projections['layer1']['w'].T, projections['layer1']['w']), jnp.eye(5), atol=1e-5)
    assert jnp.allclose(jnp.dot(projections['layer2']['w'].T, projections['layer2']['w']), jnp.eye(10), atol=1e-5)

def test_project_gradients_with_variable_rank():
    key = jax.random.PRNGKey(0)  # Initialize a random key
    params = {
        'layer1': {'w': jax.random.normal(key, shape=(10, 20)), 'b': jax.random.normal(key, shape=(20,))},
        'layer2': {'w': jax.random.normal(key, shape=(20, 30)), 'b': jax.random.normal(key, shape=(30,))}
    }
    
    rank_pytree = {
        'layer1': {'w': 8, 'b': 2},
        'layer2': {'w': 12, 'b': 3}
    }
    
    projections = reproject(params, rank_pytree)
    gradients = jax.tree.map(jnp.ones_like, params)
    
    projected_grads = project_gradients(gradients, projections)
    
    assert projected_grads['layer1']['w'].shape == (8, 20)
    assert projected_grads['layer2']['w'].shape == (12, 30)
    assert projected_grads['layer1']['b'].shape == (20,)
    assert projected_grads['layer2']['b'].shape == (30,)

def test_project_back_with_variable_rank():
    key = jax.random.PRNGKey(0)  # Initialize a random key
    params = {
        'layer1': {'w': jax.random.normal(key, shape=(10, 20)), 'b': jax.random.normal(key, shape=(20,))},
        'layer2': {'w': jax.random.normal(key, shape=(20, 30)), 'b': jax.random.normal(key, shape=(30,))}
    }
    
    rank_pytree = {
        'layer1': {'w': 8, 'b': 2},
        'layer2': {'w': 12, 'b': 3}
    }
    
    projections = reproject(params, rank_pytree)
    updates = {
        'layer1': {'w': jnp.ones((8, 20)), 'b': jnp.ones(20)},
        'layer2': {'w': jnp.ones((12, 30)), 'b': jnp.ones(30)}
    }
    
    projected_back = project_back(updates, projections, None)
    
    assert projected_back['layer1']['w'].shape == params['layer1']['w'].shape
    assert projected_back['layer2']['w'].shape == params['layer2']['w'].shape
    assert projected_back['layer1']['b'].shape == params['layer1']['b'].shape
    assert projected_back['layer2']['b'].shape == params['layer2']['b'].shape

def test_projection_roundtrip_with_variable_rank():
    key = jax.random.PRNGKey(0)  # Initialize a random key
    params = {
        'layer1': {'w': jax.random.normal(key, shape=(10, 20)), 'b': jax.random.normal(key, shape=(20,))},
        'layer2': {'w': jax.random.normal(key, shape=(20, 30)), 'b': jax.random.normal(key, shape=(30,))}
    }
    
    def rank_fn(leaf, path):
        if 'w' in path:
            return min(leaf.shape) // 2
        else:
            return 1
    
    rank_pytree = create_rank_pytree(params, rank_fn)
    projections = reproject(params, rank_pytree)
    gradients = jax.tree.map(jnp.ones_like, params)
    
    projected_grads = project_gradients(gradients, projections)
    roundtrip_grads = project_back(projected_grads, projections, None)
    
    assert jax.tree.structure(roundtrip_grads) == jax.tree.structure(gradients)
    for layer in roundtrip_grads:
        assert roundtrip_grads[layer]['w'].shape == gradients[layer]['w'].shape
        assert roundtrip_grads[layer]['b'].shape == gradients[layer]['b'].shape
