# tests/test_projections.py

import jax
import jax.numpy as jnp
import pytest
import optax_galore
from optax_galore.optax_galore import ProjectionSpec, reproject, project_gradients, project_back

@pytest.fixture
def sample_params():
    return {
        'layer1': {
            'w': jnp.array(jnp.random.normal(size=(10, 20))),
            'b': jnp.array(jnp.random.normal(size=(20,)))
        },
        'layer2': {
            'w': jnp.array(jnp.random.normal(size=(20, 30))),
            'b': jnp.array(jnp.random.normal(size=(30,)))
        }
    }

@pytest.fixture
def conv_params():
    return {
        'conv1': {
            'w': jnp.array(jnp.random.normal(size=(3, 3, 64, 128))),
            'b': jnp.array(jnp.random.normal(size=(128,)))
        },
        'conv2': {
            'w': jnp.array(jnp.random.normal(size=(5, 5, 128, 256))),
            'b': jnp.array(jnp.random.normal(size=(256,)))
        }
    }

def test_reproject_convolution(conv_params):
    rank = 32
    dimension_pytree = {
        'conv1': {'w': ProjectionSpec(2, 3), 'b': None},
        'conv2': {'w': ProjectionSpec(2, 3), 'b': None}
    }
    
    projections = reproject(conv_params, rank, dimension_pytree)
    
    assert projections['conv1']['w'].shape == (64, rank)
    assert projections['conv2']['w'].shape == (128, rank)
    assert projections['conv1']['b'] is None
    assert projections['conv2']['b'] is None

def test_project_gradients_convolution(conv_params):
    rank = 32
    dimension_pytree = {
        'conv1': {'w': ProjectionSpec(2, 3), 'b': None},
        'conv2': {'w': ProjectionSpec(2, 3), 'b': None}
    }
    
    projections = reproject(conv_params, rank, dimension_pytree)
    gradients = jax.tree_map(jnp.ones_like, conv_params)
    
    projected_grads = project_gradients(gradients, projections, dimension_pytree)
    
    assert projected_grads['conv1']['w'].shape == (3, 3, rank, 128)
    assert projected_grads['conv2']['w'].shape == (5, 5, rank, 256)
    assert projected_grads['conv1']['b'].shape == conv_params['conv1']['b'].shape
    assert projected_grads['conv2']['b'].shape == conv_params['conv2']['b'].shape

def test_project_back_convolution(conv_params):
    rank = 32
    dimension_pytree = {
        'conv1': {'w': ProjectionSpec(2, 3), 'b': None},
        'conv2': {'w': ProjectionSpec(2, 3), 'b': None}
    }
    
    projections = reproject(conv_params, rank, dimension_pytree)
    updates = {
        'conv1': {
            'w': jnp.ones((3, 3, rank, 128)),
            'b': jnp.ones_like(conv_params['conv1']['b'])
        },
        'conv2': {
            'w': jnp.ones((5, 5, rank, 256)),
            'b': jnp.ones_like(conv_params['conv2']['b'])
        }
    }
    
    projected_back = project_back(updates, projections, dimension_pytree)
    
    assert projected_back['conv1']['w'].shape == conv_params['conv1']['w'].shape
    assert projected_back['conv2']['w'].shape == conv_params['conv2']['w'].shape
    assert projected_back['conv1']['b'].shape == conv_params['conv1']['b'].shape
    assert projected_back['conv2']['b'].shape == conv_params['conv2']['b'].shape

def test_projection_roundtrip_convolution(conv_params):
    rank = 32
    dimension_pytree = {
        'conv1': {'w': ProjectionSpec(2, 3), 'b': None},
        'conv2': {'w': ProjectionSpec(2, 3), 'b': None}
    }
    
    projections = reproject(conv_params, rank, dimension_pytree)
    gradients = jax.tree_map(jnp.ones_like, conv_params)
    
    projected_grads = project_gradients(gradients, projections, dimension_pytree)
    roundtrip_grads = project_back(projected_grads, projections, dimension_pytree)
    
    for layer in roundtrip_grads:
        assert roundtrip_grads[layer]['w'].shape == conv_params[layer]['w'].shape
        assert roundtrip_grads[layer]['b'].shape == conv_params[layer]['b'].shape
        assert jnp.allclose(jnp.sum(roundtrip_grads[layer]['w']), jnp.sum(gradients[layer]['w']), atol=1e-5)

def test_different_projection_dimensions(conv_params):
    rank = 32
    dimension_pytree = {
        'conv1': {'w': ProjectionSpec(2, 3), 'b': None},  # Project over input and output channels
        'conv2': {'w': ProjectionSpec(0, 1), 'b': None}   # Project over spatial dimensions
    }
    
    projections = reproject(conv_params, rank, dimension_pytree)
    
    assert projections['conv1']['w'].shape == (64, rank)
    assert projections['conv2']['w'].shape == (5, rank)
    
    gradients = jax.tree_map(jnp.ones_like, conv_params)
    projected_grads = project_gradients(gradients, projections, dimension_pytree)
    
    assert projected_grads['conv1']['w'].shape == (3, 3, rank, 128)
    assert projected_grads['conv2']['w'].shape == (rank, rank, 128, 256)
    
    roundtrip_grads = project_back(projected_grads, projections, dimension_pytree)
    
    assert roundtrip_grads['conv1']['w'].shape == conv_params['conv1']['w'].shape
    assert roundtrip_grads['conv2']['w'].shape == conv_params['conv2']['w'].shape

def test_reproject(sample_params):
    rank = 5
    projections = reproject(sample_params, rank)
    
    assert set(projections.keys()) == set(sample_params.keys())
    for layer in projections:
        assert set(projections[layer].keys()) == set(sample_params[layer].keys())
        assert projections[layer]['w'].shape == (10, rank)  # For layer1
        assert projections[layer]['b'] is None

def test_reproject_with_dimension_pytree(sample_params):
    rank = 5
    dimension_pytree = {
        'layer1': {'w': ProjectionSpec(0, 1), 'b': None},
        'layer2': {'w': ProjectionSpec(1, 0), 'b': None}
    }
    projections = reproject(sample_params, rank, dimension_pytree)
    
    assert projections['layer1']['w'].shape == (10, rank)
    assert projections['layer2']['w'].shape == (30, rank)

def test_project_gradients(sample_params):
    rank = 5
    projections = reproject(sample_params, rank)
    gradients = jax.tree_map(jnp.ones_like, sample_params)
    
    projected_grads = project_gradients(gradients, projections)
    
    assert jax.tree_structure(projected_grads) == jax.tree_structure(gradients)
    for layer in projected_grads:
        assert projected_grads[layer]['w'].shape == (rank, 20)  # For layer1
        assert projected_grads[layer]['b'].shape == gradients[layer]['b'].shape

def test_project_back(sample_params):
    rank = 5
    projections = reproject(sample_params, rank)
    updates = jax.tree_map(lambda x: jnp.ones((rank, x.shape[1])) if x.ndim == 2 else x, sample_params)
    
    projected_back = project_back(updates, projections, None)
    
    assert jax.tree_structure(projected_back) == jax.tree_structure(sample_params)
    for layer in projected_back:
        assert projected_back[layer]['w'].shape == sample_params[layer]['w'].shape
        assert projected_back[layer]['b'].shape == sample_params[layer]['b'].shape

def test_projection_roundtrip(sample_params):
    rank = 5
    projections = reproject(sample_params, rank)
    gradients = jax.tree_map(jnp.ones_like, sample_params)
    
    projected_grads = project_gradients(gradients, projections)
    roundtrip_grads = project_back(projected_grads, projections, None)
    
    assert jax.tree_structure(roundtrip_grads) == jax.tree_structure(gradients)
    for layer in roundtrip_grads:
        assert roundtrip_grads[layer]['w'].shape == gradients[layer]['w'].shape
        assert roundtrip_grads[layer]['b'].shape == gradients[layer]['b'].shape
        assert jnp.allclose(jnp.sum(roundtrip_grads[layer]['w']), jnp.sum(gradients[layer]['w']), atol=1e-5)

def test_projection_with_4d_tensor():
    params = {
        'conv': {
            'w': jnp.array(jnp.random.normal(size=(3, 3, 64, 128))),
            'b': jnp.array(jnp.random.normal(size=(128,)))
        }
    }
    rank = 32
    dimension_pytree = {
        'conv': {'w': ProjectionSpec(2, 3), 'b': None}
    }
    
    projections = reproject(params, rank, dimension_pytree)
    assert projections['conv']['w'].shape == (64, rank)
    
    gradients = jax.tree_map(jnp.ones_like, params)
    projected_grads = project_gradients(gradients, projections, dimension_pytree)
    assert projected_grads['conv']['w'].shape == (3, 3, rank, 128)
    
    roundtrip_grads = project_back(projected_grads, projections, dimension_pytree)
    assert roundtrip_grads['conv']['w'].shape == params['conv']['w'].shape

if __name__ == "__main__":
    pytest.main([__file__])