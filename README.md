# WARNING WORK-IN-PROGRESS MOST OF THIS CODE DOES NOT YET WORK

# Optax-GaLore

Optax-GaLore is an implementation of the Gradient Low-Rank Projection (GaLore) algorithm for memory-efficient training of Large Language Models (LLMs). This project extends the Optax optimization library with GaLore functionality.

## Features

- Memory-efficient optimization for large-scale models
- Compatible with existing Optax optimizers
- Flexible projection specifications for different layer types
- Support for convolutional layers and other multi-dimensional tensors

## Installation

To install Optax-GaLore, clone this repository and install the required dependencies:

```bash
git clone https://github.com/EleutherAI/optax-galore.git
cd optax-galore
pip install -r requirements.txt
```

## Usage

Here's a basic example of how to use Optax-GaLore in your project:

```python
import optax
import optax_galore

# Define your model and loss function
# ...

# Create a GaLore optimizer
learning_rate = 0.001
rank = 64
subspace_change_freq = 1000

optimizer = optax_galore.galore(
    learning_rate=learning_rate,
    rank=rank,
    subspace_change_freq=subspace_change_freq
)

# Initialize optimizer state
opt_state = optimizer.init(params)

# In your training loop:
def train_step(params, opt_state, batch):
    def loss_fn(params):
        # Your model's loss calculation
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

### Advanced Usage

For more control over the projection dimensions, you can use the `dimension_pytree` parameter:

```python
from optax_galore import ProjectionSpec

dimension_pytree = {
    'conv1': {'w': ProjectionSpec(2, 3), 'b': None},
    'conv2': {'w': ProjectionSpec(2, 3), 'b': None}
}

optimizer = optax_galore.galore(
    learning_rate=learning_rate,
    rank=rank,
    subspace_change_freq=subspace_change_freq,
    dimension_pytree=dimension_pytree
)
```

You can also wrap other Optax optimizers with GaLore:

```python
base_optimizer = optax.adam(learning_rate=0.001)
galore_optimizer = optax_galore.galore_wrapper(
    base_optimizer,
    rank=64,
    subspace_change_freq=1000
)
```

## Testing

To run the tests, use pytest:

```bash
pytest tests/
```

## Contributing

Contributions to Optax-GaLore are welcome! Please feel free to submit a Pull Request.