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
import jax
import optax
import optax_galore.optax_galore as og

# Define your model and loss function
# ...

# Create a GaLore optimizer
learning_rate = 0.001
rank = 64
subspace_change_freq = 1000

optimizer = og.galore(
    learning_rate=learning_rate,
    rank=rank,
    subspace_change_freq=subspace_change_freq
)

# Initialize optimizer state
opt_state = optimizer.init(params)

# Define the loss function
def loss_fn(params, batch):
    # Your model's loss calculation
    return loss

# Define the update function
@jax.jit
def update(params, opt_state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

# In your training loop:
for batch in data_loader:
    params, opt_state, loss = update(params, opt_state, batch)
```

This updated usage example demonstrates how to create a jitted update function that includes the loss calculation, gradient computation, optimizer update, and parameter update. Using a jitted update function enables the compiler to optimize out the unprojected gradients to save memory (probably).

### Advanced Usage

For more control over the projection dimensions, you can use the `dimension_pytree` parameter:

```python
import optax_galore.optax_galore as og
from optax_galore.optax_galore import ProjectionSpec

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
