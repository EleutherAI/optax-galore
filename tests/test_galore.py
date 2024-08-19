import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
from jax import random, grad, jit, vmap
import pytest
import optax_galore  # Your implemented module
import time

# Memory profiling
from jax.profiler import memory_stats

# Toy dataset
def generate_toy_data(num_samples=1000, input_dim=100, num_classes=2):
    key = random.PRNGKey(0)
    X = random.normal(key, (num_samples, input_dim))
    y = random.randint(key, (num_samples,), 0, num_classes)
    return X, y

# Simple neural network
def init_network(layer_sizes, key):
    keys = random.split(key, len(layer_sizes) - 1)
    return [
        {'w': random.normal(k, (m, n)) * 0.01, 'b': jnp.zeros(n)}
        for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])
    ]

def forward(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(jnp.dot(x, layer['w']) + layer['b'])
    final_layer = params[-1]
    return jnp.dot(x, final_layer['w']) + final_layer['b']

def loss(params, x, y):
    logits = forward(params, x)
    return jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, logits.shape[-1])))

@jit
def accuracy(params, x, y):
    return jnp.mean(jnp.argmax(forward(params, x), axis=-1) == y)

@pytest.fixture
def setup_data():
    input_dim = 100
    hidden_dim = 64
    output_dim = 2
    layer_sizes = [input_dim, hidden_dim, hidden_dim, output_dim]
    key = random.PRNGKey(0)
    params = init_network(layer_sizes, key)
    X, y = generate_toy_data(input_dim=input_dim)
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    return params, layer_sizes, key, X_train, y_train, X_test, y_test

def train_and_evaluate(params, optimizer, X_train, y_train, X_test, y_test, num_epochs=10, batch_size=32):
    opt_state = optimizer.init(params)

    @jit
    def update(params, opt_state, x, y):
        loss_value, grads = jax.value_and_grad(loss)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            params, opt_state, loss_value = update(params, opt_state, batch_X, batch_y)

    test_accuracy = accuracy(params, X_test, y_test)
    return test_accuracy, loss_value

def test_galore_correctness(setup_data):
    params, layer_sizes, key, X_train, y_train, X_test, y_test = setup_data
    galore_opt = optax_galore.galore(learning_rate=0.001, rank=32)
    adam_opt = optax.adam(learning_rate=0.001)

    galore_accuracy, galore_loss = train_and_evaluate(params, galore_opt, X_train, y_train, X_test, y_test)
    params = init_network(layer_sizes, key)  # Reset params
    adam_accuracy, adam_loss = train_and_evaluate(params, adam_opt, X_train, y_train, X_test, y_test)

    print(f"GaLore - Accuracy: {galore_accuracy:.4f}, Loss: {galore_loss:.4f}")
    print(f"Adam   - Accuracy: {adam_accuracy:.4f}, Loss: {adam_loss:.4f}")

    assert galore_accuracy > 0.6, "GaLore should achieve reasonable accuracy"
    assert abs(galore_accuracy - adam_accuracy) < 0.1, "GaLore should perform similarly to Adam"

def test_galore_performance(setup_data):
    params, layer_sizes, key, X_train, y_train, X_test, y_test = setup_data
    galore_opt = optax_galore.galore(learning_rate=0.001, rank=32)
    adam_opt = optax.adam(learning_rate=0.001)

    galore_start = time.time()
    galore_accuracy, _ = train_and_evaluate(params, galore_opt, X_train, y_train, X_test, y_test)
    galore_time = time.time() - galore_start

    params = init_network(layer_sizes, key)  # Reset params
    adam_start = time.time()
    adam_accuracy, _ = train_and_evaluate(params, adam_opt, X_train, y_train, X_test, y_test)
    adam_time = time.time() - adam_start

    print(f"GaLore - Time: {galore_time:.2f}s, Accuracy: {galore_accuracy:.4f}")
    print(f"Adam   - Time: {adam_time:.2f}s, Accuracy: {adam_accuracy:.4f}")

    assert galore_time < adam_time * 1.5, "GaLore should not be significantly slower than Adam"

def test_memory_usage(setup_data):
    params, layer_sizes, key, X_train, y_train, X_test, y_test = setup_data

    def train_step(optimizer):
        opt_state = optimizer.init(params)
        batch_X, batch_y = X_train[:32], y_train[:32]

        @jit
        def update(params, opt_state, x, y):
            loss_value, grads = jax.value_and_grad(loss)(params, x, y)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        # Compile and run once
        params, opt_state = update(params, opt_state, batch_X, batch_y)

        # Measure peak memory on second run
        with jax.profiler.profile(memory_profiler=True):
            params, opt_state = update(params, opt_state, batch_X, batch_y)

        mem_stats = memory_stats()
        return mem_stats['peak_bytes'] / (1024 ** 2)  # Convert to MB

    galore_opt = optax_galore.galore(learning_rate=0.001, rank=32)
    adam_opt = optax.adam(learning_rate=0.001)

    galore_memory = train_step(galore_opt)
    params = init_network(layer_sizes, key)  # Reset params
    adam_memory = train_step(adam_opt)

    print(f"GaLore peak memory usage: {galore_memory:.2f} MB")
    print(f"Adam peak memory usage: {adam_memory:.2f} MB")

    assert galore_memory < adam_memory, "GaLore should use less memory than Adam"
