"""
Demonstration of Hyperband hyperparameter tuning with MNIST data.

This demonstrates the complete Hyperband workflow:
1. Create model builder and hyperparameter search space
2. Load MNIST data
3. Run Hyperband search to find optimal hyperparameters
4. Train best model and save results
"""
import keras
import numpy as np
from experiment_hyperband import experiment_hyperband


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "mnist_hyperband_demo"
EXPERIMENT_NAME = "mnist_hyperband_experiment"

# Hyperband parameters
MAX_EPOCHS = 10  # Maximum epochs per trial (keep small for demo)
HYPERBAND_ITERATIONS = 1  # Number of full Hyperband iterations
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
OBJECTIVE = 'val_accuracy'  # Metric to optimize

# =============================================================================
# STEP 1: CREATE MODEL BUILDER (normally done with save_model_hyperband.py)
# =============================================================================

print("="*70)
print("STEP 1: Creating model builder and hyperparameter space")
print("="*70)

# For this demo, we'll create a simple CNN model builder inline
# In practice, use save_model_hyperband.py to generate these files

import json
from pathlib import Path

models_dir = Path(__file__).parent / "models"
models_dir.mkdir(exist_ok=True)

# Create model builder function
builder_code = '''"""
Model builder for MNIST Hyperband demo.
"""
import keras
from keras import layers


def build_model(hp):
    """Build CNN model with tunable hyperparameters."""

    # Hyperparameters to tune
    conv_units_1 = hp.Int('conv_units_1', min_value=16, max_value=64, step=16)
    conv_units_2 = hp.Int('conv_units_2', min_value=32, max_value=128, step=32)
    dense_units = hp.Int('dense_units', min_value=64, max_value=256, step=64)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[0.0001, 0.0005, 0.001])

    # Build model
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(conv_units_1, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(conv_units_2, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(dropout_rate),
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ], name='mnist_hyperband_demo')

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
'''

builder_path = models_dir / f"{MODEL_NAME}_builder.py"
with open(builder_path, 'w') as f:
    f.write(builder_code)

print(f"✓ Model builder saved to: {builder_path}")

# Create hyperparameter space JSON
hyperparams = {
    "model_config": {
        "input_shape": [28, 28, 1],
        "num_classes": 10
    },
    "hyperparameters": [
        {"name": "conv_units_1", "type": "int", "min_value": 16, "max_value": 64, "step": 16},
        {"name": "conv_units_2", "type": "int", "min_value": 32, "max_value": 128, "step": 32},
        {"name": "dense_units", "type": "int", "min_value": 64, "max_value": 256, "step": 64},
        {"name": "dropout_rate", "type": "float", "min_value": 0.2, "max_value": 0.5, "step": 0.1},
        {"name": "learning_rate", "type": "choice", "values": [0.0001, 0.0005, 0.001]}
    ]
}

hyperparams_path = models_dir / f"{MODEL_NAME}_hyperparams.json"
with open(hyperparams_path, 'w') as f:
    json.dump(hyperparams, f, indent=2)

print(f"✓ Hyperparameter space saved to: {hyperparams_path}")
print(f"✓ Search space: {len(hyperparams['hyperparameters'])} hyperparameters")

# =============================================================================
# STEP 2: LOAD AND PREPARE DATA
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Loading and preparing MNIST data")
print("="*70)

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., None]
x_test = x_test[..., None]

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(f"✓ Training samples: {len(x_train)}")
print(f"✓ Test samples: {len(x_test)}")
print(f"✓ Input shape: {x_train.shape[1:]}")

# =============================================================================
# STEP 3: RUN HYPERBAND EXPERIMENT
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Running Hyperband hyperparameter search")
print("="*70)
print("\nThis will search for the best hyperparameters using Hyperband.")
print("The search may take several minutes...\n")

# Create experiment
exp = experiment_hyperband(
    model_name=MODEL_NAME,
    experiment_name=EXPERIMENT_NAME,
    max_epochs=MAX_EPOCHS,
    hyperband_iterations=HYPERBAND_ITERATIONS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    objective=OBJECTIVE,
    verbose=1
)

# Run experiment
history, test_results, best_hyperparams = exp.run(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test
)

# =============================================================================
# STEP 4: DISPLAY RESULTS
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Hyperband Search Results")
print("="*70)

print("\n1. OPTIMAL HYPERPARAMETERS:")
print("-" * 70)
for hp_name, hp_value in best_hyperparams.items():
    print(f"   {hp_name}: {hp_value}")

print("\n2. FINAL TRAINING METRICS (with optimal hyperparameters):")
print("-" * 70)
for metric_name, values in history.history.items():
    print(f"   Final {metric_name}: {values[-1]:.6f}")

print("\n3. TEST SET PERFORMANCE:")
print("-" * 70)
for metric_name, value in test_results.items():
    print(f"   {metric_name}: {value:.6f}")

print("\n" + "="*70)
print("HYPERBAND DEMO COMPLETE!")
print("="*70)
print(f"\nResults saved in: results/{EXPERIMENT_NAME}_*")
print("Files saved:")
print("  - JSON with full results and optimal hyperparameters")
print("  - Text summary")
print("  - Training history plot")
print("  - Best model (.keras file)")
print("="*70)
