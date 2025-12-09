"""
Example script showing how to create and save a model to /models directory.

This demonstrates the workflow:
1. Define a model architecture
2. Compile the model
3. Save it to /models directory
4. The saved model can then be used with experiment_framework.py
"""
import keras
from keras import layers
from pathlib import Path


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_NAME = "LSTM_01"  # Name for saving the model
INPUT_SHAPE = (28, 28, 1)   # Example: MNIST-like input
NUM_CLASSES = 10            # Example: 10 classes

# =============================================================================
# BUILD MODEL
# =============================================================================

print("Building model...")
model = keras.Sequential([
    layers.Input(shape=INPUT_SHAPE),

    # First convolutional block
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Second convolutional block
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Dense layers
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax")
], name=MODEL_NAME)

# =============================================================================
# COMPILE MODEL
# =============================================================================

print("Compiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()

# =============================================================================
# SAVE MODEL
# =============================================================================

# Ensure models directory exists
models_dir = Path(__file__).parent / "models"
models_dir.mkdir(exist_ok=True)

# Save the model
model_path = models_dir / f"{MODEL_NAME}.keras"
model.save(model_path)

print(f"\n{'='*70}")
print(f"Model saved successfully to: {model_path}")
print(f"{'='*70}")
print(f"\nYou can now use this model in experiment_framework.py by setting:")
print(f'  MODEL_NAME = "{MODEL_NAME}"')
print(f"{'='*70}")
