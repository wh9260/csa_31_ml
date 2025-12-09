"""
Test/demonstration file for the Experiment class.

This file demonstrates how to use the Experiment class to run
Keras model training experiments with the MNIST dataset.
"""
import keras
from keras import layers
from experiment import experiment


# =============================================================================
# EXPERIMENT PARAMETERS
# =============================================================================

# Experiment identification
EXPERIMENT_NAME = "mnist_cnn_test"

# Training parameters
EPOCHS = 5
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 0.001

# Model architecture parameters
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

# =============================================================================
# DATA PREPARATION
# =============================================================================

print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data
print("Preprocessing data...")
# Scale images to [0, 1] range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Expand dimensions to add channel dimension
x_train = x_train[..., None]  # Shape: (60000, 28, 28, 1)
x_test = x_test[..., None]    # Shape: (10000, 28, 28, 1)

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Number of classes: {NUM_CLASSES}")

# =============================================================================
# MODEL DEFINITION
# =============================================================================

print("\nBuilding model...")
model = keras.Sequential([
    layers.Input(shape=INPUT_SHAPE),

    # First convolutional block
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Second convolutional block
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten and dense layers
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax")
], name="mnist_cnn")

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()

# =============================================================================
# RUN EXPERIMENT
# =============================================================================

# Create experiment instance with all parameters
experiment = experiment(
    model=model,
    experiment_name=EXPERIMENT_NAME,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=1
)

# Run the experiment
history, test_results = experiment.run(
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test
)

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT RESULTS")
print("=" * 70)

print("\nFinal Training Metrics:")
for metric_name, values in history.history.items():
    print(f"  {metric_name}: {values[-1]:.6f}")

print("\nTest Set Performance:")
for metric_name, value in test_results.items():
    print(f"  {metric_name}: {value:.6f}")

print("\n" + "=" * 70)
print(f"Results saved to: results/{EXPERIMENT_NAME}_*.json")
print("=" * 70)
