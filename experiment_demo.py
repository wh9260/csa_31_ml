"""
DEMONSTRATION: Experiment framework using MNIST data from pickle file.

This demonstrates the complete workflow:
1. Load data from pickle file
2. Process dataframe into training/test datasets
3. Load pre-trained model from /models
4. Run experiment and save results
"""
import pickle
from pathlib import Path
import numpy as np
import keras
from class_experiment import experiment


# =============================================================================
# CONFIGURATION
# =============================================================================

# Data configuration
DATA_PATH = "data/mnist_data.pkl"  # Path to pickle file containing dataframe

# Model configuration
MODEL_NAME = "example_cnn"  # Name of the model file in /models (without .keras extension)

# Experiment identification
EXPERIMENT_NAME = "mnist_framework_demo"

# Training parameters (with default values)
EPOCHS = 3  # Using fewer epochs for demo
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 0.001
VERBOSE = 1

# Additional training parameters (optional)
SHUFFLE = True
CLASS_WEIGHT = None  # Can be set to a dictionary for imbalanced datasets

# =============================================================================
# DATA LOADING
# =============================================================================

print(f"Loading data from: {DATA_PATH}")
with open(DATA_PATH, 'rb') as f:
    df = pickle.load(f)

print(f"Data loaded. Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# =============================================================================
# DATA PROCESSING
# =============================================================================
print("\n" + "=" * 70)
print("PROCESSING DATA")
print("=" * 70)

# Step 1: Split by train/test based on 'split' column
print("\n1. Splitting data into train and test sets...")
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']
print(f"   Train samples: {len(train_df)}")
print(f"   Test samples: {len(test_df)}")

# Step 2: Extract features and labels
print("\n2. Extracting features and labels...")
# Stack the image data arrays
x_train = np.stack(train_df['image_data'].values)
y_train = train_df['label'].values

x_test = np.stack(test_df['image_data'].values)
y_test = test_df['label'].values

print(f"   x_train shape: {x_train.shape}")
print(f"   y_train shape: {y_train.shape}")
print(f"   x_test shape: {x_test.shape}")
print(f"   y_test shape: {y_test.shape}")

# Step 3: Preprocess - normalize pixel values
print("\n3. Normalizing pixel values to [0, 1] range...")
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Step 4: Reshape for CNN input (add channel dimension)
print("\n4. Reshaping for CNN input (28x28x1)...")
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
print(f"   x_train shape: {x_train.shape}")
print(f"   x_test shape: {x_test.shape}")

# Step 5: Convert labels to categorical (one-hot encoding)
print("\n5. Converting labels to categorical (one-hot encoding)...")
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(f"   y_train shape: {y_train.shape}")
print(f"   y_test shape: {y_test.shape}")

# Summary
print(f"\n{'='*70}")
print("DATA PROCESSING COMPLETE")
print(f"{'='*70}")
print(f"Final data shapes:")
print(f"  x_train: {x_train.shape} - Training images")
print(f"  y_train: {y_train.shape} - Training labels (one-hot)")
print(f"  x_test: {x_test.shape} - Test images")
print(f"  y_test: {y_test.shape} - Test labels (one-hot)")
print(f"{'='*70}\n")

# =============================================================================
# MODEL LOADING
# =============================================================================

models_dir = Path(__file__).parent / "models"
model_path = models_dir / f"{MODEL_NAME}.keras"

if not model_path.exists():
    raise FileNotFoundError(
        f"Model not found at: {model_path}\n"
        f"Please ensure your model is saved in the /models directory.\n"
        f"You can save a model using: model.save('models/{MODEL_NAME}.keras')"
    )

print(f"Loading model from: {model_path}")
model = keras.models.load_model(model_path)

print("\nModel architecture:")
model.summary()

# Verify model is compiled
if not model.optimizer:
    print("\nWarning: Model is not compiled. Compiling with default parameters...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

# =============================================================================
# RUN EXPERIMENT
# =============================================================================

# Create experiment instance with all parameters
exp = experiment(
    model=model,
    experiment_name=EXPERIMENT_NAME,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=VERBOSE,
    shuffle=SHUFFLE,
    class_weight=CLASS_WEIGHT
)

# Run the experiment
history, test_results = exp.run(
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
print(f"Check results/{EXPERIMENT_NAME}_*_summary.txt for detailed report")
print("=" * 70)
