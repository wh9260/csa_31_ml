"""
Experiment framework for running Keras model training experiments.

This framework allows you to:
- Load data from pickle files containing dataframes
- Load pre-defined models from the /models directory
- Configure training parameters
- Run experiments and save results
"""
import pickle
from pathlib import Path
import keras
from experiment import experiment


# =============================================================================
# CONFIGURATION
# =============================================================================

# Data configuration
DATA_PATH = "data/your_data.pkl"  # Path to pickle file containing dataframe

# Model configuration
MODEL_NAME = "your_model"  # Name of the model file in /models (without .keras extension)

# Experiment identification
EXPERIMENT_NAME = "experiment_001"

# Training parameters (with default values)
EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
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

# =============================================================================
# DATA PROCESSING
# =============================================================================
# TODO: Process the dataframe into training, validation, and test datasets
#
# This section should:
# 1. Extract features and labels from the dataframe
# 2. Split data into train/validation/test sets (if not already split)
# 3. Perform any necessary preprocessing (scaling, normalization, etc.)
# 4. Reshape data as needed for the model input
#
# Expected outputs:
# - x_train: Training features
# - y_train: Training labels
# - x_test: Test features
# - y_test: Test labels
#
# Example structure:
# -------------------
# # Extract features and labels
# features = df.drop('target_column', axis=1).values
# labels = df['target_column'].values
#
# # Split data
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#     features, labels, test_size=0.2, random_state=42
# )
#
# # Preprocess
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
#
# # Reshape if needed
# # x_train = x_train.reshape(...)
# # x_test = x_test.reshape(...)
# -------------------

x_train = None  # TODO: Replace with processed training features
y_train = None  # TODO: Replace with processed training labels
x_test = None   # TODO: Replace with processed test features
y_test = None   # TODO: Replace with processed test labels

# Validate that data processing is complete
if any(v is None for v in [x_train, y_train, x_test, y_test]):
    raise ValueError(
        "Data processing incomplete. Please implement the data processing "
        "section to convert the dataframe into train/test datasets."
    )

print(f"\nProcessed data shapes:")
print(f"  x_train: {x_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  x_test: {x_test.shape}")
print(f"  y_test: {y_test.shape}")

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

print(f"\nLoading model from: {model_path}")
model = keras.models.load_model(model_path)

print("\nModel architecture:")
model.summary()

# Verify model is compiled
if not model.optimizer:
    print("\nWarning: Model is not compiled. Compiling with default parameters...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",  # Adjust based on your task
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
print("=" * 70)
