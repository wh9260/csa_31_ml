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
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data configuration
DATA_PATH = "/mnt/1f0bc9d3-e80c-404a-8c1c-98d7b9f51c5c/csa_31_data/04_samples/epds_017_all.pkl"  # Path to pickle file containing dataframe

# Model configuration
MODEL_NAME = "lstm_multivariate_classifier"  # Name of the model file in /models (without .keras extension)

# Experiment identification
EXPERIMENT_NAME = "LSTM_01"

# Training parameters (with default values)
EPOCHS = 100
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

# Configuration
FEATURE_COLUMNS = ['press_flow', 'spo2', 'rip_abdomen', 'rip_thorax', 'rip_sum']
LABEL_COLUMN = 'score'
TEST_SIZE = 0.2
RANDOM_STATE = 42
PAD_SEQUENCES = True
MAX_TIMESTEPS = None  # None = use longest sequence

print("\n" + "=" * 70)
print("PROCESSING MULTIVARIATE TIME SERIES DATA")
print("=" * 70)

# Step 1: Check sequence lengths
print("\n1. Analyzing time series lengths...")
sequence_lengths = []
for col in FEATURE_COLUMNS:
    lengths = df[col].apply(len)
    sequence_lengths.append(lengths)
    print(f"   {col}: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}")

# Determine target timesteps
all_lengths = np.concatenate([lens.values for lens in sequence_lengths])
min_length = int(all_lengths.min())
max_length = int(all_lengths.max())
target_timesteps = MAX_TIMESTEPS if MAX_TIMESTEPS else max_length

print(f"\n   Target timesteps: {target_timesteps}")
if max_length != min_length:
    print(f"   Strategy: Padding shorter sequences to {target_timesteps}")

# Step 2: Stack features into 3D array
print("\n2. Stacking features into 3D array...")

def process_sequence(row, target_length):
    """Process a single sample's time series data."""
    features = [row[col] for col in FEATURE_COLUMNS]
    processed_features = []

    for feat in features:
        feat_array = np.array(feat, dtype=np.float32)
        if len(feat_array) < target_length:
            feat_array = np.pad(feat_array, (0, target_length - len(feat_array)), mode='constant')
        elif len(feat_array) > target_length:
            feat_array = feat_array[:target_length]
        processed_features.append(feat_array)

    return np.column_stack(processed_features)

X = np.array([process_sequence(df.iloc[i], target_timesteps) for i in range(len(df))])
print(f"   X shape: {X.shape} (samples, timesteps, features)")

# Step 3: Extract labels
print("\n3. Processing labels...")
y = df[LABEL_COLUMN].values
unique_classes = np.unique(y)
num_classes = len(unique_classes)
print(f"   Original classes: {unique_classes}, Count: {num_classes}")
for cls in unique_classes:
    count = np.sum(y == cls)
    print(f"      {cls}: {count} ({count/len(y)*100:.1f}%)")

# Create mapping from original labels to 0-indexed labels
label_to_index = {label: idx for idx, label in enumerate(unique_classes)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
print(f"   Label mapping: {label_to_index}")

# Remap labels to 0-indexed
y_remapped = np.array([label_to_index[label] for label in y])

# Step 4: Split train/test
print("\n4. Splitting into train and test sets...")
x_train, x_test, y_train_raw, y_test_raw = train_test_split(
    X, y_remapped, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_remapped
)
print(f"   Train: {len(x_train)}, Test: {len(x_test)}")

# Step 5: One-hot encode
print("\n5. One-hot encoding labels...")
y_train = keras.utils.to_categorical(y_train_raw, num_classes)
y_test = keras.utils.to_categorical(y_test_raw, num_classes)
print(f"   One-hot encoded shapes: train={y_train.shape}, test={y_test.shape}")

# Step 6: Normalize
print("\n6. Normalizing features...")
train_mean = x_train.mean(axis=(0, 1), keepdims=True)
train_std = x_train.std(axis=(0, 1), keepdims=True) + 1e-8
x_train = (x_train - train_mean) / train_std
x_test = (x_test - train_mean) / train_std
print(f"   Applied z-score normalization")

# Save normalization stats for inference
print("\n7. Saving normalization statistics for inference...")
norm_stats_path = Path(__file__).parent / "models" / "normalization_stats.npz"
np.savez(norm_stats_path, mean=train_mean.squeeze(), std=train_std.squeeze())
print(f"   Saved to: {norm_stats_path}")

print(f"\n{'='*70}")
print("DATA PROCESSING COMPLETE")
print(f"{'='*70}")

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

# Verify model input shape matches data
print("\nVerifying model input shape...")
expected_shape = model.input_shape[1:]  # Remove batch dimension
actual_shape = x_train.shape[1:]
print(f"  Model expects: {expected_shape}")
print(f"  Data provides: {actual_shape}")

if expected_shape != actual_shape:
    raise ValueError(
        f"Shape mismatch! Model expects {expected_shape} but data has {actual_shape}.\n"
        f"Please regenerate the model with TIMESTEPS={x_train.shape[1]} in save_model_lstm.py"
    )
print("  ✓ Shapes match!")

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
