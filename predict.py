"""
Simple prediction tool for loading a model and running predictions on data.

Usage:
    uv run predict.py

Configure the MODEL_PATH and DATA_PATH variables below, then run the script
to generate predictions for each sample in your dataset.
"""
import pickle
from pathlib import Path
import keras
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to the saved model (.keras file)
MODEL_PATH = "models/your_model.keras"

# Path to the data file (pickle file containing data)
DATA_PATH = "data/your_data.pkl"

# Output path for predictions (optional - set to None to skip saving)
OUTPUT_PATH = "predictions.pkl"

# Batch size for prediction (adjust based on memory constraints)
BATCH_SIZE = 32


# =============================================================================
# LOAD MODEL
# =============================================================================

print(f"Loading model from: {MODEL_PATH}")
model_path = Path(MODEL_PATH)

if not model_path.exists():
    raise FileNotFoundError(
        f"Model not found at: {MODEL_PATH}\n"
        f"Please provide a valid path to a .keras model file."
    )

model = keras.models.load_model(MODEL_PATH)

print("\nModel architecture:")
model.summary()


# =============================================================================
# LOAD DATA
# =============================================================================

print(f"\nLoading data from: {DATA_PATH}")
data_path = Path(DATA_PATH)

if not data_path.exists():
    raise FileNotFoundError(
        f"Data file not found at: {DATA_PATH}\n"
        f"Please provide a valid path to your data file."
    )

with open(DATA_PATH, 'rb') as f:
    data = pickle.load(f)

print(f"Data loaded successfully")
print(f"Data type: {type(data)}")

# If data is a dataframe, convert to numpy array
if hasattr(data, 'values'):
    print(f"Data shape (from dataframe): {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    # TODO: You may need to process the dataframe before prediction
    # For example: data = data.drop('target_column', axis=1).values
    X = data.values
elif isinstance(data, (np.ndarray, list, tuple)):
    X = np.array(data)
    print(f"Data shape: {X.shape}")
else:
    X = data
    print(f"Data loaded as type: {type(data)}")


# =============================================================================
# RUN PREDICTIONS
# =============================================================================

print(f"\nRunning predictions...")
print(f"Using batch size: {BATCH_SIZE}")

predictions = model.predict(X, batch_size=BATCH_SIZE, verbose=1)

print(f"\nPredictions completed!")
print(f"Predictions shape: {predictions.shape}")
print(f"Sample predictions (first 5):")
print(predictions[:5])


# =============================================================================
# SAVE PREDICTIONS (OPTIONAL)
# =============================================================================

if OUTPUT_PATH:
    print(f"\nSaving predictions to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(predictions, f)
    print("Predictions saved successfully!")

    # Also save as text file for easy viewing
    txt_path = Path(OUTPUT_PATH).with_suffix('.txt')
    np.savetxt(txt_path, predictions, fmt='%.6f')
    print(f"Predictions also saved as text to: {txt_path}")


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)
print(f"Total samples: {len(predictions)}")
print(f"Prediction shape: {predictions.shape}")

if predictions.ndim == 2 and predictions.shape[1] > 1:
    # Multi-class classification
    predicted_classes = np.argmax(predictions, axis=1)
    print(f"\nPredicted class distribution:")
    unique, counts = np.unique(predicted_classes, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples ({count/len(predictions)*100:.1f}%)")

    print(f"\nConfidence statistics:")
    max_probs = np.max(predictions, axis=1)
    print(f"  Mean confidence: {np.mean(max_probs):.4f}")
    print(f"  Median confidence: {np.median(max_probs):.4f}")
    print(f"  Min confidence: {np.min(max_probs):.4f}")
    print(f"  Max confidence: {np.max(max_probs):.4f}")

elif predictions.ndim == 2 and predictions.shape[1] == 1:
    # Single output (regression or binary classification)
    predictions_flat = predictions.flatten()
    print(f"\nPrediction statistics:")
    print(f"  Mean: {np.mean(predictions_flat):.4f}")
    print(f"  Median: {np.median(predictions_flat):.4f}")
    print(f"  Min: {np.min(predictions_flat):.4f}")
    print(f"  Max: {np.max(predictions_flat):.4f}")
    print(f"  Std: {np.std(predictions_flat):.4f}")

print("=" * 70)
