"""
Optimized real-time detection script for time series classification.

Improvements:
- Batch predictions for 10-50x speedup
- Proper normalization matching training
- Result saving and visualization
- Configuration parameters
- Error handling
- Memory-efficient processing
"""
import numpy as np
import pandas as pd
import keras
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

# Data configuration
TEST_DATA_PATH = "/mnt/1f0bc9d3-e80c-404a-8c1c-98d7b9f51c5c/csa_31_data/03_datasets/"
TEST_DATA_FILE = "combined_data_29_32hz_df.csv"
FEATURE_COLUMNS = ['press_flow', 'spo2', 'rip_abdomen', 'rip_thorax', 'rip_sum']

# Model configuration
MODEL_PATH = "models/lstm_multivariate_classifier.keras"
NORMALIZATION_STATS_PATH = "models/normalization_stats.npz"  # Optional: saved training stats

# Sliding window configuration
STEP_SIZE = 32              # Step between predictions (samples)
BATCH_SIZE = 128             # Number of windows to predict at once (KEY FOR SPEED!)
START_INDEX = 0             # Where to start in the data
END_INDEX = None            # None = use all data

# Output configuration
OUTPUT_DIR = "results/realtime_detection"
SAVE_PREDICTIONS = True
SAVE_VISUALIZATION = True
EXPERIMENT_NAME = "realtime_detection"

# Class names (update based on your model)
CLASS_NAMES = {0: "Class_0", 1: "Class_1", 2: "Class_2"}

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_and_validate_model(model_path: str):
    """Load model and display information."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    print("\nModel Information:")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    print(f"  Number of classes: {model.output_shape[-1]}")

    return model


def load_normalization_stats(stats_path: str = None):   
    """Load normalization statistics from training or compute from data."""
    if stats_path and Path(stats_path).exists():
        print(f"Loading normalization stats from: {stats_path}")
        stats = np.load(stats_path)
        return stats['mean'], stats['std']
    else:
        print("Warning: No normalization stats found. Will use global statistics.")
        print("         This may not match training normalization!")
        return None, None


def load_data(data_path: str, filename: str, feature_columns: list):
    """Load and validate data from CSV."""
    full_path = Path(data_path) / filename

    if not full_path.exists():
        raise FileNotFoundError(f"Data file not found: {full_path}")

    print(f"\nLoading data from: {full_path}")
    df = pd.read_csv(full_path)
    #df = df.iloc[0:275500]
    
    # Check if feature columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    print(f"  Data shape: {df.shape}")
    print(f"  Feature columns: {feature_columns}")

    # Extract feature data
    # Assumes CSV has columns with array-like data or individual time points
    try:
        # Try stacking if data is in array format per cell
        X = np.stack(df[feature_columns].apply(
            lambda row: np.stack(row.values, axis=-1), axis=1
        ))
        X = np.squeeze(X)
    except:
        # If that fails, assume data is already in proper format
        X = df[feature_columns].values

    X = X.astype('float32')
    print(f"  Loaded data shape: {X.shape}")

    return X, df


def normalize_data(X: np.ndarray, train_mean=None, train_std=None):
    """Normalize data using training statistics or compute from data."""
    if train_mean is not None and train_std is not None:
        print("\nNormalizing with training statistics...")
        X_normalized = (X - train_mean) / (train_std + 1e-8)
    else:
        print("\nNormalizing with global statistics (WARNING: may not match training)...")
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        X_normalized = (X - mean) / (std + 1e-8)

    print(f"  Normalized data range: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
    return X_normalized


def create_sliding_windows_batch(X: np.ndarray, window_size: int, step_size: int,
                                  start_idx: int = 0, end_idx: int = None):
    """
    Create sliding windows for batch prediction (OPTIMIZED FOR SPEED).

    This pre-creates all windows at once instead of creating them in a loop.
    """
    if end_idx is None:
        end_idx = len(X) - window_size

    # Calculate number of windows
    num_windows = (end_idx - start_idx) // step_size + 1

    print(f"\nCreating sliding windows:")
    print(f"  Window size: {window_size}")
    print(f"  Step size: {step_size}")
    print(f"  Start index: {start_idx}")
    print(f"  End index: {end_idx}")
    print(f"  Total windows: {num_windows}")

    # Pre-allocate array for all windows (MUCH faster than growing in loop)
    windows = np.zeros((num_windows, window_size, X.shape[1]), dtype=np.float32)
    indices = np.zeros(num_windows, dtype=np.int32)

    # Fill windows
    for i, start in enumerate(range(start_idx, end_idx, step_size)):
        windows[i] = X[start:start + window_size]
        indices[i] = start + window_size  # Prediction corresponds to end of window

    return windows, indices


def batch_predict(model, windows: np.ndarray, batch_size: int):
    """
    Predict on windows in batches for maximum speed.

    This is 10-50x faster than predicting one window at a time!
    """
    num_windows = len(windows)
    num_batches = (num_windows + batch_size - 1) // batch_size

    print(f"\nRunning predictions:")
    print(f"  Total windows: {num_windows}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of batches: {num_batches}")

    predictions = []

    # Process in batches with progress bar
    for i in tqdm(range(0, num_windows, batch_size), desc="Predicting"):
        batch = windows[i:i + batch_size]
        batch_pred = model.predict(batch, verbose=0)
        predictions.append(batch_pred)

    # Concatenate all predictions
    predictions = np.vstack(predictions)

    return predictions


def save_results(predictions: np.ndarray, indices: np.ndarray,
                 output_dir: str, experiment_name: str, class_names: dict):
    """Save predictions to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw predictions
    pred_path = output_path / f"{experiment_name}_{timestamp}_predictions.npz"
    np.savez(pred_path, predictions=predictions, indices=indices)
    print(f"\n  Saved predictions: {pred_path}")

    # Save as CSV with class probabilities and predicted class
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_class_names = [class_names.get(c, f"Class_{c}") for c in predicted_classes]

    results_df = pd.DataFrame({
        'index': indices,
        **{f'prob_{class_names.get(i, f"class_{i}")}': predictions[:, i]
           for i in range(predictions.shape[1])},
        'predicted_class': predicted_classes,
        'predicted_class_name': predicted_class_names,
        'confidence': np.max(predictions, axis=1)
    })

    csv_path = output_path / f"{experiment_name}_{timestamp}_predictions.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    # Save summary statistics
    summary = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "total_predictions": int(len(predictions)),
        "class_distribution": {
            class_names.get(i, f"Class_{i}"): int(np.sum(predicted_classes == i))
            for i in range(predictions.shape[1])
        },
        "average_confidence": float(np.mean(np.max(predictions, axis=1))),
        "indices_range": [int(indices[0]), int(indices[-1])]
    }

    summary_path = output_path / f"{experiment_name}_{timestamp}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {summary_path}")

    return results_df, timestamp


def visualize_predictions(predictions: np.ndarray, indices: np.ndarray,
                         output_dir: str, experiment_name: str,
                         timestamp: str, class_names: dict):
    """Create visualization of predictions over time."""
    output_path = Path(output_dir)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # Plot 1: Class probabilities over time
    ax = axes[0]
    for i in range(predictions.shape[1]):
        ax.plot(indices, predictions[:, i], label=class_names.get(i, f"Class {i}"), linewidth=1.5)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Class Probabilities Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 2: Predicted class over time
    ax = axes[1]
    predicted_classes = np.argmax(predictions, axis=1)
    colors = plt.cm.tab10(predicted_classes)
    ax.scatter(indices, predicted_classes, c=colors, s=10, alpha=0.6)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Predicted Class', fontsize=12)
    ax.set_title('Predicted Class Over Time', fontsize=14, fontweight='bold')
    ax.set_yticks(range(predictions.shape[1]))
    ax.set_yticklabels([class_names.get(i, f"Class {i}") for i in range(predictions.shape[1])])
    ax.grid(True, alpha=0.3)

    # Plot 3: Prediction confidence over time
    ax = axes[2]
    confidence = np.max(predictions, axis=1)
    ax.plot(indices, confidence, linewidth=1, alpha=0.7)
    ax.fill_between(indices, confidence, alpha=0.3)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('Prediction Confidence Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    plot_path = output_path / f"{experiment_name}_{timestamp}_visualization.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved visualization: {plot_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("="*70)
    print("OPTIMIZED REAL-TIME DETECTION")
    print("="*70)

    # Load model
    model = load_and_validate_model(MODEL_PATH)
    window_size = model.input_shape[1]

    # Load normalization stats (if available)
    train_mean, train_std = load_normalization_stats(NORMALIZATION_STATS_PATH)

    # Load data
    X, df = load_data(TEST_DATA_PATH, TEST_DATA_FILE, FEATURE_COLUMNS)

    # Normalize data
    X_normalized = normalize_data(X, train_mean, train_std)

    # Create sliding windows (OPTIMIZED)
    end_idx = END_INDEX if END_INDEX else len(X_normalized) - window_size
    windows, indices = create_sliding_windows_batch(
        X_normalized, window_size, STEP_SIZE, START_INDEX, end_idx
    )

    # Batch predict (FAST!)
    predictions = batch_predict(model, windows, BATCH_SIZE)

    print(f"\nPrediction complete!")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  Prediction shape: {predictions.shape}")

    # Display class distribution
    predicted_classes = np.argmax(predictions, axis=1)
    print(f"\nClass distribution:")
    for i in range(predictions.shape[1]):
        count = np.sum(predicted_classes == i)
        percentage = count / len(predicted_classes) * 100
        print(f"  {CLASS_NAMES.get(i, f'Class {i}')}: {count} ({percentage:.1f}%)")

    # Save results
    if SAVE_PREDICTIONS:
        print("\nSaving results...")
        results_df, timestamp = save_results(
            predictions, indices, OUTPUT_DIR, EXPERIMENT_NAME, CLASS_NAMES
        )

    # Create visualization
    if SAVE_VISUALIZATION:
        print("\nCreating visualization...")
        visualize_predictions(
            predictions, indices, OUTPUT_DIR, EXPERIMENT_NAME, timestamp, CLASS_NAMES
        )

    print("\n" + "="*70)
    print("DETECTION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
