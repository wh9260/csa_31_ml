"""
Helper script to save normalization statistics for inference.

Run this after training to save the mean and std used for normalization,
so they can be reused during real-time detection.
"""
import numpy as np
from pathlib import Path


def save_normalization_stats(train_mean, train_std, output_path="models/normalization_stats.npz"):
    """
    Save training normalization statistics.

    Args:
        train_mean: Mean values used for normalization (shape: (1, 1, num_features) or (num_features,))
        train_std: Std values used for normalization (shape: (1, 1, num_features) or (num_features,))
        output_path: Where to save the stats
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Squeeze to remove unnecessary dimensions
    if train_mean.ndim > 1:
        train_mean = train_mean.squeeze()
    if train_std.ndim > 1:
        train_std = train_std.squeeze()

    np.savez(output_path, mean=train_mean, std=train_std)

    print(f"Normalization statistics saved to: {output_path}")
    print(f"  Mean shape: {train_mean.shape}")
    print(f"  Std shape: {train_std.shape}")
    print(f"  Mean values: {train_mean}")
    print(f"  Std values: {train_std}")


# Example usage:
if __name__ == "__main__":
    # Example: Save dummy statistics
    # In practice, get these from your training script

    # For 5 features (press_flow, spo2, rip_abdomen, rip_thorax, rip_sum)
    num_features = 5

    # Example values (replace with actual values from training)
    train_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    train_std = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    print("="*70)
    print("SAVE NORMALIZATION STATISTICS")
    print("="*70)
    print("\nNOTE: This is an example. You should extract these values from")
    print("      your actual training process in experiment_framework_LSTM.py")
    print("\nTo get these values from your training script, add this code")
    print("after the normalization step (around line 132):")
    print()
    print("  # Save normalization stats for inference")
    print("  from save_normalization_stats import save_normalization_stats")
    print("  save_normalization_stats(train_mean, train_std)")
    print()
    print("="*70)

    # Save example stats
    save_normalization_stats(train_mean, train_std)
