"""
Script to create and save an LSTM model for multivariate time series classification.

This model is designed for:
- Input: 5 multivariate time series features (press_flow, spo2, rip_abdomen, rip_thorax, rip_sum)
- Output: 3 classes for classification
- Architecture: LSTM layers for temporal pattern learning

Workflow:
1. Define LSTM model architecture
2. Compile the model
3. Save it to /models directory
4. The saved model can then be used with experiment_framework_LSTM.py
"""
import keras
from keras import layers
from pathlib import Path


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_NAME = "lstm_multivariate_classifier"  # Name for saving the model

# Input configuration
NUM_FEATURES = 5        # Number of time series features (press_flow, spo2, rip_abdomen, rip_thorax, rip_sum)
TIMESTEPS = 128         # Length of each time series (adjust based on your data)
INPUT_SHAPE = (TIMESTEPS, NUM_FEATURES)

# Output configuration
NUM_CLASSES = 3         # Number of classification categories

# Architecture hyperparameters
LSTM_UNITS_1 = 128      # Units in first LSTM layer
LSTM_UNITS_2 = 64       # Units in second LSTM layer
LSTM_UNITS_3 = 32       # Units in second LSTM layer
DROPOUT_RATE = 0.3      # Dropout rate for regularization
DENSE_UNITS = 32           # Units in dense layer before output

# =============================================================================
# BUILD MODEL
# =============================================================================

print("Building LSTM model for multivariate time series classification...")
print(f"Input shape: {INPUT_SHAPE} (timesteps={TIMESTEPS}, features={NUM_FEATURES})")
print(f"Output classes: {NUM_CLASSES}")

model = keras.Sequential([
    layers.Input(shape=INPUT_SHAPE),

    # First LSTM layer - returns sequences for stacking
    layers.LSTM(LSTM_UNITS_1, return_sequences=True),
    layers.Dropout(DROPOUT_RATE),
    
    # Second LSTM layer - returns sequences for stacking
    layers.LSTM(LSTM_UNITS_2, return_sequences=True),
    layers.Dropout(DROPOUT_RATE),

    # Third LSTM layer - returns final output
    layers.LSTM(LSTM_UNITS_3, return_sequences=False),
    layers.Dropout(DROPOUT_RATE),

    # Dense layers for classification
    layers.Dense(DENSE_UNITS, activation="relu"),
    layers.Dropout(DROPOUT_RATE),

    # Output layer
    layers.Dense(NUM_CLASSES, activation="softmax")
], name=MODEL_NAME)

# =============================================================================
# COMPILE MODEL
# =============================================================================

print("\nCompiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()

# =============================================================================
# MODEL INFORMATION
# =============================================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)
print(f"Model name: {MODEL_NAME}")
print(f"\nInput specifications:")
print(f"  - Shape: {INPUT_SHAPE}")
print(f"  - Timesteps: {TIMESTEPS} (length of each time series)")
print(f"  - Features: {NUM_FEATURES} (multivariate channels)")
print(f"    1. press_flow")
print(f"    2. spo2")
print(f"    3. rip_abdomen")
print(f"    4. rip_thorax")
print(f"    5. rip_sum")
print(f"\nOutput specifications:")
print(f"  - Classes: {NUM_CLASSES}")
print(f"  - Activation: softmax (for multi-class classification)")
print(f"\nArchitecture:")
print(f"  - LSTM Layer 1: {LSTM_UNITS_1} units (with sequences)")
print(f"  - LSTM Layer 2: {LSTM_UNITS_2} units")
print(f"  - Dense Layer: {DENSE_UNITS} units")
print(f"  - Dropout: {DROPOUT_RATE}")
print(f"\nTotal parameters: {model.count_params():,}")
print("=" * 70)

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
print(f"\nYou can now use this model in experiment_framework_LSTM.py by setting:")
print(f'  MODEL_NAME = "{MODEL_NAME}"')
print(f'\nIMPORTANT: Make sure your time series data has {TIMESTEPS} timesteps.')
print(f'If your data has a different length, either:')
print(f'  1. Update TIMESTEPS in this script and regenerate the model, OR')
print(f'  2. Pad/truncate your time series to match {TIMESTEPS} timesteps')
print(f"{'='*70}")
