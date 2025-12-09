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

# =============================================================================
# BUILD MODEL
# =============================================================================

print("Building LSTM model for multivariate time series classification...")
print(f"Input shape: {INPUT_SHAPE} (timesteps={TIMESTEPS}, features={NUM_FEATURES})")
print(f"Output classes: {NUM_CLASSES}")

model = models.Sequential([
    # --- C1 ---
    layers.Conv2D(
        64, (3, 3),
        activation='relu',
        strides=1,
        padding='same',
        input_shape=input_shape,
        name='C1_Conv2D'
    ),
    
    # --- S2 ---
    layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2,
        padding='same',
        name='S2_MaxPool'
    ),
    
    # --- C3 ---
    layers.Conv2D(
        128,
        (5, 5),
        activation='relu',
        strides=1,
        padding='same',
        name='C3_Conv2D'
    ),
    
    # --- S4 ---
    layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=2,
        padding='same',
        name='S4_MaxPool'
    ),
    
    # Flatten before fully connected layers
    layers.Flatten(),
    
    # --- F5 ---
    layers.Dense(256, activation='relu', name='F5_Dense'),
    
    # --- F6 ---
    layers.Dense(128, activation='relu', name='F6_Dense'),
    
    # --- F7 (Output Layer) ---
    # For 3 categories, use softmax instead of sigmoid
    layers.Dense(NUM_CLASSES, activation='softmax', name='F7_Output')
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
