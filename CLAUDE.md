# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`csa-31-ml` is a Keras-based machine learning experimentation framework for managing and tracking model training experiments. The project provides a structured workflow for running multiple experiments with different models and datasets while automatically tracking and saving results.

## Development Commands

### Running Experiments
```bash
# Run a quick test with MNIST data
uv run experiment_test.py

# Run an experiment with custom data and model
uv run experiment_framework.py

# Save a new model to /models
uv run save_model_example.py
```

### Package Management
The project uses `uv` for fast Python package management:
```bash
# Install dependencies
uv pip install <package-name>

# Run Python scripts with uv
uv run <script.py>
```

## Project Structure

### Core Framework Files
- `experiment.py` - Core `experiment` class that manages training, evaluation, and result persistence
- `experiment_framework.py` - Template for running experiments with custom data and models
- `experiment_test.py` - Working example using Keras MNIST dataset

### Helper Files
- `save_model_example.py` - Example showing how to create and save models to /models
- `main.py` - Original entry point (can be removed if not needed)

### Directories
- `/models` - Store compiled Keras models (.keras files)
- `/results` - Experiment results (auto-generated JSON and summary text files)
- `/data` - Place for pickle files containing dataframes

## Experiment Workflow

### 1. Create and Save a Model
```python
# Define model architecture
model = keras.Sequential([...])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save to /models directory
model.save('models/my_model.keras')
```

### 2. Prepare Your Data
- Save dataframe as pickle file in `/data` directory
- Data can be in any format, will be processed in the framework

### 3. Configure experiment_framework.py
Set the following variables:
- `DATA_PATH` - Path to your pickle file
- `MODEL_NAME` - Name of model in /models (without .keras extension)
- `EXPERIMENT_NAME` - Unique name for this experiment run
- Training parameters (EPOCHS, BATCH_SIZE, LEARNING_RATE, etc.)

### 4. Implement Data Processing
Fill in the data processing section to convert your dataframe into:
- `x_train`, `y_train` - Training features and labels
- `x_test`, `y_test` - Test features and labels

### 5. Run the Experiment
```bash
uv run experiment_framework.py
```

Results will be automatically saved to `/results` with:
- JSON file: Complete metrics, parameters, and training history
- Summary text file: Human-readable report

## Key Classes and Parameters

### experiment class (in experiment.py)
Main class for managing experiments. Key parameters:
- `model` - Compiled Keras model
- `experiment_name` - Identifier for the experiment
- `epochs` - Number of training epochs (default: 10)
- `batch_size` - Training batch size (default: 32)
- `validation_split` - Validation data fraction (default: 0.2)
- `verbose` - Training output verbosity (default: 1)

### Methods
- `run(x_train, y_train, x_test, y_test)` - Train and evaluate model
- Returns: (history, test_results) tuple

## Result Files

Each experiment generates two timestamped files:
1. `{experiment_name}_{timestamp}.json` - Machine-readable complete data
2. `{experiment_name}_{timestamp}_summary.txt` - Human-readable report with:
   - Experiment parameters
   - Model architecture details
   - Training metrics per epoch
   - Final test set performance

## Dependencies

Current dependencies (managed in pyproject.toml):
- `keras>=3.0.0` - Deep learning framework
- `tensorflow` - Backend for Keras (auto-installed with keras)

Additional suggested packages for data processing:
- `scikit-learn` - For train/test splitting and preprocessing
- `pandas` - For dataframe operations
- `numpy` - For numerical operations
