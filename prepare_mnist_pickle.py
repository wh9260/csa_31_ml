"""
Script to prepare MNIST data and save as a pickle file.

This demonstrates the workflow of saving data as a pickle file
that can then be loaded by experiment_framework.py
"""
import pickle
import pandas as pd
import numpy as np
import keras
from pathlib import Path


print("Loading MNIST dataset from Keras...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"Original shapes:")
print(f"  x_train: {x_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  x_test: {x_test.shape}")
print(f"  y_test: {y_test.shape}")

# Create a dataframe with all the data
# For demonstration, we'll store the data in a format that mimics
# a real-world scenario where data comes in a dataframe

print("\nCreating combined dataframe...")

# Flatten images for dataframe storage
train_data = []
for i in range(len(x_train)):
    row = {
        'image_data': x_train[i].flatten(),  # Flatten 28x28 to 784
        'label': y_train[i],
        'split': 'train'
    }
    train_data.append(row)

test_data = []
for i in range(len(x_test)):
    row = {
        'image_data': x_test[i].flatten(),
        'label': y_test[i],
        'split': 'test'
    }
    test_data.append(row)

# Combine into single dataframe
all_data = train_data + test_data
df = pd.DataFrame(all_data)

print(f"\nDataframe shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nDataframe info:")
print(df.info())
print(f"\nLabel distribution:")
print(df['label'].value_counts().sort_index())

# Save as pickle
data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

pickle_path = data_dir / "mnist_data.pkl"
print(f"\nSaving dataframe to: {pickle_path}")

with open(pickle_path, 'wb') as f:
    pickle.dump(df, f)

print(f"Successfully saved! File size: {pickle_path.stat().st_size / 1024 / 1024:.2f} MB")

print("\n" + "=" * 70)
print("Data preparation complete!")
print("=" * 70)
print(f"\nYou can now use this data in experiment_framework.py by setting:")
print(f'  DATA_PATH = "data/mnist_data.pkl"')
print("=" * 70)
