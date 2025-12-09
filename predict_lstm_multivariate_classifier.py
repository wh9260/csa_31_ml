import pickle
import keras
from experiment import experiment
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm



# Data path and file

test_data_path = "/mnt/1f0bc9d3-e80c-404a-8c1c-98d7b9f51c5c/csa_31_data/04_samples/epds_017_all.pkl"

types = ['press_flow', 'spo2', 'rip_abdomen', 'rip_thorax', 'rip_sum']

# Load model

model_path = "models/"
model_name = "lstm_multivariate_classifier.keras"

class_names = [0,1,7]

model = load_model(model_path + model_name)

print(model.summary())

print('Model input shape = ', model.input_shape)

# Load data


with open(test_data_path, 'rb') as f:
    df = pickle.load(f)

X = np.stack(df[types].apply(lambda row: np.stack(row.values, axis=-1), axis=1))

X = X.astype('float32')

X = (X - np.mean(X)) / np.std(X)

# Determine where to run from and to, and where to locate prediction

prediction = []
category = df["score"].to_numpy()
    
predictions = model.predict(X, batch_size = 64, verbose = 0)

predicted_classes = np.argmax(predictions, axis=1)

predicted_classes_mapped = [class_names[i] for i in predicted_classes]

print()


df = pd.DataFrame({"Actual": category, "Predicted": predicted_classes_mapped})
cm = pd.crosstab(df["Actual"], df["Predicted"])

cm = cm / len(df)
print(cm)