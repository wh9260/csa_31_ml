import pickle
import keras
from experiment import experiment
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Data path and file

test_data_file_path = "/mnt/1f0bc9d3-e80c-404a-8c1c-98d7b9f51c5c/csa_31_data/03_datasets/"
test_data_file_name = "combined_data_05_32hz_df.csv"

types = ['press_flow', 'spo2', 'rip_abdomen', 'rip_thorax', 'rip_sum']

# Load model

model_path = "models/"
model_name = "lstm_multivariate_classifier.keras"



model = load_model(model_path + model_name)

print(model.summary())

print('Model input shape = ', model.input_shape)

# Load data

df = pd.read_csv(test_data_file_path + test_data_file_name)

X = np.stack(df[types].apply(lambda row: np.stack(row.values, axis=-1), axis=1))

X = np.squeeze(X)

X = X.astype('float32')

X = (X - np.mean(X)) / np.std(X)

print(X)

# Determine where to run from and to, and where to locate prediction

t_start_index = 0

sample_length = model.input_shape[1]

t_end_index = len(X) - sample_length - 1

prediction = []
prediction_index = []

for i in tqdm(range(t_start_index, t_end_index, 32)):
    
    data = X[i:i+sample_length,:]
    
    data = np.expand_dims(data, axis=0)
    
    new_prediction = model.predict(data, verbose=0)
    
    if len(prediction) == 0:
        prediction = new_prediction
        prediction_index = i + sample_length
    else:        
        prediction = np.vstack([prediction, new_prediction])
        prediction_index = np.vstack([prediction_index, i + sample_length])
        
print()