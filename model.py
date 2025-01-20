import time 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pickle

start_time = time.time()

def preprocessing(filepath):
    data = pd.read_csv(filepath)
    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #split of data
    scaler = StandardScaler() #scaling
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def modelling(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),  # Define the input shape
        Dense(64, activation='sigmoid'),
        Dense(32, activation='sigmoid'),
        Dense(1, activation='sigmoid')
    ])

    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train(model, X_train, X_test, y_train, y_test):
    history = model.fit(X_train, y_train,
                       epochs=25,
                       batch_size=4,
                       validation_split=0.2,
                       verbose=1)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test) #evaluation
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    return history

def save(model, scaler, model_path, scaler_path):
    with open(model_path, 'wb') as f: #save model
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f: #save scaler for future inference
        pickle.dump(scaler, f)

if __name__ == "__main__":
    DATA_PATH = "/home/redsat/Téléchargements/ecg2.csv" #dataset/ecg.csv
    MODEL_PATH = "/home/redsat/Téléchargements/model.pkl"
    SCALER_PATH = "/home/redsat/Téléchargements/scaler.pkl"
    X_train, X_test, y_train, y_test, scaler = preprocessing(DATA_PATH) 
    model = modelling(X_train.shape[1])
    history = train(model, X_train, X_test, y_train, y_test)
    save(model, scaler, MODEL_PATH, SCALER_PATH)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.4f} seconds")