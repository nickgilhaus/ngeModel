import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical
import os

path = "../nge/docs/articles"
files = []

try:
    for filename in os.listdir(path):
        files.append(filename)
except Exception as e:
    print("Error: ", e)
    
print(files)
print("Name of file 1: ", files[0])

for file in files:
    with open(path+"/"+file, "r") as f:
        text = f.read().lower()
        characters = sorted(list(set(text)))
        n_to_char = {n:char for n, char in enumerate(characters)}
        char_to_n = {char:n for n, char in enumerate(characters)}
        
        X = []
        Y = []
        length = len(text)
        seq_length = 100
        for i in range(0, length-seq_length, 1):
            sequence = text[i:i + seq_length]
            label = text[i + seq_length]
            X.append([char_to_n[char] for char in sequence])
            Y.append(char_to_n[label])
            
        print("Test Data - ", Y)
        
        x = (np.reshape(X, (len(X), seq_length, 1)))
        X_modified = (x / float(len(characters)))
        Y_modified = (to_categorical(Y))
        

        model = Sequential()
        model.add(LSTM(400, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(400))
        model.add(Dropout(0.2))
        model.add(Dense(Y_modified.shape[1], activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam")

        string_mapped = X[99]
        # generating characters
        for i in range(seq_length):
            x = np.reshape(string_mapped, (1, len(string_mapped), 1))
            x = x / float(len(characters))

            pred_index = np.argmax(model.predict(x, verbose=0))
            seq = [n_to_char[value] for value in string_mapped]
            string_mapped.append(pred_index)
            string_mapped = string_mapped[1:len(string_mapped)]
                