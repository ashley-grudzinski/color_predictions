import json
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer


def train_data(dataset, color_names, folder_name, filename):
    # integer encode text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([color_names])
    encoded = tokenizer.texts_to_sequences([color_names])[0]

    data_X = list()
    data_y = list()

    # Iterate through every color and add to the data 
    count = 0
    for i, row in dataset.iterrows():
        color_array = [float(row['Red (8 bit)']), float(row['Green (8 bit)']), float(row['Blue (8 bit)'])]
        data_X.append(color_array)
        data_y.append(encoded[i])
        count+=1

    X = np.array(data_X) #X[0] = y name red green blue
    y = np.array(data_y) # name

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1 # 4 OR 5
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # define model
    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape=(3, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # save models
    checkpoint_name = folder_name + '_Weights-LSTM-improvement-{epoch:03d}-{loss:.5f}-bigger.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose = 1, save_best_only = True, mode ='min')
    callbacks_list = [checkpoint]

    # fit model
    # model.fit(X, y, epochs=500, verbose=0, callbacks=callbacks_list)

    # use model
    model.load_weights(filename)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    # demonstrate prediction
    x_input = np.array([13, 43, 122])
    x_input = x_input.reshape((1, 3, n_features))

    x_input+=1
    predicted_char = model.predict(x_input, verbose=0)[0][0]
    rounded = predicted_char.round()
    out_word = ''

    while (out_word == ''):
        for word, index in tokenizer.word_index.items():
            if index == rounded:
                out_word = word
                break
        rounded+=1

        if (rounded > len(tokenizer.word_index.items())):
            rounded = 1
    
    return out_word

if __name__ == "__main__":
    dataset = pd.read_csv('colorhexa_com.csv', encoding = "latin1")

    # make one big string of the color names
    # seperating the adjective descriptive names versus the actual color names
    descriptive_color = ""
    actual_color = ""
    for i, row in dataset.iterrows():
        names = row['Name'].split(' ')
        if len(names) == 1:
            descriptive_color += " pretty"
            actual_color += " " + names[0]
        else:
            for j in range(0, len(names) - 1):
                descriptive_color += " " + names[j]
            actual_color += " " + names[len(names) - 1]

    description = train_data(dataset, descriptive_color, "descriptive", "./descriptive_models/descriptive_Weights-LSTM-improvement-486-2400.41797-bigger.hdf5")
    actual = train_data(dataset, actual_color, "actual", "./actual_models/actual_Weights-LSTM-improvement-493-2459.88354-bigger.hdf5")

    print(description + " " + actual)