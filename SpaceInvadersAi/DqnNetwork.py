import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np

class DqnNetwork:
    def __init__(self, input_shape, action_num):

        self.input_shape = input_shape
        self.action_num = action_num
        self.main_model = self.create_mode()
        self.target_model = self.create_mode()
        self.target_model.set_weights(self.main_model.get_weights())

    def create_mode(self):
       
        model = Sequential()
        model.add(Conv2D(256, (3, 3), input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(self.action_num, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model

    def update_target_network(self):
        self.target_model.set_weights(self.main_model.get_weights())

    def get_qvalues(self, state):        
        return self.main_model.predict(np.array([state]))[0]