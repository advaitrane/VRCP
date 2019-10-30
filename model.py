import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def get_model():
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))
    
    return model