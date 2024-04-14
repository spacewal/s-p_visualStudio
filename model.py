import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X_train, y_train):
    model = build_model((X_train.shape[1], 1))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        ModelCheckpoint('best_model.keras', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    ]
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=callbacks)
    return model
