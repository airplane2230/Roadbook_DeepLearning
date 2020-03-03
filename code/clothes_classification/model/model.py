from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

def get_model():
    inputs = Input(shape=(112, 112, 3))

    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(11, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    return model
