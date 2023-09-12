import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model

# basic CNN model
def get_model():
    def get_preprocess():
        # rescaling, 1 / 255
        preprocessing_layer = tf.keras.models.Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
            ])
        
        return preprocessing_layer
    
    preprocessing_layer = get_preprocess()
    
    inputs = Input(shape = (28, 28, 1))
    preprocessing_inputs = preprocessing_layer(inputs)
    
    x = Conv2D(filters = 32, kernel_size = (3, 3), activation='relu')(preprocessing_inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(filters = 64, kernel_size = (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(filters = 64, kernel_size =(3, 3), activation='relu')(x)
    
    x = Flatten()(x)
    x = Dense(64, activation = 'relu')(x)
    outputs = Dense(10, activation = 'softmax')(x)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    return model