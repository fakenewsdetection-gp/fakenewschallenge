import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.optimizers import Adam

print(tf.VERSION)
print(tf.keras.__version__)


def build_mlp(input_dim, num_classes, hidden_layers_dim,
                dropout_rate=0.6, learning_rate=0.01):
    model = Sequential()
    model.add(InputLayer(input_shape=(input_dim, )))
    for layer_dim in hidden_layers_dim:
        model.add(Dense(layer_dim))
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=learning_rate), loss='sparse_categorical_crossentropy')
    model.summary()
    return model
