import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

print(tf.VERSION)
print(tf.keras.__version__)


def build_mlp(input_dim, num_classes, hidden_layers_dim,
                dropout_rate=0.5, learning_rate=0.01):
    model = Sequential()
    for i, layer_dim in enumerate(hidden_layers_dim):
        if i == 0:
            model.add(Dense(layer_dim, input_shape=(input_dim, ), activation='relu'))
        else:
            model.add(Dense(layer_dim, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=learning_rate), loss='sparse_categorical_crossentropy')
    model.summary()
    return model
