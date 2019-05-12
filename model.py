import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

print(tf.VERSION)
print(tf.keras.__version__)


def build_mlp(input_dim, num_classes, hidden_layers_dim,
                lambda_rate=0.001, learning_rate=0.01, learning_rate_decay=0.0):
    model = Sequential()
    for i, layer_dim in enumerate(hidden_layers_dim):
        if i == 0:
            model.add(Dense(layer_dim, input_shape=(input_dim, ), activation='relu',
                                kernel_regularizer=l2(lambda_rate)))
        else:
            model.add(Dense(layer_dim, activation='relu', kernel_regularizer=l2(lambda_rate)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=learning_rate, decay=learning_rate_decay),
                    loss='sparse_categorical_crossentropy')
    model.summary()
    return model
