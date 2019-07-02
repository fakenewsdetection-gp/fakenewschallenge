import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.constraints import max_norm

print(tf.VERSION)
print(tf.keras.__version__)


def build_mlp(input_dim, num_classes, hidden_layers_dim,
                lambda_rate=0.001, learning_rate=0.01, dropout_rate=0.4, clip_max=5.0):
    model = Sequential()
    for i, layer_dim in enumerate(hidden_layers_dim):
        if i == 0:
            model.add(Dense(layer_dim, input_shape=(input_dim, ), activation='relu',
                                kernel_regularizer=l2(lambda_rate), kernel_constraint=max_norm(clip_max)))
        else:
            model.add(Dense(layer_dim, activation='relu',
                kernel_regularizer=l2(lambda_rate), kernel_constraint=max_norm(clip_max)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax', kernel_constraint=max_norm(clip_max)))
    model.compile(Adam(lr=learning_rate), loss='categorical_crossentropy')
    model.summary()
    return model
