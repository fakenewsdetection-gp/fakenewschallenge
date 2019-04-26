import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import os
import random
import itertools
import numpy as np
from util import *
from score import report_score
from model import build_mlp
from features.TfidfFeatureGenerator import TfidfFeatureGenerator
from features.SentimentFeatureGenerator import SentimentFeatureGenerator


random.seed(42)

# Prompt for mode
mode = input('mode (load / train)? ')

# Set file names
file_train_instances = "fnc-1/train_stances.csv"
file_test_instances = "fnc-1/competition_test_stances.csv"
file_predictions = "predictions_test.csv"
models_dir = "models"
mlp_model_file = "mlp.hdf5"

# Check if models directory doesn't exist
if not os.path.isdir(models_dir):
    os.makedirs(models_dir)

# Initialise hyperparameters
num_classes = 4
hidden_layers_dim = [100]
dropout_rate = 0.4
learning_rate = 0.005
batch_size = 500
epochs = 120

# Loading training labels (stances)
train_instances = read(file_train_instances)
train_stances = []
for instance in train_instances:
    train_stances.append(instance['Stance'])
n_train = len(train_stances)
train_stances = np.array(train_stances)

# Loading testing labels (stances)
test_instances = read(file_test_instances)
test_stances = []
for instance in test_instances:
    test_stances.append(instance['Stance'])
n_test = len(test_stances)
test_stances = np.array(test_stances)

# Loading feature vectors
generators = [
    TfidfFeatureGenerator(),
    SentimentFeatureGenerator()
]

train_features_list = list(itertools.chain.from_iterable([g.read('train') for g in generators]))
train_feature_vector = np.empty((n_train, 0))
for f in train_features_list:
    train_feature_vector = np.concatenate((train_feature_vector, f), axis=1)

test_features_list = list(itertools.chain.from_iterable([g.read('test') for g in generators]))
test_feature_vector = np.empty((n_test, 0))
for f in test_features_list:
    test_feature_vector = np.concatenate((test_feature_vector, f), axis=1)

# Train model
if mode == 'train':
    mlp_model = build_mlp(feature_size, num_classes,
                            hidden_layers_dim,
                            dropout_rate=dropout_rate,
                            learning_rate=learning_rate)
    checkpoint = ModelCheckpoint(os.path.join(models_dir, mlp_model_file),
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    mode='min')
    print(f"\n\nShape of training set (Inputs): {train_set.shape}")
    print(f"Shape of training set (Labels): {train_stances.shape}\n\n")
    mlp_history = mlp_model.fit(train_feature_vector, train_stances,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_split=0.2,
                                    callbacks=[checkpoint])
    plot_history(mlp_history)

# Load model
if mode == 'load':
    mlp_model = load_model(os.path.join(models_dir, mlp_model_file))

print(f"\n\nShape of test set (Inputs): {test_set.shape}")
print(f"Shape of test set (Labels): {test_stances.shape}\n\n")

# Prediction
test_predictions = mlp_model.predict_classes(test_feature_vector)

predicted = [label_ref_rev[i] for i in test_predictions]
actual = [label_ref_rev[i] for i in test_stances]

print("Scores on test set")
report_score(actual, predicted)

# Save predictions
save_predictions(test_predictions, file_predictions)
