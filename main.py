import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import os
import gc
import pickle
import numpy as np
from util import *
from dataset import Dataset
from score import report_score
from model import build_mlp
from sklearn.metrics import classification_report

np.random.seed(23)
tf.set_random_seed(42)

# Prompt for mode
mode = input('Mode (load / train)? ')

# Prompt whether to use sentiment features or not
sentiment = input('Use sentiment features (yes / no)? ')

# Prompt for path of hyperparameters dictionary file
hyperparameters_filepath = input('Hyperparameters filepath: ')

# Loading hyperparameters dictionary
with open(hyperparameters_filepath, 'rb') as hyperparametes_file:
    hyperparameters = pickle.load(hyperparametes_file)

# Set file names
file_predictions = "predictions_test.csv"
models_dir = "models"
mlp_model_file = "ucl-based.hdf5"

# Initialise hyperparameters
num_classes = 4
hidden_layers_dim = hyperparameters["hidden_layers_dim"]
lambda_rate = hyperparameters["lambda_rate"]
dropout_rate = hyperparameters["dropout_rate"]
clip_max = hyperparameters["clip_max"]
learning_rate = hyperparameters["learning_rate"]
batch_size = hyperparameters["batch_size"]
epochs = hyperparameters["epochs"]

# Check if models directory doesn't exist
if not os.path.isdir(models_dir):
    os.makedirs(models_dir)

# Train model
if mode == 'train':
    if sentiment == 'yes':
        train_features = np.concatenate((np.load('train.tfidf.npy'), np.load('train.sent.npy')), axis=1)
    else:
        train_features = np.load('train.tfidf.npy')
    train_labels = np.load('train.labels.npy')

    train_stances = []
    for label in train_labels:
        hot_encoded = [0 for _ in range(num_classes)]
        hot_encoded[label] = 1
        train_stances.append(hot_encoded)
    train_stances = np.array(train_stances)
    feature_size = train_features.shape[1]
    mlp_model = build_mlp(feature_size, num_classes,
                            hidden_layers_dim,
                            lambda_rate=lambda_rate,
                            learning_rate=learning_rate,
                            dropout_rate=dropout_rate,
                            clip_max=clip_max)
    checkpoint = ModelCheckpoint(os.path.join(models_dir, mlp_model_file),
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    mode='min')
    print(f"\n\nShape of training set (Inputs): {train_features.shape}")
    print(f"Shape of training set (Labels): {train_stances.shape}\n\n")
    mlp_history = mlp_model.fit(train_features, train_stances,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0.2,
                                callbacks=[checkpoint])
    plot_history(mlp_history)

    del train_features
    del train_labels
    gc.collect()

mlp_model = load_model(os.path.join(models_dir, mlp_model_file))

if sentiment == 'yes':
    test_features = np.concatenate((np.load('test.tfidf.npy'), np.load('test.sent.npy')), axis=1)
else:
    test_features = np.load('test.tfidf.npy')
test_labels = np.load('test.labels.npy')

print(f"\n\nShape of test set (Inputs): {test_features.shape}")
print(f"Shape of test set (Labels): {test_labels.shape}\n\n")

# Prediction
test_predictions = mlp_model.predict(test_features)
test_predictions = np.argmax(test_predictions, axis=1)
test_predictions = [label_ref_rev[i] for i in test_predictions]
test_labels = [label_ref_rev[i] for i in test_labels]

print("Scores on test set")
report_score(test_labels, test_predictions)

print()
print()

print(classification_report(test_labels, test_predictions))

# Save predictions
save_predictions(test_predictions, file_predictions)
