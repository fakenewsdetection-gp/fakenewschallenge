import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import os
import gc
import random
import numpy as np
from util import *
from dataset import Dataset
from score import report_score
from model import build_mlp


random.seed(42)

# Prompt for mode
mode = input('mode (load / train)? ')

# Set file names
file_predictions = "predictions_test.csv"
models_dir = "models"
mlp_model_file = "mlp.hdf5"

# Initialise hyperparameters
num_classes = 4
hidden_layers_dim = [100]
dropout_rate = 0.5
learning_rate = 0.01
batch_size = 500
epochs = 90

# Check if models directory doesn't exist
if not os.path.isdir(models_dir):
    os.makedirs(models_dir)

# Train model
if mode == 'train':
    train_features = np.concatenate((np.load('train.tfidf.npy'), np.load('train.sent.npy')), axis=1)
    train_labels = np.load('train.labels.npy')
    feature_size = train_features.shape[1]
    mlp_model = build_mlp(feature_size, num_classes,
                            hidden_layers_dim,
                            dropout_rate=dropout_rate,
                            learning_rate=learning_rate)
    checkpoint = ModelCheckpoint(os.path.join(models_dir, mlp_model_file),
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    mode='min')
    print(f"\n\nShape of training set (Inputs): {train_features.shape}")
    print(f"Shape of training set (Labels): {train_labels.shape}\n\n")
    mlp_history = mlp_model.fit(train_features, train_labels,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0.2,
                                callbacks=[checkpoint])
    plot_history(mlp_history)

    del train_features
    del train_labels
    gc.collect()


# Load model
if mode == 'load':
    mlp_model = load_model(os.path.join(models_dir, mlp_model_file))

test_features = np.concatenate((np.load('test.tfidf.npy'), np.load('test.sent.npy')), axis=1)
test_labels = np.load('test.labels.npy')

print(f"\n\nShape of test set (Inputs): {test_features.shape}")
print(f"Shape of test set (Labels): {test_labels.shape}\n\n")

# Prediction
test_predictions = mlp_model.predict_classes(test_features)

test_predictions = [label_ref_rev[i] for i in test_predictions]
test_labels = [label_ref_rev[i] for i in test_labels]

print("Scores on test set")
report_score(test_labels, test_predictions)

# Save predictions
save_predictions(test_predictions, file_predictions)
