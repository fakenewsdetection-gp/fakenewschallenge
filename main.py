import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import os
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
max_num_words = 5000
num_classes = 4
hidden_layers_dim = [100]
dropout_rate = 0.4
learning_rate = 0.005
batch_size = 500
epochs = 5

# Process data sets
train_data = load_features('tfidf')
feature_size = len(train_data['features'][0])
test_data = load_features('tfidf', header='test')

# Check if models directory doesn't exist
if not os.path.isdir(models_dir):
    os.makedirs(models_dir)

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
    train_features = np.array(train_data['features'].to_list())
    train_labels = np.array(train_data['labels'].to_list())
    print(f"\n\nShape of training set (Inputs): {train_features.shape}")
    print(f"Shape of training set (Labels): {train_labels.shape}\n\n")
    mlp_history = mlp_model.fit(train_features, train_labels,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0.2,
                                callbacks=[checkpoint])
    plot_history(mlp_history)

# Load model
if mode == 'load':
    mlp_model = load_model(os.path.join(models_dir, mlp_model_file))

test_features = np.array(test_data['features'].to_list())
test_labels = np.array(test_data['labels'].to_list())
print(f"\n\nShape of test set (Inputs): {test_features.shape}")
print(f"Shape of test set (Labels): {len(test_labels)}\n\n")

# Prediction
test_predictions = mlp_model.predict_classes(test_features)
test_predictions = [label_ref_rev[i] for i in test_predictions]

print("Scores on test set")
report_score(test_labels, test_predictions)

# Save predictions
save_predictions(test_predictions, file_predictions)
