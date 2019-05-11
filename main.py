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
from features.tf_idf_feature_generator import TfidfFeatureGenerator


random.seed(42)

# Prompt for mode
mode = input('mode (load / train)? ')

# Set file names
file_train_instances = "fnc-1/train_stances.csv"
file_train_bodies = "fnc-1/train_bodies.csv"
file_test_instances = "fnc-1/competition_test_stances.csv"
file_test_bodies = "fnc-1/competition_test_bodies.csv"
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

# Load data sets
raw_train = Dataset(file_train_instances, file_train_bodies)
raw_test = Dataset(file_test_instances, file_test_bodies)
n_train = len(raw_train.instances)

# Process data sets
tfidfFeatureGenerator = TfidfFeatureGenerator()
train_data, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
    tfidfFeatureGenerator.process_train(raw_train, raw_test, max_num_words=max_num_words)
feature_size = len(train_data['features'][0])
test_data = tfidfFeatureGenerator.process_test(raw_test, bow_vectorizer,
                                                tfreq_vectorizer, tfidf_vectorizer)

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
    train_labels = np.array(train_data['stance'].to_list())
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

test_stances = [instance['Stance'] for instance in raw_test.instances]
test_features = np.array(test_data['features'].to_list())
print(f"\n\nShape of test set (Inputs): {test_features.shape}")
print(f"Shape of test set (Labels): {len(test_stances)}\n\n")

# Prediction
test_predictions = mlp_model.predict_classes(test_features)
test_predictions = [label_ref_rev[i] for i in test_predictions]

print("Scores on test set")
report_score(test_stances, test_predictions)

# Save predictions
save_predictions(test_predictions, file_predictions)
