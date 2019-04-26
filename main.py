import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import random
import os
from util import *
from dataset import Dataset
from score import report_score
from model import build_mlp


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

# Check if models directory doesn't exist
if not os.path.isdir(models_dir):
    os.makedirs(models_dir)

# Initialise hyperparameters
r = random.Random()
lim_unigram = 5000
num_classes = 4
hidden_layers_dim = [100]
dropout_rate = 0.4
learning_rate = 0.01
batch_size = 500
epochs = 90

# Load data sets
raw_train = Dataset(file_train_instances, file_train_bodies)
raw_test = Dataset(file_test_instances, file_test_bodies)
n_train = len(raw_train.instances)

# Process data sets
train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
    pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
feature_size = len(train_set[0])
test_set, test_stances = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

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
    print(f"Shape of training set (Labels): {train_stances.shape}")
    mlp_history = mlp_model.fit(train_set, train_stances,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_split=0.2,
                                    callbacks=[checkpoint])
    plot_history(mlp_history)

# Load model
if mode == 'load':
    mlp_model = load_model(os.path.join(models_dir, mlp_model_file))

print(f"\n\nShape of test set (Inputs): {test_set.shape}")
print(f"Shape of test set (Labels): {test_stances.shape}")

# Prediction
test_predictions = mlp_model.predict_classes(test_set)

predicted = [label_ref_rev[i] for i in test_predictions]
actual = [label_ref_rev[i] for i in test_stances]

print("\n\nScores on test set")
report_score(actual, predicted)

# Save predictions
save_predictions(test_predictions, file_predictions)
