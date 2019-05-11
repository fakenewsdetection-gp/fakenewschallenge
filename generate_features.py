import numpy as np
import gc
from dataset import Dataset
import features.tf_idf_feature_generator as tfidf
import features.sentiment_feature_generator as sent
from util import *


# Set file names
file_train_instances = "fnc-1/train_stances.csv"
file_train_bodies = "fnc-1/train_bodies.csv"
file_test_instances = "fnc-1/competition_test_stances.csv"
file_test_bodies = "fnc-1/competition_test_bodies.csv"

# Load data sets
raw_train = Dataset(file_train_instances, file_train_bodies)
raw_test = Dataset(file_test_instances, file_test_bodies)

max_num_words = 5000

print("\nGenerating tf-idf features for the training set\n")

# Generate tf-idf features for the training dataset
train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
    tfidf.process_train(raw_train, raw_test, max_num_words=max_num_words)

print("\nSaving tf-idf features and labels of the training set\n")

np.save('train.tfidf', train_set)
np.save('train.labels', train_stances)

del train_set
del train_stances
gc.collect()

print("\nGenerating sentiment features for the training set\n")

# Generate sentiment features for the training dataset
train_set = sent.process(raw_train)

print("\nSaving sentiment features of the training set\n")

np.save('train.sent', train_set)

del train_set
gc.collect()

print("\nGenerating tf-idf features for the test set\n")

# Generate tf-idf features for the test dataset
test_set = tfidf.process_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
test_stances = [label_ref[instance['Stance']] for instance in raw_test.instances]

print("\nSaving tf-idf features and labels of the test set\n")

np.save('test.tfidf', test_set)
np.save('test.labels', test_stances)

del test_set
del test_stances
gc.collect()

print("\nSaving sentiment features of the training set\n")

# Generate sentiment features for the test dataset
test_set = sent.process(raw_test)

print("\nSaving sentiment features of the test set\n")

np.save('test.sent', test_set)
