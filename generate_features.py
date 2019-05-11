from dataset import Dataset
import pandas as pd
import gc
import features.tf_idf_feature_generator as tfidf
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

print("Generating tf-idf features for the training set\n")

# Process training dataset
train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
    tfidf.process_train(raw_train, raw_test, max_num_words=max_num_words)

print("Saving tf-idf features for the training set\n")

train_data = pd.DataFrame({'features': train_set, 'labels': train_stances})

del train_set
del train_stances
gc.collect()

save_features(train_data, 'tfidf')

print("tf-idf features for the training set saved\n")

del train_data
gc.collect()

print("Generating tf-idf features for the test set\n")

# Process test dataset
test_set = tfidf.process_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

print("Saving tf-idf features for the test set\n")

test_stances = [label_ref[instance['Stance']] for instance in raw_test.instances]
test_data = pd.DataFrame({'features': test_set, 'labels': test_stances})

del test_set
del test_stances
gc.collect()

save_features(test_data, 'tfidf', header='test')

print("tf-idf features for the test set saved\n")
