from dataset import Dataset
import pandas as pd
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

# Process data sets
train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
    tfidf.process_train(raw_train, raw_test, max_num_words=max_num_words)
feature_size = len(train_data['features'][0])
test_set = tfidf.process_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

test_stances = [label_ref[instance['Stance']] for instance in raw_test.instances]

train_data = pd.DataFrame({'features': train_set, 'labels': train_stances})
test_data = pd.DataFrame({'features': test_set, 'labels': test_stances})

save_features(train_data, 'tfidf')
save_features(test_data, 'tfidf', header='test')
