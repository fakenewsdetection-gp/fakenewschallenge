import pandas as pd
import pickle
from dataset import Dataset
from features.tf_idf_feature_generator import TfidfFeatureGenerator


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
tfidfFeatureGenerator = TfidfFeatureGenerator()
train_data, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
    tfidfFeatureGenerator.process_train(raw_train, raw_test, max_num_words=max_num_words)
feature_size = len(train_data['features'][0])
test_data = tfidfFeatureGenerator.process_test(raw_test, bow_vectorizer,
                                                tfreq_vectorizer, tfidf_vectorizer)
