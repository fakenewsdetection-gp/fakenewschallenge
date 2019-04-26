from features.FeatureGenerator import *
import pandas as pd
import numpy as np
import pickle
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfFeatureGenerator(FeatureGenerator):

    def __init__(self, name='tfidf'):
        super(TfidfFeatureGenerator, self).__init__(name)


    def process(self, df):
        def _cat_headline_body(x):
            res = '%s %s' % (' '.join(x['Headline_unigram']), ' '.join(x['articleBody_unigram']))
            return res

        n_train = df[~df['target'].isnull()].shape[0]
        n_features = 200

        vectorizer_headlines = TfidfVectorizer(ngram_range=(1, 3), max_features=n_features)
        tfidf_headlines = vectorizer_headlines.fit_transform(
            df['Headline_unigram'].map(lambda x: ' '.join(x))).toarray()
        tfidf_headlines_train = tfidf_headlines[:n_train, :]
        tfidf_headlines_test  = tfidf_headlines[n_train:, :]

        vectorizer_bodies = TfidfVectorizer(ngram_range=(1,3), max_features=n_features)
        tfidf_bodies = vectorizer_bodies.fit_transform(
            df['articleBody_unigram'].map(lambda x: ' '.join(x))).toarray()
        tfidf_bodies_train = tfidf_bodies[:n_train, :]
        tfidf_bodies_test  = tfidf_bodies[n_train:, :]

        tfidf_cos_sim = np.zeros((df.shape[0], 1))
        for i in range(df.shape[0]):
            tfidf_cos_sim[i] = cosine_similarity(
                tfidf_headlines[i].reshape(1, -1), tfidf_bodies[i].reshape(1, -1))
        tfidf_cos_sim_train = tfidf_cos_sim[:n_train]
        tfidf_cos_sim_test  = tfidf_cos_sim[n_train:]

        self._dump(tfidf_headlines_train, 'train.headline.tfidf.pkl')
        self._dump(tfidf_headlines_test, 'test.headline.tfidf.pkl')
        self._dump(tfidf_bodies_train, 'train.body.tfidf.pkl')
        self._dump(tfidf_bodies_test, 'test.body.tfidf.pkl')
        self._dump(tfidf_cos_sim_train, 'train.sim.tfidf.pkl')
        self._dump(tfidf_cos_sim_test, 'test.sim.tfidf.pkl')


    def read(self, header='train'):
        files = ['.headline.tfidf.pkl', '.body.tfidf.pkl', '.sim.tfidf.pkl']
        files = [''.join([header, f]) for f in files]
        res = []

        for f in files:
            with open(f, 'rb') as infile:
                res.append(pickle.load(infile))
        return res
