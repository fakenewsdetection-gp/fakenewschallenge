from features.FeatureGenerator import *
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfFeatureGenerator(FeatureGenerator):
    
    def __init__(self, name='tfidf'):
        super(TfidfFeatureGenerator, self).__init__(name)


    def process(self, df):
        def _cat_headline_body(x):
            res = '%s %s' % (' '.join(x['Headline_unigram']), ' '.join(x['articleBody_unigram']))
            return res

        df['all_text'] = list(df.apply(_cat_headline_body, axis=1))

        n_train = df[~df['target'].isnull()].shape[0]

        vectorizer_all = TfidfVectorizer(ngram_range=(1, 3))
        vectorizer_all.fit(df['all_text'])
        vocab = vectorizer_all.vocabulary_

        vectorizer_headlines = TfidfVectorizer(ngram_range=(1, 3), vocabulary=vocab)
        tfidf_headlines = vectorizer_headlines.fit_transform(
            df['Headline_unigram'].map(lambda x: ' '.join(x)))
        tfidf_headlines_train = tfidf_headlines[:n_train, :]
        tfidf_headlines_test  = tfidf_headlines[n_train:, :]

        vectorizer_bodies = TfidfVectorizer(ngram_range=(1,3), vocabulary=vocab)
        tfidf_bodies = vectorizer_bodies.fit_transform(
            df['articleBody_unigram'].map(lambda x: ' '.join(x)))
        tfidf_bodies_train = tfidf_bodies[:n_train, :]
        tfidf_bodies_test  = tfidf_bodies[n_train:, :]

        # tfidf_cos_sim = np.array(cosine_similarity(
        #     tfidf_headlines.reshape(:, -1), tfidf_bodies.reshape(:, -1)))
        # print(tfidf_cos_sim.shape)
        # tfidf_cos_sim_train = tfidf_cos_sim[:n_train]
        # tfidf_cos_sim_test  = tfidf_cos_sim[n_train:]

        self._dump(tfidf_headlines_train, 'train.headline.tfidf.pkl')
        self._dump(tfidf_headlines_test, 'test.headline.tfidf.pkl')
        self._dump(tfidf_bodies_train, 'train.body.tfidf.pkl')
        self._dump(tfidf_bodies_test, 'test.body.tfidf.pkl')
        # self._dump(tfidf_cos_sim_train, 'train.sim.tfidf.pkl')
        # self._dump(tfidf_cos_sim_test, 'test.sim.tfidf.pkl')


    def read(self, header='train'):
        # files = ['.headline.tfidf.pkl', '.body.tfidf.pkl', '.sim.tfidf.pkl']
        files = ['.headline.tfidf.pkl', '.body.tfidf.pkl', '.sim.tfidf.pkl']
        files = [str(header + f) for f in files]
        res = []

        for f in files:
            with open(f, 'rb') as infile:
                res.append(pickle.load(infile))
        return res


    def _dump(self, df, filename):
        print(filename, "--- Shape:", df.shape)
        with open(filename, 'wb') as outfile:
            pickle.dump(df, outfile, -1)
