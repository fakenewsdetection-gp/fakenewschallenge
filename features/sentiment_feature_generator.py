from features.FeatureGenerator import *
import pandas as pd
import numpy as np
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize


class SentimentFeatureGenerator(FeatureGenerator):

    def __init__(self, name='sent'):
        super(SentimentFeatureGenerator, self).__init__(name)

    def process(self, df_original):
        df = df_original.copy()

        n_train = df[~df['target'].isnull()].shape[0]

        sid = SentimentIntensityAnalyzer()
        def _compute_sentiment(sentences):
            return pd.DataFrame([sid.polarity_scores(s) for s in sentences]).mean()

        df['headline_sentiment'] = df['Headline'].apply(lambda x: sent_tokenize(x))
        df = pd.concat([df, df['headline_sentiment'].apply(lambda x: _compute_sentiment(x))], axis=1)
        df.rename(columns={'compound':'h_compound', 'neg':'h_neg', 'neu':'h_neu', 'pos':'h_pos'}, inplace=True)
        sent_headlines = df[['h_compound','h_neg','h_neu','h_pos']].values

        sent_headlines_train = sent_headlines[:n_train, :]
        sent_headlines_test  = sent_headlines[n_train:, :]

        df['body_sentiment'] = df['articleBody'].map(lambda x: sent_tokenize(x))
        df = pd.concat([df, df['body_sentiment'].apply(lambda x: _compute_sentiment(x))], axis=1)
        df.rename(columns={'compound':'b_compound', 'neg':'b_neg', 'neu':'b_neu', 'pos':'b_pos'}, inplace=True)
        sent_bodies = df[['b_compound','b_neg','b_neu','b_pos']].values

        sent_bodies_train = sent_bodies[:n_train, :]
        sent_bodies_test  = sent_bodies[n_train:, :]

        self._dump(sent_headlines_train, 'train.headline.sent.pkl')
        self._dump(sent_headlines_test, 'test.headline.sent.pkl')
        self._dump(sent_bodies_train, 'train.body.sent.pkl')
        self._dump(sent_bodies_test, 'test.body.sent.pkl')

    
    def read(self, header='train'):
        files = ['.headline.sent.pkl', '.body.sent.pkl']
        files = [''.join([header, f]) for f in files]
        res = []

        for f in files:
            with open(f, 'rb') as infile:
                res.append(pickle.load(infile))
        return res