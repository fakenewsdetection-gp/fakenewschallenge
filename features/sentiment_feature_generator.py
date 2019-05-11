import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

nltk.download('vader-lexicon')

def process(data):
    # Initialize
    heads_sentiment = {}
    bodies_sentiment = {}
    res = []

    sentiment_analyzer = SentimentIntensityAnalyzer()
    def _computer_sentiment(sentences):
        return np.array([sentiment_analyzer.polarity_scores(s) for s in sentences]).mean()

    for instance in data.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in heads_sentiment:
            sentences = sent_tokenize(head)
            heads_sentiment[head] = _computer_sentiment(sentences)
        if body_id not in bodies_sentiment:
            sentences = sent_tokenize(data.bodies[body_id])
            heads_sentiment[body_id] = _computer_sentiment(sentences)
        res.append(np.concatenate((heads_sentiment[head], bodies_sentiment[body_id]), axis=None))

    return np.array(res)
