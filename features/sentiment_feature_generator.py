import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize


nltk.download('vader_lexicon')
nltk.download('punkt')


def process(data):
    """
    Process dataset.

        Args:
            data: Dataset object.

        Returns:
            res: numpy array of numpy arrays which contain the
                 sentiment features of each headline and body pair.
    """
    # Initialize
    heads_sentiment = {}
    bodies_sentiment = {}
    res = []

    sentiment_analyzer = SentimentIntensityAnalyzer()
    def _computer_sentiment(sentences):
        return np.mean([list(sentiment_analyzer.polarity_scores(s).values()) for s in sentences], axis=0)

    for instance in data.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in heads_sentiment:
            sentences = sent_tokenize(head)
            heads_sentiment[head] = _computer_sentiment(sentences)
        if body_id not in bodies_sentiment:
            sentences = sent_tokenize(data.bodies[body_id])
            bodies_sentiment[body_id] = _computer_sentiment(sentences)
        res.append(np.concatenate((heads_sentiment[head], bodies_sentiment[body_id]), axis=None))
    return np.array(res)
