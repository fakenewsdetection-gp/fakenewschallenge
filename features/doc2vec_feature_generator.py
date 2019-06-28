import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from util import *


nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))


def process_train(train, test, max_num_words=5000):
    # Initialize
    heads = []
    heads_track = {}
    bodies = []
    bodies_track = {}
    body_ids = []
    id_ref = {}
    train_set = []
    train_stances = []
    cos_track = {}
    test_heads = []
    test_heads_track = {}
    test_bodies = []
    test_bodies_track = {}
    test_body_ids = []
    head_docvec_track = {}
    body_docvec_track = {}

    # Identify unique heads and bodies
    for instance in train.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in heads_track:
            heads.append(head)
            heads_track[head] = 1
        if body_id not in bodies_track:
            bodies.append(train.bodies[body_id])
            bodies_track[body_id] = 1
            body_ids.append(body_id)

    for instance in test.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in test_heads_track:
            test_heads.append(head)
            test_heads_track[head] = 1
        if body_id not in test_bodies_track:
            test_bodies.append(test.bodies[body_id])
            test_bodies_track[body_id] = 1
            test_body_ids.append(body_id)

    # Create reference dictionary
    for i, elem in enumerate(heads + body_ids):
        id_ref[elem] = i

    tagged_data = [TaggedDocument(words=nltk.tokenize.word_tokenize(_d.lower()),
        tags=[str(i)]) for i, _d in enumerate(heads + bodies)]

    max_epochs = 100
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)

    model.build_vocab(tagged_data)

    for epoch in range(1):
        model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    for instance in train.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        try:
            if head not in head_docvec_track:
                head_docvec = model.docvecs(id_ref[head])
                head_docvec_track[head] = head_docvec
        except:
            print('Error: ', id_ref[head])
        else:
            head_docvec = head_docvec_track[head]

        if body_id not in body_docvec_track:
            body_docvec = model.docvecs(id_ref[body_id])
            body_docvec_track[body_id] = body_docvec
        else:
            body_docvec = body_docvec_track[body_id]

        if (head, body_id) not in cos_track:
            docvec_cos = cosine_similarity(head_docvec, body_docvec)[0].reshape(1, 1)
            cos_track[(head, body_id)] = docvec_cos
        else:
            cos_track[(head, body_id)] = docvec_cos
        feat_vec = np.squeeze(np.c[head_docvec, body_docvec, docvec_cos])
        train_set.append(feat_vec)
        train_stances.append(label_ref[instance['Stance']])
    return np.array(train_set), np.array(train_stances), model


def process_test(test, model, steps=1000):
    """
    Process test set.

        Args:
            test: Dataset object, test set.

        Returns:
            test_data: pandas dataframe, contains the generated features only.
    """

    # Initialise
    test_set = []
    heads_track = {}
    bodies_track = {}
    cos_track = {}

    # Process test set
    for instance in test.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in heads_track:
            head_docvec = model.infer_vector(nltk.tokenize.word_tokenize(head.lower()), steps=steps).reshape(1, -1)
            heads_track[head] = head_docvec
        else:
            head_docvec = heads_track[head]
        if body_id not in bodies_track:
            body_docvec = model.infer_vector(nltk.tokenize.word_tokenize(body.lower()), steps=steps).reshape(1, -1)
            bodies_track[body_id] = body_docvec
        else:
            body_docvec = bodies_track[body_id]
        if (head, body_id) not in cos_track:
            docvec_cos = cosine_similarity(head_docvec, body_docvec)[0].reshape(1, 1)
            cos_track[(head, body_id)] = docvec_cos
        else:
            docvec_cos = cos_track[(head, body_id)]
        feat_vec = np.squeeze(np.c_[head_docvec, body_docvec, docvec_cos])
        test_set.append(feat_vec)
    return test_set
