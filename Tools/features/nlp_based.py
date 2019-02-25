import numpy as np
import pandas as pd

import nltk
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

## SECOND ATTEMPT : ABSTRACT 2 VEC

def doc2vec_similarities(data, node_info, IDs, name = 'd2v_10_3'):
    """
    data:
    """
    # Load pre-trained w2v
    model = Doc2Vec.load(name + ".model")

    # Number of overlapping words in title?
    cosine_similarities = []
    #predictions_source = []
    #predictions_target = []

    counter = 0
    for i in range(0, len(data)):

        source_info = [element for element in node_info if element[0]==data[i][0]][0]
        target_info = [element for element in node_info if element[0]==data[i][1]][0]

        # Tokenize
        source_abstract = clean_doc(source_info[5].lower(), cut_words = 2)
        source_abstract = word_tokenize(source_abstract)

        target_abstract = clean_doc(target_info[5].lower(), cut_words = 2)
        target_abstract = word_tokenize(target_abstract)

        # Predictions
        predictions_source = model.infer_vector(source_abstract)
        predictions_target = model.infer_vector(target_abstract)

        cosine_similarities.append(cosine_similarity(predictions_source, predictions_target))

    # predictions_source = np.vstack(predictions_source).T
    # predictions_target = np.vstack(predictions_target).T

    #results = np.concatenate((np.vstack(predictions_source).T, np.vstack(predictions_target).T), axis = 1)

    # Convert list of lists into array
    # Documents as rows, unique words as columns (i.e., example as rows, features as columns)

    return(pd.DataFrame(np.array(cosine_similarities).T, columns = ['cosine_similarity']))

def train_w2v(node_info, name, train_on = 'title', cut_words = 2, train = True):

    if train == False:
        return('Model not trained')
    # Tag documents for
    if train_on == 'title':
        documents = [element[2] for element in node_info]
    else:
        documents = [element[5] for element in node_info]

    lower_documents = [_d.lower() for i, _d in enumerate(documents)]
    lower_documents = map(lambda x: clean_doc(x, cut_words), lower_documents)
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(lower_documents)]

    # Training D2V
    max_epochs = 10
    vec_size = 20
    alpha = 0.025

    model = Doc2Vec(vec_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1)

    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('Iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save(name + ".model")
    print("Model Saved")

## FIRST ATTEMPT

def compute_cosine(data, features, IDs):
    IDs_to_idx = dict(zip(IDs, [int(x) - 1001 for x in IDs]))
    cosine = []

    for sample in data:
        # Get indices
        idx1 = IDs_to_idx[sample[0]]
        idx2 = IDs_to_idx[sample[1]]
        cosine.append(cosine_similarity(features, idx1, idx2))

    cosine = np.array(cosine)
    cosine = cosine.T
    cosine = cosine.reshape(-1, 1)
    print(cosine.shape)

    return(pd.DataFrame(cosine, columns = ['cosine']))

## HELPERS

def clean_doc(document, cut_words):
    return(' '.join(word for word in document.split() if len(word) > cut_words))

def cosine_similarity(u, v):
    dot_product = np.dot(u, v)
    normalizer = np.linalg.norm(u) * np.linalg.norm(v)
    return(dot_product/float(normalizer))

def cosine_similarity_(features, ID1, ID2):
    # Compute cosine similarity
    try:
        cosine_similarities = float(linear_kernel(features[idx1], features[idx2]).flatten())
    except:
        cosine_similarities = 0
    return(cosine_similarities)


def tfidf(node_info):
    # TFIDF vector of each paper (3) paper title (string)
    corpus = [element[2] for element in node_info]
    vectorizer = TfidfVectorizer(stop_words="english")
    # Each row is a node in the order of node_info
    features = vectorizer.fit_transform(corpus)
    return(vectorizer, features)
