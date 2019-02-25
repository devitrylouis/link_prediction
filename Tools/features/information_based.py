import numpy as np
import pandas as pd
import nltk
from sklearn.metrics.pairwise import linear_kernel

def given_feature_engineering(data, node_info, IDs, features_TFIDF):
    """
    data:
    """
    stpwds = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()

    # Number of overlapping words in title?
    overlap_title = []
    # Temporal distance between the papers
    temp_diff = []
    # Number of common authors
    comm_auth = []

    cosimi = []

    counter = 0
    for i in xrange(len(data)):

        source = data[i][0]
        target = data[i][1]

        index_source = IDs.index(source)
        index_target = IDs.index(target)

        source_info = [element for element in node_info if element[0]==source][0]
        target_info = [element for element in node_info if element[0]==target][0]

        # convert to lowercase and tokenize
        source_title = source_info[2].lower().split(" ")
        # remove stopwords
        source_title = [token for token in source_title if token not in stpwds]
        source_title = [stemmer.stem(token) for token in source_title]

        cosimi.append(linear_kernel(features_TFIDF[index_source:index_source+1],
                                features_TFIDF[index_target:index_target+1])[0][0])

        target_title = target_info[2].lower().split(" ")
        target_title = [token for token in target_title if token not in stpwds]
        target_title = [stemmer.stem(token) for token in target_title]

        source_auth = source_info[3].split(",")
        target_auth = target_info[3].split(",")

        overlap_title.append(len(set(source_title).intersection(set(target_title))))
        temp_diff.append(int(source_info[1]) - int(target_info[1]))
        comm_auth.append(len(set(source_auth).intersection(set(target_auth))))

    # Convert list of lists into array
    # Documents as rows, unique words as columns (i.e., example as rows, features as columns)
    return(pd.DataFrame(np.vstack((overlap_title, temp_diff, comm_auth, cosimi)).T, columns = ['overlap_title', 'temp_diff', 'comm_auth', 'cosimi']))
