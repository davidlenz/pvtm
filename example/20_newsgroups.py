from pvtm.pvtm import PVTM
from pvtm.pvtm import clean
from sklearn.datasets import fetch_20newsgroups
import nltk
import argparse
import numpy as np

newsgroups_train = fetch_20newsgroups(subset='train')
input_texts = newsgroups_train.data
print('There are', len(newsgroups_train.data), 'documents.')

nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))

# Create PVTM class object
pvtm = PVTM(input_texts)
pvtm.preprocess(lemmatize = False, lang = 'en', min_df = 0.005)
pvtm.fit(vector_size= 50,
         hs=0,
         dbow_words=1, # train word vectors!
         dm=0, # Distributed bag of words (=word2vec-Skip-Gram) (dm=0) OR distributed memory (=word2vec-cbow) (dm=1)
         epochs=10, # doc2vec training epochs
         window=1, # doc2vec window size
         seed=123, # doc3vec random seed
         random_state = 123, # gmm random seed
         min_count=5, # minimal number of appearences for a word to be considered
         workers=1, # doc2vec num workers
         alpha=0.025, # doc2vec initial learning rate
         min_alpha=0.025, # doc2vec final learning rate. Learning rate will linearly drop to min_alpha as training progresses.
         n_components=15, # number of Gaussian mixture components, i.e. Topics
         covariance_type='diag', # gmm covariance type
         verbose=1, # verbosity
         n_init=1, # gmm number of initializations

         )
print("Finished fitting")
# You can get document vectors from the Doc2Vec model by calling the command
document_vectors = np.array(pvtm.model.docvecs.vectors_docs)

# You can get distribution of the document over the defined topics by calling the command
# Thereby, each row is a single document, each column is one topic. The entries within the matrix are probabilities.
document_topics = np.array(pvtm.gmm.predict_proba(np.array(pvtm.model.docvecs.vectors_docs)))

# Inference
newsgroups_test = fetch_20newsgroups(subset='test')
new_text = newsgroups_test.data[0]
pvtm.infer_topics(new_text)

pvtm.start_webapp()

