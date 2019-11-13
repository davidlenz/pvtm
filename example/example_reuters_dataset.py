from pvtm.pvtm import PVTM
from pvtm.pvtm import clean
from keras.datasets import reuters
import nltk
import argparse
import numpy as np

# load reuters text data
def reuters_news_wire_texts():
    (x_train, y_train), (x_test, y_test) = reuters.load_data()
    wordDict = {y:x for x,y in reuters.get_word_index().items()}
    texts = []
    for x in x_train:
        texts.append(" ".join([wordDict.get(index-3) for index in x if wordDict.get(index-3) is not None]))
    return texts, y_train

input_texts, y = reuters_news_wire_texts()
len_docs = 5000
# load stop words
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))

# Create PVTM class object
pvtm = PVTM(input_texts[:len_docs])
pvtm.preprocess(lemmatize = False, lang = 'en', min_df = 0.005)
pvtm.fit(vector_size = 50,
         hs=0,
         dbow_words=1, # train word vectors!
         dm=0, # Distributed bag of words (=word2vec-Skip-Gram) (dm=0) OR distributed memory (=word2vec-cbow) (dm=1)
         epochs= 20, # doc2vec training epochs
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

# You can get document vectors from the Doc2Vec model by calling the command
document_vectors = np.array(pvtm.model.docvecs.vectors_docs)

# You can get distribution of the document over the defined topics by calling the command
# Thereby, each row is a single document, each column is one topic. The entries within the matrix are probabilities.
document_topics = np.array(pvtm.gmm.predict_proba(np.array(pvtm.model.docvecs.vectors_docs)))

# Inference
new_text = input_texts[len_docs+1]
pvtm.infer_topics(new_text)

pvtm.start_webapp()
