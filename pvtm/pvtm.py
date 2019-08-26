import numpy as np

np.random.seed(23)
import random

random.seed(1)

import pandas as pd
import time
from sklearn import mixture
import gensim
# from cleantext import clean
import re

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import inspect

import spacy


clean = lambda x: re.sub("[^abcdefghijklmnopqrstuvwxyzäöüß& ']", '', str(x).lower()).strip()


class Documents(object):
    """
    """

    def __init__(self, documents):
        self.documents = documents
        self.len = len(documents)

    def __iter__(self):
        for i, doc in enumerate(self.documents):
            yield gensim.models.doc2vec.TaggedDocument(words=doc, tags=[i])


class PVTM(Documents):

    def __init__(self, texts, preprocess=True, **kwargs):
        if preprocess == True:
            texts = self.preprocess(texts, **kwargs)
        x = [kk.split() for kk in texts]
        self.documents = texts
        self.x_docs = Documents(x)

    def get_allowed_vocab(self, data, min_df=0.05, max_df=0.95):
        """
        Takes a list of strings.
        Result is a vocabulary based on the corpora from the input dataset.
        Pre-filtering is done using sklearns tfidfVectorizer with settings for min_df and max_df.
        """
        print(min_df, max_df)
        self.tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df)

        # fit on dataset
        self.tfidf.fit(data)
        # get vocabulary
        vocabulary = set(self.tfidf.vocabulary_.keys())
        print(len(vocabulary), 'words in the vocabulary')
        return vocabulary

    def popularity_based_prefiltering(self, data, min_df=0.05, max_df=0.95, stopwords=None):
        """popularity based pre-filtering. Ignore rare and common words. 
        Takes a list of strings as input. stopwords is a list of words if provided."""
        vocabulary = self.get_allowed_vocab(data, min_df=min_df, max_df=max_df)
        vocabulary = frozenset(vocabulary)
        if stopwords:
            stopwords = frozenset(stopwords)
        pp = []
        for i, line in enumerate(data):
            rare_removed = list(filter(lambda word: word.lower() in vocabulary, line.split()))
            if stopwords:
                stops_removed = list(filter(lambda word: word.lower() not in stopwords, rare_removed))
                pp.append(" ".join(stops_removed))
            else:
                pp.append(" ".join(rare_removed))
        return pp, vocabulary

    def preprocess(self, texts, lemmatized, **kwargs):
        '''
        The finction takes a list of texts and removes stopwords as well as very frequent and very unfrequent words.
        The function also uses clean() function from clean-text to lowercase, 
        to remove special characters, number, currency symbols etc. 
        '''
        # translator = str.maketrans('', '', string.punctuation)  # check was der genau macht
        # texts = [text.translate(translator) for text in texts]

        # texts = [clean(x, **{key: value for key, value in kwargs.items()
        #                      if key in inspect.getfullargspec(clean).args}) for x in texts]

        texts = [clean(x) for x in texts]

        if not lemmatized:
            print('Start lemmatization...')
            t0 = time.time()
            nlp = spacy.load('de')
            nlp.disable_pipes('tagger', 'ner')
            doclist = list(nlp.pipe(texts, n_threads=6, batch_size=500))
            texts = [' '.join([listitem.lemma_ for listitem in doc]) for i, doc in enumerate(doclist)]

            print("Save lemmatized texts to lemmatized.txt")
            with open("lemmatized.txt", "w", encoding="utf-8") as file:
                for line in texts:
                    file.write(line)
                    file.write("\n")

            t1 = time.time()
            print('Finished lemmatization. Process took', t1-t0, 'seconds')

        print('len(texts)', len(texts))
        cleaned_text, self.vocab = self.popularity_based_prefiltering(texts,
                                                                      **{key: value for key, value in kwargs.items()
                                                                         if key in inspect.getfullargspec(
                                                                              self.popularity_based_prefiltering).args})
        return cleaned_text

    def fit(self, epochs=10, **kwargs):
        """
        """
        # self.model = gensim.models.Doc2Vec(seed=1, **{key: value for key, value in kwargs.items()
        #                                               if key in inspect.getfullargspec(gensim.models.Doc2Vec).args})

        self.model = gensim.models.Doc2Vec(
        vector_size = 100,
        hs = 0,
        #negative=0,
        dbow_words = 1,  # train word vectors!
        dm = 0,  # Distributed bag of words (=word2vec-Skip-Gram) (dm=0) OR distributed memory (=word2vec-cbow) (dm=1)
        epochs = epochs,  # doc2vec training epochs
        window = 5,  # doc2vec window size
        # seed=123, # doc3vec random seed
        # random_state = 123, # gmm random seed
        min_count = 5,  # minimal number of appearences for a word to be considered
        workers = 1,  # doc2vec num workers
        alpha = 0.025,  # doc2vec initial learning rate
        min_alpha = 0.025,  # doc2vec final learning rate. Learning rate will linearly drop to min_alpha as training progresses.
        #n_components = n_topics,  # number of Gaussian mixture components, i.e. Topics
        covariance_type = 'diag',  # gmm covariance type
        verbose = 1,  # verbosity
        n_init = 1,)
        # generate doc2vec model vocab
        print('Building vocab')
        self.model.build_vocab(self.x_docs)
        print(list(self.model.wv.vocab.keys())[:10])

        self.model.random.seed(1)

        print(self.model.wv.vectors[:2,:10])

        doc_counts = len(self.documents)
        print('Start training..')
        self.model.train(self.x_docs, total_examples=doc_counts, epochs=epochs)
        print('Finished training.')

        print(self.model.wv.vectors[:2, :10])

        print('Start clustering..')
        self.gmm = mixture.GaussianMixture(random_state=1, **{key: value for key, value in kwargs.items()
                                                              if key in inspect.getfullargspec(
                mixture.GaussianMixture).args})

        print('Finished clustering.')

        self.doc_vectors = np.array(self.model.docvecs.vectors_docs)

        self.gmm.fit(self.doc_vectors)
        self.BIC = self.gmm.bic(self.doc_vectors)
        self.cluster_center = self.gmm.means_
        print('BIC: {}'.format(self.BIC))

        # self.cluster_center(np.array(self.model.docvecs.vectors_docs), self.gmm)
        self.document_vectors = np.array(self.model.docvecs.vectors_docs)
        self.document_topics = np.array(self.gmm.predict_proba(self.document_vectors))

        self.get_document_topics()

        self.most_similar_words_per_topic = pd.DataFrame(
            [self.most_similar_words_per_topic(topic, 200) for topic in range(self.gmm.n_components)])

    def cluster_center(self, vectors, gmm):
        """
        Approximates cluster centers for a given clustering from a GMM.
        First method only takes the topic with the highest probability per document into account.
        Averaging the document vectors per topic cluster provides the cluster center for the topic.
        Second method  approximates cluster centers for a given clustering 
        from a GMM with weighted single vectors from a certain topic.
        Returns two lists of the cluster centers.
        """
        self.clustercenter = []
        assignments = gmm.predict(vectors)
        n_components = np.unique(assignments).shape[0]
        for i in range(n_components):
            gmm_centerindexe = np.where(assignments == i, True, False)
            self.clustercenter.append(vectors[gmm_centerindexe].mean(0))

        self.clustercenter_probas = []
        print('vectors', vectors.shape)
        assignments_proba = gmm.predict_proba(vectors)
        n_components = assignments_proba.shape[1]
        for i in range(n_components):
            center = []
            for j in range(len(vectors)):
                center_j = vectors[j] * assignments_proba[j, i]
                center.append(center_j)
            self.clustercenter_probas.append(np.mean(center, axis=0))

    def get_string_vector(self, strings, steps=10):
        '''
        The function takes a string (document) and
        transforms it to vector using a trained model.
        '''
        return [self.model.infer_vector(string.split(), steps=steps) for string in strings]

    def get_topic_weights(self, vector):
        '''
        The function takes a document vector 
        and returns a distribution of a given document over all topics.
        '''

        return self.gmm.predict_proba(vector)

    def create_wordcloud_by_topic(self, topic, n_words=100):
        """
        The function takes a number of a topic and represents 
        most common word (default=100) in a wordcloud.
        """
        text = pd.DataFrame(pvtm.model.wv.similar_by_vector(pvtm.cluster_center[topic],
                                                            topn=100),
                            columns=['word', "similarity"]).word.values
        text = ', '.join(text)
        wordcloud = WordCloud(max_font_size=50, max_words=n_words, background_color="white").generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear", )
        ax.axis("off")
        plt.show()
        #plt.savefig(f"results/{modelname}/sim_{topic}.pdf", bbox_inches="tight")
        #plt.savefig(f"results/{modelname}/sim_{topic}.png", bbox_inches="tight")
#        plt.close()


    def similarity_wordcloud(self, topic, n_words=100):
        text = pd.DataFrame(self.model.wv.similar_by_vector(self.cluster_center[topic],
                                                            topn=100),
                            columns=['word', "similarity"]).word.values
        text = ', '.join(text)
        wordcloud = WordCloud(max_font_size=50, max_words=n_words, background_color="white").generate(text)
        fig, ax = plt.subplots( figsize=(7, 4))
        ax.imshow(wordcloud, interpolation="bilinear", )
        ax.axis("off")

        # topic_over_time_df.groupby("year")[topic].sum().plot(ax=ax2)
        #
        # plt.savefig(f"results/{modelname}/sim_{topic}.pdf", bbox_inches="tight")
        # plt.savefig(f"results/{modelname}/sim_{topic}.png", bbox_inches="tight")
        # plt.close()

    def get_document_topics(self):

        most_relevant_topic_per_document = pd.DataFrame(self.document_topics).idxmax(1)
        kk = pd.DataFrame(self.documents).join(pd.DataFrame(most_relevant_topic_per_document, columns=['top_topic']))
        # kk[0] = kk[0].apply(lambda x : " ".join([word for word in x.split() if word.lower() not in stop_words]))
        self.wordcloud_df = kk.groupby('top_topic')[0].apply(list).apply(" ".join).str.lower()
        self.topic_words = pd.DataFrame(
            [pd.Series(self.wordcloud_df.loc[topic].split()).value_counts().iloc[:100].index.values for topic in
             range(self.gmm.n_components)])

    def most_similar_words_per_topic(self, topic, n_words):
        sims = cosine_similarity([self.cluster_center[topic]], self.model.wv.vectors)
        sims = np.argsort(sims)[0][::-1]
        text = [self.model.wv.index2word[k] for k in sims[:n_words]]
        return text



