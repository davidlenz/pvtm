import numpy as np
from sklearn import mixture
import gensim
from cleantext import clean
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import inspect


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
        vec = TfidfVectorizer(min_df=min_df, max_df=max_df)

        # fit on dataset
        vec.fit(data)
        # get vocabulary
        vocabulary = set(vec.vocabulary_.keys())
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

    def preprocess(self, texts, **kwargs):
        '''
        The finction takes a list of texts and removes stopwords as well as very frequent and very unfrequent words.
        The function also uses clean() function from clean-text to lowercase, 
        to remove special characters, number, currency symbols etc. 
        '''
        translator = str.maketrans('', '', string.punctuation)  # check was der genau macht
        texts = [text.translate(translator) for text in texts]

        texts = [clean(x, **{key: value for key, value in kwargs.items()
                             if key in inspect.getfullargspec(clean).args}) for x in texts]

        cleaned_text, self.vocab = self.popularity_based_prefiltering(texts,
                                                                      **{key: value for key, value in kwargs.items()
                                                                         if key in inspect.getfullargspec(
                                                                              self.popularity_based_prefiltering).args})
        return cleaned_text

    def fit(self, n_range=10, alpha_value_steps=0.002, **kwargs):
        """
        """
        self.model = gensim.models.Doc2Vec(**{key: value for key, value in kwargs.items()
                                              if key in inspect.getfullargspec(gensim.models.Doc2Vec).args})
        # generate doc2vec model vocab
        print('Building vocab')
        self.model.build_vocab(self.x_docs)

        doc_counts = len(self.documents)
        for epoch in range(n_range):
            print("epoch " + str(epoch))
            self.model.train(self.x_docs, total_examples=doc_counts, epochs=1)
            #     model.save(MODEL_SAVE_NAME)
            self.model.alpha -= alpha_value_steps

        self.gmm = mixture.GaussianMixture(**{key: value for key, value in kwargs.items()
                                              if key in inspect.getfullargspec(mixture.GaussianMixture).args})

        print('GMM Clustering')

        self.doc_vectors = np.array(self.model.docvecs.vectors_docs)

        self.gmm.fit(self.doc_vectors)
        self.BIC = self.gmm.bic(self.doc_vectors)
        print('BIC: {}'.format(self.BIC))
        print('Cluster Center Computation')
        self.cluster_center(np.array(self.model.docvecs.vectors_docs), self.gmm)
        self.document_vectors = np.array(self.model.docvecs.vectors_docs)
        self.document_topics = np.array(self.gmm.predict_proba(self.document_vectors))

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

    def create_wordcloud(self, topic, n_words=100, use_probas=True):
        """
        The function takes a number of a topic and represents 
        most common word (default=100) in a wordcloud.
        """
        if use_probas == True:
            center_probas = self.clustercenter_probas
        else:
            center_probas = self.clustercenter
        sims = cosine_similarity([center_probas[topic]], self.model.wv.vectors)
        sims = np.argsort(sims)[0][::-1]
        text = [self.model.wv.index2word[k] for k in sims[:n_words]]
        text = ', '.join(text)
        wordcloud = WordCloud(max_font_size=50, max_words=n_words, background_color="white").generate(text)
        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
