import numpy as np

np.random.seed(23)
import random

random.seed(1)

import pandas as pd
import time
from sklearn import mixture
import gensim
import re
import joblib

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import inspect

import spacy

clean = lambda x: re.sub('\W+',' ', re.sub(" \d+", '',re.sub('[äöüß]','',str(x).lower())).strip())


class Documents(object):
    '''
    :return: tagged documents.
    '''
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

    def preprocess(self, texts, lemmatized, **kwargs):
        '''
        The function takes a list of texts and removes stopwords, special characters, punctuation as well
        as very frequent and very unfrequent words.
        :param texts: original documents.
        :param lemmatized: if lemmatized=False, lemmatization of the texts will be done.
        :param kwargs: additional key word arguments passed to popularity_based_prefiltering() function.
        :return: cleaned texts.
        '''

        texts = [clean(x) for x in texts]

        if not lemmatized:
            texts = self.lemmatize(texts)

        cleaned_text, self.vocab = self.popularity_based_prefiltering(texts,
                                                                      **{key: value for key, value in kwargs.items()
                                                                         if key in inspect.getfullargspec(
                                                                              self.popularity_based_prefiltering).args})
        return cleaned_text

    def get_allowed_vocab(self, data, min_df=0.05, max_df=0.95):
        '''
        Vocabulary building using sklearn's tfidfVectorizer.
        :param data: a list of strings(documents)
        :param min_df: words are ignored if the frequency is lower than min_df.
        :param max_df: words are ignored if the frequency is higher than man_df.
        :return: corpus specific vocabulary
        '''

        print(min_df, max_df)
        self.tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df)

        # fit on dataset
        self.tfidf.fit(data)
        # get vocabulary
        vocabulary = set(self.tfidf.vocabulary_.keys())
        print(len(vocabulary), 'words in the vocabulary')
        return vocabulary

    def popularity_based_prefiltering(self, data, min_df=0.05, max_df=0.95, stopwords=None):
        '''
        Prefiltering function which removes very rare/common words.
        :param data: a list of strings(documents)
        :param min_df: words are ignored if the frequency is lower than min_df.
        :param max_df: words are ignored if the frequency is higher than max_df.
        :param stopwords: a list of stopwords.
        :return: filtered documents' texts and corpus specific vocabulary.
        '''
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

    def lemmatize(self, texts, lang='en'):
        '''
        Lemmatization of input texts.
        :param texts: original documents.
        :param lang: language of the text documents.
        :return: lemmmatized texts.
        '''
        print('Start lemmatization...')
        t0 = time.time()
        nlp = spacy.load(lang + "_core_web_sm")
        nlp.disable_pipes('tagger', 'ner')
        doclist = list(nlp.pipe(texts, n_threads=6, batch_size=500))
        texts = [' '.join([listitem.lemma_ for listitem in doc]) for i, doc in enumerate(doclist)]
        print("Save lemmatized texts to lemmatized.txt")
        with open("lemmatized.txt", "w", encoding="utf-8") as file:
            for line in texts:
                file.write(line)
                file.write("\n")

            t1 = time.time()
            print('Finished lemmatization. Process took', t1 - t0, 'seconds')
        print('len(texts)', len(texts))
        return texts

    def fit(self, **kwargs):
        '''
        First, a Doc2Vec model is trained and clustering of the documents is done by means of GMM.
        :param kwargs: additional arguments which should be passed to Doc2Vec and GMM.
        :param save: if you want to save the trained model set save=True.
        :param filename: name of the saved model.
        :return: Doc2Vec model and GMM clusters.
        '''
        # generate doc2vec model
        self.model = gensim.models.Doc2Vec(self.x_docs, **{key: value for key, value in kwargs.items()
                                                           if
                                                           key in inspect.getfullargspec(gensim.models.Doc2Vec).args or
                                                           key in inspect.getfullargspec(
                                                               gensim.models.base_any2vec.BaseAny2VecModel).args or
                                                           key in inspect.getfullargspec(
                                                               gensim.models.base_any2vec.BaseWordEmbeddingsModel).args}
                                           )

        print('Start clustering..')
        self.gmm = mixture.GaussianMixture(**{key: value for key, value in kwargs.items()
                                              if key in inspect.getfullargspec(mixture.GaussianMixture).args})
        print('Finished clustering.')

        self.doc_vectors = np.array(self.model.docvecs.vectors_docs)
        self.gmm.fit(self.doc_vectors)
        self.BIC = self.gmm.bic(self.doc_vectors)
        self.cluster_center = self.gmm.means_
        print('BIC: {}'.format(self.BIC))

        self.get_document_topics()
        self.top_topic_center_words = pd.DataFrame(
            [self.most_similar_words_per_topic(topic, 200) for topic in range(self.gmm.n_components)])

    def get_string_vector(self, string, steps=10):
        '''
        The function takes a string (document) and
        transforms it to vector using a trained model.
        :param strings: new document string.
        :param steps: number of times to train the new document.
        :return: document vector.
        '''
        return self.model.infer_vector(string.split(), steps=steps).reshape(1,-1)

    def get_topic_weights(self, vector):
        '''
        The function takes a document vector
        and returns a distribution of a given document over all topics.
        :param vector: document vector.
        :return: probability distribution of the vector over all topics.
        '''
        return self.gmm.predict_proba(vector)

    def wordcloud_by_topic(self, topic, variant='sim', stop_words=None, n_words=100):
        '''
        Create a wordcloud to the defined topic.
        :param topic: number of a topic.
        :param n_words: number of words to be shown in a wordcloud.
        :return: a wordcloud with most common words.
        '''
        if variant == 'sim':
            text = self.top_topic_center_words.iloc[topic,:n_words]
            text = " ".join(text)
            wordcloud = WordCloud(max_font_size=50, max_words=n_words, stopwords=stop_words,
                                  background_color="white").generate(text)
        if variant=='count':
            text = self.topic_words.iloc[topic,:n_words]
            text = " ".join(text)
            wordcloud = WordCloud(max_font_size=50, max_words=n_words, stopwords=stop_words,
                                  background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.imshow(wordcloud, interpolation="bilinear", )
        ax.axis("off")
        return wordcloud


    def get_document_topics(self):
        '''
        Create data frame with the most frequent words per topic as well as the nearest words to the single topic centers.
        :return: self.topic_words data frame with 100 frequent words per topic and self.wordcloud_df containing all the
        document texts assigned to the resulted topics.
        '''
        document_topics = np.array(self.gmm.predict_proba(np.array(self.model.docvecs.vectors_docs)))
        most_relevant_topic_per_document = pd.DataFrame(document_topics).idxmax(1)
        kk = pd.DataFrame(self.documents).join(pd.DataFrame(most_relevant_topic_per_document, columns=['top_topic']))
        self.wordcloud_df = kk.groupby('top_topic')[0].apply(list).apply(" ".join).str.lower()
        self.topic_words = pd.DataFrame(
            [pd.Series(self.wordcloud_df.loc[topic].split()).value_counts().iloc[:100].index.values for topic in
             range(self.gmm.n_components)])

    def most_similar_words_per_topic(self, topic, n_words):
        '''
        The function returns most similar words to the selected topic.
        :param topic: topic number.
        :param n_words: number of words to be shown.
        :return: returns words which are most similar to the topic center as measured by cosine similarity.
        '''
        sims = cosine_similarity([self.cluster_center[topic]], self.model.wv.vectors)
        sims = np.argsort(sims)[0][::-1]
        text = [self.model.wv.index2word[k] for k in sims[:n_words]]
        return text

    def search_topic_by_term(self, term, variant='sim', method= 'combine', n_words=100):
        '''
        This function returns topic number and a wordcloud which represent defined search terms.
        :param term: a list with search terms, e.g. ['market','money']
        :param variant: if there is only one search term, there two variants to find a representative topic/wordcloud.
        if 'sim' is chosen the term is searched for among all the words that are nearest to the single topic centers.
        If 'count' is chosen only words from the texts which were assigned to the single topics are considered.
        :param method: if there are multiple search terms, three different methods can be used to find a representative
        wordcloud. Method 'combine' considers all the search terms as a single text and the process is simple to assigning
        new documents with .get_string_vector and .get_topic_weights. Method 'sim_docs' also considers all the search terms as
        a single text and searches for the most similar documents and topics these similar documents were assigned to.
        Third method (vec_sim) transforms single search terms into word vectors and considers cosine similarity between each word
        vector and topic centers.
        :param n_words: number of words to be shown in a wordcloud.
        :return: best matching topic and a wordcloud.
        '''
        assert isinstance(term, (list, tuple)), 'term parameter should be a list or a tuple'
        if len(term) == 1:
            if variant == "sim":
                matches = self.top_topic_center_words[self.top_topic_center_words == term[0]]
                best_matching_topic = pd.DataFrame(list(matches.stack().index)).sort_values(1).iloc[0][0]
                text = self.top_topic_center_words.loc[best_matching_topic].values
                text = " ".join(text)
            elif variant == "count":
                matches = self.topic_words[self.topic_words == term[0]]
                best_matching_topic = pd.DataFrame(list(matches.stack().index)).sort_values(1).iloc[0][0]
                text = self.wordcloud_df.loc[best_matching_topic]
            print("best_matching_topic", best_matching_topic)
            self.wordcloud_by_topic(best_matching_topic)

        elif len(term) > 1:

            if method == 'combine':
                string = ' '.join(term)
                vector = np.array(self.get_string_vector(string))
                best_matching_topic = self.gmm.predict(vector)[0]
                print("best_matching_topic", best_matching_topic)
                self.wordcloud_by_topic(best_matching_topic)

            elif method == 'sim_docs':
                string = ' '.join(term)
                vector = np.array(self.get_string_vector(string))
                docs_num = [self.model.docvecs.most_similar(positive=[np.array(vector).reshape(-1, )], topn=10)[i][0]
                            for i in range(10)]
                document_topics = np.array(self.gmm.predict(np.array(self.model.docvecs.vectors_docs)))
                unique, counts = np.unique(document_topics[docs_num], return_counts=True)
                top_topics = np.asarray((unique, counts)).T
                df = pd.DataFrame(top_topics).sort_values(by=1, ascending=False)
                df.columns = ['topic', 'frequency']
                best_matching_topic = df.iloc[0, 0]
                print("best_matching_topic", best_matching_topic)
                self.wordcloud_by_topic(best_matching_topic)

            elif method == 'vec_sim':
                vectors = [self.get_string_vector(t) for t in term]
                terms_df = pd.DataFrame({'topic': range(self.gmm.n_components)})
                for i in range(len(term)):
                    sims = [cosine_similarity([self.cluster_center[j]], vectors[i]) for j in
                            range(self.gmm.n_components)]
                    sims = [j for i in sims for j in i]
                    sims = [sims[i][0] for i in range(len(sims))]
                    terms_df[term[i]] = sims
                topics = [np.argsort(terms_df[term[i]].values)[::-1][0] for i in range(len(term))]
                s = pd.Series(topics)
                best_matching_topic = s.value_counts().index[0]
                print("best_matching_topic", best_matching_topic)
                self.wordcloud_by_topic(best_matching_topic)

    def infer_topics(self, text):
        """
        Infer topics from unseen documents.
        :param text: array or list of strings.
        :return: probability distribution of the text vector over all topics.
        """
        vec = self.get_string_vector(text)
        return self.get_topic_weights(vec)

    def start_webapp(self, model_path = None, title = "pvtm_webapp"):
        '''
        Run the dash app for visualization of results.
        :param model_path: path to the saved model.
        :param title: title displayed above the CMD window.
        :return: link to the running dash app.
        '''
        import os
        if not model_path:
            tmp_save_path = 'pvtm_model_tmp'
        self.save(tmp_save_path)
        time.sleep(1)
        os.system(f'start cmd.exe /c "TITLE {title}&& python webapp.py -m {tmp_save_path}"')


    def stop_webapp(self, title = "pvtm_webapp"):
        '''
        Stops running the dash app.
        :param title: title displayed above the CMD window.
        :return: stops running the dash app.
        '''
        import os
        print(title)
        os.system(f'taskkill /fi "WindowTitle eq {title}"')
        print("done")

    def save(self, savepath):
        '''
        Save the trained model.
        :param savepath: path the defined model should be stored in.
        '''
        joblib.dump(self, savepath)
    def load_example_data():
        '''
        Loads example text data.
        :return: an array with texts.
        '''
        df = pd.read_csv("data/sample_5000.csv")
        texts = df.text.values
        return texts
