import numpy as np
np.random.seed(23)
import random
random.seed(1)

import os
import nltk
import pandas as pd
from nltk.corpus import stopwords
from pvtm.pvtm import PVTM
from sklearn.externals import joblib

import heise_pvtm_helper

# Regarding Gensim deterministic results
# https://github.com/RaRe-Technologies/gensim/wiki/Recipes-&-FAQ#q11-ive-trained-my-word2vecdoc2vecetc-model-repeatedly-using-the-exact-same-text-corpus-but-the-vectors-are-different-each-time-is-there-a-bug-or-have-i-made-a-mistake-2vec-training-non-determinism
# https://groups.google.com/forum/?pli=1#!topic/gensim/TCIrgMagoFc
import os
#print(os.environ['PYTHONHASHSEED'])
print('PYTHONHASHSEED:', os.environ.get('PYTHONHASHSEED'))
print("make sure that the PYTHONHASHSEED IS SET TO 1")

epochs=10
nrows = 200000
n_topics = 675
min_df, max_df = 0.0005, 0.65
text_col = "lemmatized" # name of the column holding the text
lemmatized=True # is the text lemmatized? If not, it will get lemmatized.
prefix="original" # to distinguish runs with similar hyperparameters


# get stopwords
stop_words = heise_pvtm_helper.get_stopwords()
df = heise_pvtm_helper.load_heise_data(nrows)
print("Data:", df.shape)

pvtm = PVTM(df[text_col].values, preprocess=True, lemmatized=lemmatized, min_df=min_df, max_df=max_df)


## Watch out! Parameters are currently hard set in the pvtm runfile ##
pvtm.fit(vector_size=100,
         hs=0,
         dbow_words=1, # train word vectors!
         dm=0, # Distributed bag of words (=word2vec-Skip-Gram) (dm=0) OR distributed memory (=word2vec-cbow) (dm=1)
         epochs=epochs, # doc2vec training epochs
         window=1, # doc2vec window size
         #seed=123, # doc3vec random seed
         #random_state = 123, # gmm random seed
         min_count=5, # minimal number of appearences for a word to be considered
         workers=1, # doc2vec num workers
         alpha=0.025, # doc2vec initial learning rate
         min_alpha=0.025, # doc2vec final learning rate. Learning rate will linearly drop to min_alpha as training progresses.
         n_components=n_topics, # number of Gaussian mixture components, i.e. Topics
         covariance_type='diag', # gmm covariance type
         verbose=1, # verbosity
         n_init=1, # gmm number of initializations

         )

# pvtm.topic_over_time_df = pd.DataFrame(pvtm.document_topics).join(df.reset_index()[["date", 'year', "month", 'quarter']])

bic = "{:10.4f}".format(float(pvtm.gmm.bic(pvtm.doc_vectors)))
savename = f"{bic}_{prefix}_mindf_{min_df}_maxdf_{max_df}_ntopics_{n_topics}_epochs_{epochs}_lemmatize_{lemmatized}_nrows_{nrows}"
print("savename", savename)
# def search_topic_by_term(self, term="windows", mode="best", variant="sim", date_unit="month"):
#     """ Available modes are 'best' and 'all' """
#
#     if variant == "sim":
#         matches = self.most_similar_words_per_topic[self.most_similar_words_per_topic == term]
#     elif variant == "count":
#         matches = self.topic_words[self.topic_words == term]
#     else:
#         print("choose other variant ('count' or 'sim')")
#     print(list(matches.stack().index))
#     best_matching_topic = pd.DataFrame(list(matches.stack().index)).sort_values(1).iloc[0][0]
#     print("best_matching_topic", best_matching_topic)
#
#     if mode == "best":
#         if variant == 'sim':
#             text = self.most_similar_words_per_topic.loc[best_matching_topic].values
#             text = " ".join(text)
#         elif variant == 'count':
#             text = self.wordcloud_df.loc[best_matching_topic]
#
#         wordcloud = WordCloud(max_font_size=50, max_words=n_words, background_color="white").generate(text)
#         plt.figure(figsize=(8, 6))
#
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
#
#         self.topic_over_time_df.groupby(date_unit)[best_matching_topic].sum().plot(ax=ax2)
#         ax1.imshow(wordcloud, interpolation="bilinear", )
#         ax1.axis("off")
#         plt.show()
#
#     elif mode == "all":
#         for idx, col in list(matches.stack().index):
#             if variant == 'sim':
#                 text = self.most_similar_words_per_topic.loc[idx].values
#                 text = " ".join(text)
#             elif variant == 'count':
#                 text = self.wordcloud_df.loc[idx]
#
#             print(f"Rank of word {term} in the topic {idx} : {col}")
#             wordcloud = WordCloud(max_font_size=50, max_words=n_words, background_color="white").generate(text)
#             plt.figure(figsize=(8, 6))
#
#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
#
#             self.topic_over_time_df.groupby(date_unit)[idx].sum().plot(ax=ax2)
#
#             ax1.imshow(wordcloud, interpolation="bilinear", )
#             ax1.axis("off")
#
#             plt.show()
#
#         else:
#             print("no valid mode choosen (use 'best' or 'all')")
#
# pvtm.search_topic_by_term = search_topic_by_term.__get__(pvtm)


# save model to disk
os.makedirs("results/" + savename + "/", exist_ok=True)
joblib.dump(pvtm, "results/" + savename + "/pvtm_model")
