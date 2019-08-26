import glob
import os
import subprocess

import heise_pvtm_helper
import matplotlib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import svgutils.compose as sc
from PyPDF2 import PdfFileMerger
from nltk.corpus import stopwords
from pandas.tseries.offsets import MonthBegin, QuarterBegin, YearBegin, MonthEnd
from reportlab.graphics import renderPDF
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
from svglib.svglib import svg2rlg
from wordcloud import WordCloud
import os
n_words = 100
use_latest_run = True  # use the latest run or some specific run? If the latter, fill in below
date_unit = "year"


# if use_latest_run:
#     modelname = max(glob.iglob('results/**'), key=os.path.getctime).replace("results\\", "")
# else:
#     modelname = 'mindf_0.0005_maxdf_0.65_ntopics_15_epochs_15_nrows_200000'


models = [folder.replace("results\\","") for folder in glob.glob('results/*')]

for modelname in models:
    print(f"Model: {modelname}")
    print("Load pvtm model...")
    pvtm = joblib.load("results/" + modelname + "/pvtm_model")
    print("Successfully loaded pvtm model.")

    #########################'##
    ## PARAMETERS TO PDF FILE ##
    ##########################'#

    d2v_params = pvtm.model.__dict__.copy()
    [d2v_params.pop(key, None) for key in [ 'vocabulary', 'trainables', 'wv', 'docvecs', "load"]]

    param_str_d2v = "Doc2Vec:\n\n" + str(d2v_params).replace(',', '\n').replace('{', '').replace('}', '')
    param_str_gmm = "GMM:\n\n" + str(pvtm.gmm.get_params()).replace(',', '\n').replace('{', '').replace('}', '')
    param_str_tfidf = "Tf-Idf:\n\n" + str(pvtm.tfidf.get_params()).replace(',', '\n').replace('{', '').replace('}', '')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,4))
    ax1.text(0, 0, param_str_d2v)
    ax1.axis("off")
    ax2.text(0.2, 0, param_str_gmm)
    ax2.axis("off")
    ax3.text(0, 0, param_str_tfidf)
    ax3.axis("off")
    plt.savefig(f"results/{modelname}/model_parameters.png", bbox_inches='tight')
    plt.savefig(f"results/{modelname}/model_parameters.pdf", bbox_inches='tight')
    plt.close()


    nrows= int(modelname.split("_nrows_")[-1].replace("_nrows_", ""))  # get data split from file name

    print(f"nrows: {nrows} ")

    df = heise_pvtm_helper.load_heise_data(nrows)

    topic_over_time_df = pd.DataFrame(pvtm.document_topics).join(df.reset_index()[["date", 'year', "month", 'quarter']])
    topic_over_time_df.to_csv(f"results/{modelname}/topic_over_time.tsv", sep='\t')

    # get stopwords
    stop_words = heise_pvtm_helper.get_stopwords()

    # create results directory
    os.makedirs("results/" + modelname + '/', exist_ok=True)

    ###########################
    ## SIMILARITY WORDCLOUDS ##
    ###########################

    # def best_words(topic, n_words):
    #     sims = cosine_similarity([pvtm.cluster_center[topic]], pvtm.model.wv.vectors)
    #     sims = np.argsort(sims)[0][::-1]
    #     text = [pvtm.model.wv.index2word[k] for k in sims[:n_words]]
    #     return text
    #
    #
    # best_words_df = pd.DataFrame([best_words(topic, n_words) for topic in range(pvtm.gmm.n_components)])
    # best_words_df.to_csv(f"results/{modelname}/top_words_sim.csv")

    for topic in range(pvtm.gmm.n_components):
        print(topic)

        text = pd.DataFrame(pvtm.model.wv.similar_by_vector(pvtm.cluster_center[topic],
                                                            topn=100),
                            columns=['word', "similarity"]).word.values
        text = ', '.join(text)
        wordcloud = WordCloud(max_font_size=50, max_words=n_words, background_color="white").generate(text)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        topic_over_time_df.groupby(date_unit)[topic].mean().plot(ax=ax2)
        ax1.imshow(wordcloud, interpolation="bilinear", )
        ax1.axis("off")

        plt.savefig(f"results/{modelname}/sim_{topic}.pdf", bbox_inches="tight")
        plt.savefig(f"results/{modelname}/sim_{topic}.png", bbox_inches="tight")
        plt.close()

    ###########################
    ## WORD COUNT WORDCLOUDS ##
    ###########################

    jk = pd.DataFrame(pvtm.document_topics).idxmax(1)
    kk = pd.DataFrame(pvtm.documents).join(pd.DataFrame(jk, columns=['top_topic']))
    kk[0] = kk[0].apply(lambda x: " ".join([word for word in x.split() if word.lower() not in stop_words]))
    wordcloud_df = kk.groupby('top_topic')[0].apply(list).apply(" ".join).str.lower()
    topic_words = pd.DataFrame(
        [pd.Series(wordcloud_df.loc[topic].split()).value_counts().iloc[:100].index.values for topic in
         range(pvtm.gmm.n_components)])

    topic_words.to_csv(f"results/{modelname}/topic_words.csv")

    for topic in range(pvtm.gmm.n_components):
        print(topic)
        wordcloud = WordCloud(max_font_size=50, max_words=n_words, background_color="white").generate(
            wordcloud_df.loc[topic])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        topic_over_time_df.groupby(date_unit)[topic].mean().plot(ax=ax2)
        ax1.imshow(wordcloud, interpolation="bilinear", )
        ax1.axis("off")

        plt.savefig(f"results/{modelname}/count_{topic}.pdf", bbox_inches="tight")
        plt.savefig(f"results/{modelname}/count_{topic}.png", bbox_inches="tight")
        plt.close()




    ###############################
    ## MERGE PDF FILES TO REPORT ##
    ###############################


    def svg_to_pdf(in_path, out_path):
        # svg to pdf
        drawing = svg2rlg(in_path)
        renderPDF.drawToFile(drawing, out_path)


    print('Start merging...')
    # merge topic pdf files into single pdf file
    pdfs = glob.glob(f"results/{modelname}/count_*.pdf")
    pdfs = [pdf for pdf in pdfs if '_' != pdf.split('\\')[1][0]]
    pdfs = [f"results/{modelname}/model_parameters.pdf"] + pdfs
    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write(f"results/{modelname}/_all_topics_count.pdf")
    merger.close()

    print('Start merging...')
    # merge topic pdf files into single pdf file
    pdfs = glob.glob(f"results/{modelname}/sim_*.pdf")
    pdfs = [pdf for pdf in pdfs if '_' != pdf.split('\\')[1][0]]
    pdfs = [f"results/{modelname}/model_parameters.pdf"] + pdfs
    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write(f"results/{modelname}/_all_topics_sim.pdf")
    merger.close()
