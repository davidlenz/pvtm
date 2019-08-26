import glob

import heise_pvtm_helper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import os

def most_similar_words_per_topic(topic, n_words, pvtm):
    return pd.DataFrame(pvtm.model.wv.similar_by_vector(pvtm.cluster_center[topic],
                                                        topn=n_words),
                        columns=['word', "similarity"]).word.values


def search_topic_by_term(term="windows", variant="sim"):
    """ Available modes are 'count' and 'sim' """

    if variant == "sim":
        matches = pvtm.most_similar_words_per_topic[pvtm.most_similar_words_per_topic == term]
    elif variant == "count":
        matches = pvtm.topic_words[pvtm.topic_words == term]
    else:
        print("choose other variant ('count' or 'sim')")
    print(list(matches.stack().index))
    return pd.DataFrame(list(matches.stack().index)).sort_values(1)

def search_topic_by_term_plot(term="windows", mode="best", variant="sim", date_unit="month"):
    """ Available modes are 'best' and 'all' """

    if variant == "sim":
        matches = pvtm.most_similar_words_per_topic[pvtm.most_similar_words_per_topic == term]
    elif variant == "count":
        matches = pvtm.topic_words[pvtm.topic_words == term]
    else:
        print("choose other variant ('count' or 'sim')")
    print(list(matches.stack().index))
    best_matching_topic = pd.DataFrame(list(matches.stack().index)).sort_values(1).iloc[0][0]
    print("best_matching_topic", best_matching_topic)

    if mode == "best":
        if variant == 'sim':
            text = pvtm.most_similar_words_per_topic.loc[best_matching_topic].values
            text = " ".join(text)
        elif variant == 'count':
            text = pvtm.wordcloud_df.loc[best_matching_topic]

        wordcloud = WordCloud(max_font_size=50, max_words=n_words, background_color="white").generate(text)
        plt.figure(figsize=(8, 6))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

        pvtm.topic_over_time_df.groupby(date_unit)[best_matching_topic].mean().plot(ax=ax2)
        ax1.imshow(wordcloud, interpolation="bilinear", )
        ax1.axis("off")
        plt.show()

    elif mode == "all":
        for idx, col in list(matches.stack().index):
            if variant == 'sim':
                text = pvtm.most_similar_words_per_topic.loc[idx].values
                text = " ".join(text)
            elif variant == 'count':
                text = pvtm.wordcloud_df.loc[idx]

            print(f"Rank of word {term} in the topic {idx} : {col}")
            wordcloud = WordCloud(max_font_size=50, max_words=n_words, background_color="white").generate(text)
            plt.figure(figsize=(8, 6))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

            pvtm.topic_over_time_df.groupby(date_unit)[idx].mean().plot(ax=ax2)

            ax1.imshow(wordcloud, interpolation="bilinear", )
            ax1.axis("off")

            plt.show()

        else:
            print("no valid mode choosen (use 'best' or 'all')")
    return best_matching_topic


date_unit = "month"
n_words = 1000
models = [folder.replace("results\\","") for folder in glob.glob('results/*')]
gt_terms = [file.replace("../_doc/PVTM/code/gt_data\\", "").replace(".csv", "") for file in
            glob.glob("../_doc/PVTM/code/gt_data/*.csv")] + ["windows"]
print(models)
print(gt_terms)

df_heise = heise_pvtm_helper.load_heise_data(nrows=5000000)


for modelname in models:
    print("Load pvtm model..")
    pvtm = joblib.load("results/" + modelname + "/pvtm_model")

    print("Calculate similar words and topics over time..")
    pvtm.most_similar_words_per_topic = pd.DataFrame(
        [most_similar_words_per_topic(topic, n_words, pvtm) for topic in range(pvtm.gmm.n_components)])

    pvtm.topic_over_time_df = pd.DataFrame(pvtm.document_topics).join(df_heise.reset_index()[[date_unit, "date"]])

    os.makedirs(f"results/{modelname}/gt_analysis/", exist_ok=True)
    print("Start iteration over gt terms..")
    for gt in gt_terms:

        for variant in ["count", "sim"]:
            print(gt, variant)
            try:
                pvtm_topic = search_topic_by_term(term=gt, variant=variant).iloc[0][0]
            except Exception as e:
                print(e)
                continue

            df = pvtm.topic_over_time_df.groupby(date_unit).mean()
            df["month"] = df.index.values

            # read gt data
            wiki = pd.read_csv(f'../_doc/PVTM/code/gt_data/{gt}.csv', skiprows=1)
            wiki = wiki.set_index(pd.to_datetime(wiki["Month"]))

#             # restrict data to same time period
            wiki = wiki[wiki.index <= df.index.max()]
            df = df[df.index >= wiki.index.min()]

            # generate month for pvtm data in the gt data time format
            xticks = ["-".join(list(k)) for k in
                      list(zip(df.index.year.astype(str), df.index.month.astype(str)))]  # [::12]
            xticks = [f"{tick}-01" for tick in xticks]
            xticks = [tick if len(tick) == 10 else tick[:5] + '0' + tick[5:] for tick in xticks]
            df['x'] = xticks
            df = df.set_index(pd.to_datetime(df.x))

            # join pvtm and gt data
            joined = wiki.join(df[pvtm_topic])
            joined.columns = ['monat', 'google_trends', 'pvtm']

            joined.google_trends = joined.google_trends.replace('<1', 1)
            # fig, ax = plt.subplots()
            # ax1 = ax.twinx()
            # plot stuff
            ax1 = joined.pvtm.ewm(span=12).mean().plot(y='pvtm', label='PVTM')
            ax = joined.google_trends.astype(float).plot(y='google_trends', secondary_y=True, label='Google Trends',
                                                         linestyle='--')

            lines = ax1.get_lines() + ax.get_lines()
            plt.legend(lines, [l.get_label() for l in lines], loc='best')

            ax1.set_xlabel('')
            ax.set_ylabel('Google Trends Index')
            ax1.set_ylabel('PVTM Topic Importance')
            plt.title(f"Comparison Google Trends vs PVTM for {gt} ({variant})".title())
            plt.tight_layout()
            plt.savefig(f'results/{modelname}/gt_analysis/{gt}_.pdf', bbox_inches="tight")
            plt.savefig(f'results/{modelname}/gt_analysis/{gt}_.png', bbox_inches="tight")
            plt.close()

            x = joined.values[:, 1:]  # .T[::-1].T
            joined.to_csv(f'results/{modelname}/gt_analysis/{gt}.csv', header=False, index=False)
            pd.DataFrame(x).to_csv(f'results/{modelname}/gt_analysis/{gt}_eviews.csv', header=False, index=False)
            y = pvtm.topic_over_time_df.groupby(date_unit).mean()[pvtm_topic].to_csv(f'results/{modelname}/gt_analysis/pvtm_{gt}_eviews.csv',
                                                                        header=False, index=False)






################################
### Plot GT Analysis Overview ##
################################
import pandas as pd
import glob

results = glob.glob("results/**/gt_analysis/pvtm*.csv")
# results = [res.replace("results\\","") for res in results]

df = pd.DataFrame([res.split('\\') for res in results], columns = ['resultspath', "run", "egal", "topic"])
df["path"] = results
df.topic = df.topic.str.replace("pvtm_","").str.replace("_eviews.csv","")
df['imagepath'] = df.path.str.replace("pvtm_","").str.replace("_eviews.csv","_.png")

df.groupby('topic').run.apply(list).apply(pd.Series)


topics = df.groupby('topic').imagepath.apply(list).apply(pd.Series)

import matplotlib.pyplot as  plt
import matplotlib.image as mpimg

fig, ax = plt.subplots(topics.shape[0], 4, figsize=(30, 140))

for k, (i, row) in enumerate(topics.iterrows()):

    for j in range(len(row)):
        path = row[j]

        if len(str(path)) > 5:
            name = path.split("_")[1]
            img = mpimg.imread(path)
            imgplot = ax[k][j].imshow(img)
            ax[k][j].set_title(f"{i} {name}")
        ax[k][j].axis("off")

plt.savefig('../_doc/PVTM/overview_gt_analysis.pdf', bbox_inches="tight")
plt.close()
