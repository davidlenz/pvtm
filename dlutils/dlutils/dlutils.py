import ast
import datetime
import sys
import glob
import random
import re
# import stopwords_generator
import subprocess
import time
from collections import Counter
import os
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models.doc2vec import TaggedDocument
#from reportlab.graphics import renderPDF
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
# from svglib.svglib import svg2rlg
#import nltk
#import stop_words
# from langdetect import detect

#nltk.download('stopwords')
#from nltk.corpus import stopwords
#import ast

import pkg_resources





def get_all_stopwords(text_sample='This is an englisch sentence'):
    """ Combines Stopwords for englisch, german, french, and spanish from NLTK. Further adds stopwords from the stop_words module.
    Finally, stopwords from a text file stopwords.txt are added to come up with a list of stopwords."""
    # detect language
    # lang = detect(text_sample)
    # print('DETECTED LANGUAGE : {}'.format(lang))
    # get nltk stopwords for common languages
    stopwordssss = stopwords.words('german') + \
                   stopwords.words('english') + \
                   stopwords.words('french') + \
                   stopwords.words('spanish')
    
    
    return stopwordlist


def _find_language(text):
    if text != '':
        return detect(text[:5000])


def get_allowed_vocab(data, min_df=0.05, max_df=0.95):
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



def popularity_based_prefiltering(data, min_df=0.05, max_df=0.95, stopwords=None):
    """popularity based pre-filtering. Ignore rare and common words. Takes a list of strings as input. stopwords is a list of words if provided."""
    vocabulary = get_allowed_vocab(data, min_df=min_df, max_df=max_df)
    vocabulary = frozenset(vocabulary)
    if stopwords:
        stopwords = frozenset(stopwords)
    pp = []
    for i, line in enumerate(data):
        rare_removed = list(filter(lambda word: word.lower() in vocabulary, line.split()))
        if stopwords:
            stops_removed = list(filter(lambda word: word.lower() not in stopwords, rare_removed))
        pp.append(" ".join(stops_removed))
    return pp, vocabulary


def preprocess(str):
    """

    """
    # remove links
    str = re.sub(r'http(s)?:\/\/\S*? ', "", str)
    return str


def preprocess_document(text):
    """
    Checks if a character is alphanumeric or a space, if not it is replaced by ''.
    Splits the resulting String to return a list of words.
    """
    text = preprocess(text)
    return ''.join([x if x.isalnum() or x.isspace() or x=='ß' else " " for x in text])#.split()


def ngrams(inputt, n):
    """return n grams given inputt string"""
    inputt = inputt.split(' ')
    output = {}
    for i in range(len(inputt)-n+1):
        g = ' '.join(inputt[i:i+n])
        output.setdefault(g, 0)
        output[g] += 1
    return output

def easy_stopwords():
    """easy on the fly list of stopwords. may exlude words one wants to keep in certain contexts."""
    resource_package = 'dlutils'  # Could be any module/package name
    resource_path = '/'.join(['stopwords.txt'])  # Do not use os.path.join()
    template = pkg_resources.resource_string(resource_package, resource_path)
    # or for a file-like stream:
#     template = pkg_resources.resource_stream(resource_package, resource_path)
    stopwords = template.decode('utf-8')
    stopwords = stopwords.split('\r\n')
    return stopwords

def clean_svg(path):
    files = glob.glob(path + '*.svg')
    print(files)
    for file in files:
        print(file)
        print(file[-10:])
        if file[-10:] == '_clean.svg':
            continue
        new_file_name = file[:-4] + '_clean.svg'
        command = """scour -i {} -o {} --enable-viewboxing --enable-id-stripping 
        --enable-comment-stripping --shorten-ids --indent=none""".format(file, new_file_name)
        subprocess.call(command)
        print('ok')


class Documents(object):
    """

    """

    def __init__(self, documents):
        self.documents = documents

    def __iter__(self):
        for i, doc in enumerate(self.documents):
            yield TaggedDocument(words=doc, tags=[i])


def check_path(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def get_objects_by_cluster(out, column, label):
    """ Returns all rows from 'out' where 'column' equals 'label'. """
    return out.loc[out[column] == label].data.values


def get_topics(objects, stopwords, num_words=15):
    """ Returns the most common words found in the documents which belong to a specific topic, stopwords removed."""

    words = [preprocess_document(' '.join(x).lower()) for x in objects]
    # print(words)
    words = [word for sublist in words for word in sublist]


    filtered_words = [word for word in words if word not in stopwords and not word.isdigit()]
    count = Counter(filtered_words)
    return count.most_common()[:num_words]


def get_all_topics_from_centers(center, out, column, stopwordssss, num_words=10):
    """  """
    json_data = []
    for label in range(0, len(center)):
        cur_obj = {}

        objects = get_objects_by_cluster(out, column, label)
        topics = get_topics(objects, stopwordssss, num_words=num_words)

        cur_obj["topics"] = topics
        cur_obj["articles"] = objects
        json_data.append(cur_obj)

        if label % 20 == 0:
            print(label)

    topics = pd.DataFrame([p['topics'] for p in json_data])
    articles = pd.DataFrame([p['articles'] for p in json_data])
    return topics, articles


def get_most_similar_words_and_docs(center, model, num_words, num_docs):
    """
    Returns a dataframe with one row per cluster center.
    Each row holds the most similar words given the current cluster center
    and the most similar documents, both with similarity measurements.
    """
    _list = []
    for cent in center:
        # find the most similar words given the vector of the cluster center
        add_words = pd.DataFrame(model.wv.similar_by_vector(cent, topn=num_words))
        sim_words = add_words[0].values.tolist()
        sim_words_prob = add_words[1].values.tolist()

        # find the most similar documents given the current cluster center
        # can be interpreted as the documents that best describe the current cluster
        similar_docs = model.docvecs.most_similar([cent], topn=num_docs)
        sim_docs_indx = pd.DataFrame(similar_docs).values[:, 0].astype(int).tolist()
        sim_docs_prob = pd.DataFrame(similar_docs).values[:, 1].tolist()

        # append to list
        _list.append([center, sim_words, sim_words_prob, sim_docs_indx, sim_docs_prob])

    # list to df. Rename columns
    simsdf = pd.DataFrame(_list)
    simsdf.columns = ['center', 'sim_words', 'sim_words_prob', 'sim_docs_indx', 'sim_docs_prob']

    return simsdf


def get_headers_from_similar_docs(out, topics, idx, num_headers):
    """
    Returns the headlines (headline column must be named 'title' in the input dataframe 'out') from the documents
    that are closest to the current ('idx') cluster center,
    i.e. that best represent this topic.
    """
    try:
        _list = topics.iloc[idx].sim_docs_indx.values[0][:num_headers]
        probs = topics.iloc[idx].sim_docs_prob.values[0][:num_headers]

    except Exception as e:
        print('ignoring:', e)
        _list = topics.iloc[idx].sim_docs_indx[:num_headers]
        probs = topics.iloc[idx].sim_docs_prob[:num_headers]
    # print(_list)
    returns = [out.iloc[int(i)]['title'] for i in _list]

    return returns, probs, _list


def extract_time_info(infodf, datefield):
    infodf['time'] = infodf[datefield].dt.time
    infodf['year'] = infodf[datefield].dt.year
    infodf['hour'] = infodf[datefield].dt.ceil('H')
    infodf['day'] = infodf[datefield].dt.ceil('D')
    infodf['week'] = infodf[datefield].dt.to_period('W').dt.to_timestamp()
    infodf['month'] = infodf[datefield].dt.to_period('M').dt.to_timestamp()
    infodf['quarter'] = infodf[datefield].dt.to_period('Q').dt.to_timestamp()
    # list of offsets: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    return infodf


def show_topics_per_choosen_granularity(out, topic_label_column, TOPICS, date_column, TIMESTEP):
    count = Counter
    relevant_indx = out.loc[(out[date_column] == TIMESTEP) & (out[topic_label_column].isin(TOPICS))]
    a = relevant_indx[topic_label_column].values
    df = pd.DataFrame(list(count(a).values()), list(count(a).keys()), columns=[TIMESTEP]).T
    return df


def get_weighted_most_similar_words(model, word, topn=100, probability_multiplier=1000):
    """
    Takes a single lowercase word or a list of words and returns a list with the most similar words.
    Each word appears multiple times in the list, depending on how similar (cosine similarity)
    it is to the query word. Probability_multiplier is the number to multiply the similarity value with,
    so int(probability_multiplier x similarity) = number of occurences of the word in the output list.
    The list of words is shuffled before returning.
    topn is the number of similar words to be returned.
    model is a trained doc2vec model.
    """

    # get the most similar words
    words = np.array(model.wv.most_similar(word, topn=topn))[:, 0]
    # multiply the similarity of that word by the multiplier
    # then round and turn into integer
    word_counts = np.array(model.wv.most_similar(word, topn=topn))[:, 1].astype(float) * probability_multiplier
    word_counts = word_counts.round(0).astype(int)

    aa = []
    for i, word in enumerate(words):
        # multiply every word x times according to the similarity to the search word
        aa.append(' '.join([word] * word_counts[i]))

    print(aa)
    aa = [w for sublist in aa for w in sublist.split(' ')]
    random.shuffle(aa)
    return aa


def wordcloud_from_words(words, stopwords, outputfilepath='a.png', grey=False, store=False, show=True):
    from wordcloud import WordCloud
    from PIL import Image

    mask = np.array(Image.open("round_canvas.png"))

    def grey_color_func():
        return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

    wc = WordCloud(width=1400, height=800, max_words=1000, stopwords=stopwords,
                   margin=1, background_color='white', relative_scaling=0.5, random_state=10, mask=mask)

    combined_words = ' '.join(words)
    wc.generate(combined_words)
    # store default colored image
    default_colors = wc.to_array()
    fig = plt.figure(figsize=(24, 16))

    if grey:
        plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3), interpolation="bilinear")
    else:
        plt.imshow(default_colors, interpolation="bilinear")
    plt.axis("off")

    if show:
        plt.show()
    # return combined_words
    if store:
        wc.to_file("{}".format(outputfilepath))


def wordcloud_from_topic(topics, stopwords, topic_index,
                         outputfilepath='wc.png', show=False, return_=False, _save=True):
    from wordcloud import WordCloud
    from PIL import Image

    mask = np.array(Image.open("round_canvas.png"))

    combined_words = ' '.join(topics.top_words[topic_index]) + ' '.join(topics.sim_words[topic_index])
    wc = WordCloud(width=1400, height=800, max_words=1000, stopwords=stopwords, margin=1, background_color='white',
                   relative_scaling=0.5, random_state=10, mask=mask).generate(combined_words)

    # store default colored image
    default_colors = wc.to_array()

    # plt.axis("off")
    fig = plt.figure(figsize=(18, 14))

    plt.imshow(default_colors, interpolation="bilinear")
    plt.axis("off")
    if show:
        plt.show()
    if return_:
        return fig
    if _save:
        plt.savefig(outputfilepath)


def plot_timelines(topics, out, searchword, search_column, GRANULARITY, n_random=10, num_top_words=8, num_headers=5):
    if searchword == 'none':
        TOPICS = random.sample(range(topics.shape[0]), n_random)
    else:
        TOPICS = [i for i, top in topics.iterrows() if
                  searchword in top[search_column]]  # get indices of the rows where the term is in the specified column

    granulars = out.sort_values('date')[GRANULARITY].unique()  # [-48:]
    _list = [show_topics_per_choosen_granularity(out, 'gmm_top_topic', TOPICS, GRANULARITY, granular) for granular in
             granulars]

    tmpdf = pd.concat(_list)
    traces = []
    for top in TOPICS:
        words1 = '# Top Words: ' + ', '.join(topics.iloc[top].top_words[:num_top_words])
        words2 = '# Similar Words: ' + ', '.join(topics.iloc[top].sim_words)
        words3 = '#<br># '.join(
            [tmp for tmp in get_headers_from_similar_docs(out, topics, [top], num_headers=num_headers)])
        words = '<br>'.join(['Topic: ' + str(top), words1, words2, words3])

        try:
            trace1 = go.Scatter(
                x=tmpdf.index,
                y=tmpdf[top].values,
                mode='lines',
                name='Topic {}'.format(top),
                text=words,
                textposition='left'
            )
            traces.append(trace1)
        except Exception as e:
            print(e)
            pass
    data = traces
    layout = go.Layout(
        title=searchword,
        hovermode='closest',
        showlegend=True,

    )
    fig = go.Figure(data=data, layout=layout)
    return fig
    # plot_url = py.iplot(fig, filename='Example_Topics_over_time')
    # plt.show(plot_url)


def load_topics_dataframe(topics_path):
    # read from file
    topics = pd.read_csv(topics_path, index_col='Unnamed: 0')

    # there are lists stored in the df columns as strings
    # we need to evaluate them to python lists via ast.literal_eval
    for column in topics.columns:
        if column != 'center':
            # print(topics[column])
            topics[column] = topics[column].apply(lambda x: ast.literal_eval(x))
    return topics


def lookup(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    dates = {date: pd.to_datetime(date) for date in s.unique()}
    return s.map(dates)


def split_data_columns(stry):
    return re.sub("[^\w]", " ", stry).split()


def load_document_dataframe(path, columns):
    # read from file
    print('load document dataframe')
    print('start reading csv...')
    # out = pd.read_csv(path, parse_dates=['date'], verbose=True)
    out = pd.read_csv(path, verbose=True)
    print('finished reading csv file. start date conversion')
    out['date'] = lookup(out['date'])
    print('finished date conversion, evaluate columns where necessary...')

    out['data'] = out.data.apply(split_data_columns)

    # evaluate the necessary columns
    for column in columns:
        if column in out.columns:
            print(column)
            out[column] = out[column].apply(lambda x: ast.literal_eval(x))

    # return df
    return out


def get_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d__%H-%M-%S')


def compare_article_text_with_headers_from_similiar_topics(out, topics, relevant_index):
    print(out.iloc[relevant_index].text)
    topic_index = out.gmm_topics[relevant_index]  # get the relevant topics for a specific document
    print()
    print('Document headers from topics this article is similar to:')
    for tops in topic_index:
        print()
        print('Topic {} with {:.2f} %'.format(tops, 100 * out.iloc[relevant_index].gmm_probas[tops]))
        print('Most representative headlines for this Topic:')
        print(get_headers_from_similar_docs(out, topics, [tops], 5))


def sort_date_string(L):
    """
    Sorts a date string. Datestring has to be in the format 'year-xx' where x can be any of week, month, quarter.
    Returns the input if there is nothing to split.
    """
    try:
        splitup = L.split('-')
        return splitup[0], splitup[1]
    except:
        return L


def get_topic_importance_df(level_of_aggregation, out, datefield='date'):
    """
    Aggregate the probabilities over all Documents wrt their topics given time granularity, for example 'year', 'quarter_year' or 'month_year'.
    Returns df with one row per time granularity (one row per year for instance) and one column per topic with the aggregated importance.
    Based on GMM probabilites.
    """

    # get the topic probabilites on document level from the out dataframe
    # turn them into a df with one column per topic and num documents number of rows


    out[datefield] = out[datefield].apply(pd.to_datetime, errors='coerce')
    out = extract_time_info(out, datefield=datefield)

    probas = out.gmm_probas.apply(pd.Series)

    # join the respective time aggregation level from the out dataframe
    probas[level_of_aggregation] = out[level_of_aggregation]

    # group by time aggregation of choice and sum the probabilites per topic
    topic_importance_df = probas.groupby(level_of_aggregation).mean()

    # sort the dataframe based on the date strings
    old_index = topic_importance_df.index.tolist()
    new_index = pd.DataFrame(index=sorted(old_index, key=sort_date_string))
    topic_importance_df = new_index.join(topic_importance_df)

    return topic_importance_df


def get_top_n_trending_topics(imp_per_my, k_timesteps, column, nlargest=10):
    """
    Calculate growth rates from period to period.
    """

    pcchange = imp_per_my.pct_change(k_timesteps)  # .dropna()

    # transpose the pcchange column
    pcchange_transposed = pcchange.T

    # find the top n largest indices in the transposed pcchange dataframe
    nlargest_indices = [[pcchange_transposed[column].nlargest(nlargest).index.values,
                         pcchange_transposed[column].nlargest(nlargest).values] for column in
                        pcchange_transposed.columns]

    # list to dataframe, set index from the original pcchange
    top_n_trending_topics = pd.DataFrame(nlargest_indices, index=pcchange.index)

    return top_n_trending_topics


# def R_wordclouds_from_topics(out, stopwords):
#     import subprocess
#
#     for i in out.gmm_top_topic.unique():
#         a = out[out.gmm_top_topic.isin([i])].data.values
#
#         cc = [word.lower().strip().replace('ä', 'ae').replace('ü', 'ue').replace('ö', 'oe').replace('ß', 'ss') for _list
#               in a for word in _list if (word not in stopwords) and (not word.isdigit())]
#
#         with open('Output/wordclouds/test_{}.txt'.format(i), 'w', encoding='utf-8') as textfile:
#             textfile.write('\n'.join(cc))
#
#
#     subprocess.call(['RScript', 'wordcloud_test.R'])
#
#     files = glob.glob('Output/wordclouds/*.txt')
#     for file in files:
#         file = os.path.abspath(file)
#         filepdf = file + '.pdf'
#         filesvg = file.replace('.txt', '.svg')
#         sstr = """inkscape,--file={},--export-plain-svg={}""".format(filepdf, filesvg)
#         sstr = sstr.split(',')
#         print(sstr)
#     # subprocess.call(['nbconvert', 'R_wordclouds_to_svg.ipynb'])
#     subprocess.call(
#         'jupyter nbconvert --execute R_wordclouds_to_svg.ipynb --ExecutePreprocessor.timeout={} --output Output/R_wordclouds_to_svg.html'.format(
#             settings.XSNE_TIMEOUT))
#
#     # from svg to png with inkscape
#     subprocess.call(['convert_wordclouds_from_svg_to_png_using_inkscape.bat'])
#
#
# def colored_bhtsne(out):
#     def plot_bhtsne(new, colors, bhtsne, out, name):
#
#         plt.figure(figsize=(settings.BHTSNE_PLOT_SIZE * 1.5, settings.BHTSNE_PLOT_SIZE))
#
#         if settings.SAMPLE_DATA_FOR_BHTSNE:
#             rndinds = random.sample(range(len(out)), settings.BHTSNE_NUM_SAMPLES)
#             plt.scatter(bhtsne['x'].values[rndinds], bhtsne['y'].values[rndinds])  # , c=new[0][rndinds])
#             plt.title('Symmetric tSNE for 5000 sample points.')
#         else:
#             plt.scatter(bhtsne['x'].iloc[:len(out)].values, bhtsne['y'].iloc[:len(out)].values, c=new[0])
#             plt.savefig('Output/colored_bhtsne_{}_{}.png'.format(settings.BHTSNE_PLOT_SIZE, name), bbox_inches='tight')
#
#     bhtsne = pd.read_csv('Output/bhtsne.csv', names=['x', 'y'])
#
#     # create a dataframe of different colors
#     colors = np.array(list(matplotlib.colors.cnames.items()))[:, 0]
#     a = pd.DataFrame(colors)
#     colors = pd.concat([a]).reset_index()
#
#     # join colors on hard topic labels
#     new = out.join(colors, on='gmm_top_topic', rsuffix='r_')
#     plot_bhtsne(new, colors, bhtsne, out, name='gmm')
#
#     if 'label_k_means' in out.columns.values:
#         new = out.join(colors, on='label_k_means', rsuffix='r_')
#         plot_bhtsne(new, colors, bhtsne, out, name='k-means')
#
#         # new = out.join(colors, on = 'label_mean_shift', rsuffix='r_')



def svg_to_pdf(in_path):
    out_path = in_path[:-4] + '.pdf'
    print('create pdf :', out_path)
    # svg to pdf
    drawing = svg2rlg(in_path)
    renderPDF.drawToFile(drawing, out_path)


def load_pvtm_outputs(path):
    """
    Load all relevant outputs from PVTM.
    This includes the doc2vec model, gmm model, topics dataframe and the documents dataframe.
    """

    # Load doc2vec model
    model = gensim.models.doc2vec.Doc2Vec.load(path + '/doc2vec.model')

    # load document dataframe
    data = load_document_dataframe('{}/documents.csv'.format(path),
                                   ['gmm_topics', 'gmm_probas'])

    # load topics dataframe
    topics = load_topics_dataframe('{}/topics.csv'.format(path))

    # load gmm model
    gmm = joblib.load('{}/gmm.pkl'.format(path))

    # docvecs
    vectors = np.array(model.docvecs.vectors_docs).astype('float64')
    vecs_with_center = pd.read_csv('{}/vectors_with_center.tsv'.format(path), sep='\t', index_col=0)
    return model, gmm, data, topics  # , vectors, vecs_with_center


def spacy_lemmatizer(text, nlp, LEMATIZER_N_THREADS, LEMMATIZER_BATCH_SIZE):
    """
    text is a list of string. nlp is a spacy nlp object. Use nlp.disable_pipes('tagger','ner') to speed up lemmatization
    """
    doclist = list(nlp.pipe(text, n_threads=LEMATIZER_N_THREADS, batch_size=LEMMATIZER_BATCH_SIZE))

    docs = []
    for i, doc in enumerate(doclist):
        docs.append(' '.join([listitem.lemma_ for listitem in doc]))
    return docs
