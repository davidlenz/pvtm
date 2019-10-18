<h1 align="center">PVTM</h1>
<p align="center">Paragraph Vector Topic Model</p>

**PVTM** represents documents in a semantic space using [Doc2Vec](https://arxiv.org/abs/1405.4053/), which are then clustered using [Gaussian mixture models](https://link.springer.com/referenceworkentry/10.1007%2F978-1-4899-7488-4_196) (GMM). Doc2Vec has been shown to capture latent variables of documents, e.g., the underlying topics of a document. Clusters of documents in the vector space can be interpreted and transformed into meaningful topics by means of Gaussian mixture modeling.

<h2 align="center">Highlights</h2>

-  :speech_balloon: **Easily identify latent topics in large text corpora** 
-  :chart_with_upwards_trend: **Detect trends and measure topic importance over time** 
-  :bar_chart: **Identify topics in unseen documents** 
-  :telescope: **Built-In text preprocessing** 

<h2 align="center">Install</h2>

Install the module via `pip` command.

```
pip install pvtm 
```

<h2 align="center">Getting Started</h2>
<h3 align="center">Importing & Preprocessing documents</h3>

Once you have installed the **pvtm** module, you can conduct analysis on your text documents stored in a *.txt* or *.csv* file.
The example below considers [reuters dataset](https://keras.io/datasets/#reuters-newswire-topics-classification) from `keras.datasets`.

```python
import pandas as pd
df = pd.read_csv("data/sample_5000.csv")
texts = df.text.values
```
After that, `PVTM` object should be created with the defined input texts.
Parameter `lemmatized` should be set to `False` when documents' texts should be lemmatized. However, take into account that this step could lead to improved results but also takes some time depending on the size of the document corpus. If you want to lemmatize your texts, you should first download [language models](https://spacy.io/usage/models/) and set the parameter lang, e.g. `lang='en'`. 
Set the parameter `preprocess=True` when the documents texts should be preprocessed, e.g. removal of special characters, number, currency symbols etc.
With the parameters `min_df` and `max_df` you set the thresholds for very rare/common words which should not be included in the corpus specific vocabulary. Further, you can also exclude language specific stopwords by importing your own stopwords list or using nlkt library as shown below.  

```python
from pvtm.pvtm import PVTM
from pvtm.pvtm import clean
import nltk
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english') + ['reuter', '\x03'])
stop_words = list(stop_words)
pvtm = pvtm.PVTM(texts, lemmatized = True, stopwords=stop_words)
```

<h2 align="center">Training</h2>

The next step includes training the Doc2Vec model and clustering of the resulted document vectors by means of GGM. For this, you only need to call the `pvtm.fit()` method and pass all the [parameters](pvtm#parameters) needed for the Doc2Vec model training and GMM clustering. For more detailed description of the parameters see information provided by [gesim](https://radimrehurek.com/gensim/models/doc2vec.html)(Doc2Vec model) and [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)(GMM).

```python
pvtm.fit(n_components = 15, vector_size = 30)
```

<h2 align="center">Visualize topics</h3>

The words closest to a topic center vector are considered as topic words. You can visualize topic words with a wordcloud:

```python
pvtm.wordcloud_by_topic(0)
pvtm.wordcloud_by_topic(5)
pvtm.wordcloud_by_topic(16)
```


<h2 align="center">Insert some wordclouds here</h3>




<h3 align="center">Parameters</h3>


| param        | default | description                                                                                        |
|--------------|-------|----------------------------------------------------------------------------------------------------|
| vector_size            | 300     | dimensionality of the feature vectors (Doc2Vec)                                        |
| n_components            | 15     | number of Gaussian mixture components, i.e. Topics (GMM)                                        |
| hs           | 0     | negative sampling will be used for model training (Doc2Vec)                                        |
| dbow_words   | 1     | simultaneous training of word vectors and document vectors (Doc2Vec)                               |
| dm           | 0     | Distributed bag of words (word2vec-Skip-Gram) (dm=0) OR distributed memory (dm=1)                  |
| epochs       | 1     | training epochs (Doc2Vec)                                                                          |
| window       | 1     | window size (Doc2Vec)                                                                              |
| seed         | 123   | seed for the random number generator (Doc2Vec)                                                     |
| min_count    | 5     | minimal number of appearences for a word to be considered (Doc2Vec)                                |
| workers      | 1     | number workers (Doc2Vec)                                                                           |
| alpha        | 0.025 | initial learning rate (Doc2Vec)                                                                    |
| min_alpha    | 0.025 | doc2vec final learning rate. Learning rate will linearly drop to min_alpha as training progresses. |
| random_state | 123   | random seed (GMM)                                                                                  |
| language | 123   | random seed (GMM)                                                                                  |



`pvtm.topic_words`contains 100 frequent words from the texts which were assingned to single topics. 
`pvtm.wordcloud_df`contains all texts which were assingned to single topics. 

<h2 align="center">Inference</h2>

PVTM allows you to easily estimate the topic distribution for unseen documents using `.infer_topics()`. This methods explicitly calls
`.get_string_vector`(getting a vector from the input text) and `.get_topic_weights`(probability distribution over all topics) consecutively.  

```python
topics = pvtm.infer_topics(new_texts)
pd.Series(topics).plot(kind="bar")
```

which returns:

```text
array([[0., 0., 0., 1., 0.]])
```

<h2 align="center">PVTM Web Viewer</h2>

You can also run the [example](example/reuters_with_dash.py) described above with a dash app extension 
Interactively explore detected topics in the browser. PVTM includes a web app build on dash to visualize results.

```
python ".../path to the example file/reuters_with_dash.py"
```

and view all results in your browser: 

<img src="https://github.com/davidlenz/pvtm/blob/master/img/reuters_dash_demo.gif" width="600" height="400" />



<h2 align="center">Troubleshooting</h2>

If you get the following warning message during model training:

|:warning: **User Warning**: C extension not loaded, training will be slow. Install a C compiler and reinstall gensim for     fast training. "C extension not loaded, training will be slow."|
| --- |

just run:

```
conda install -c conda-forge gensim
```
