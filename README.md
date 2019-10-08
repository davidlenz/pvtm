<h1 align="center">PVTM</h1>
<p align="center">Paragraph Vector Topic Model</p>

**PVTM** relies upon the Paragraph Vector, also known as [Doc2Vec](https://arxiv.org/abs/1405.4053/), to generate document  representations which are then clustered using Gaussian Mixture Models. **PVTM** represents documents in a semantic space to capture latent variables of the underlying documents, e.g. the latent topics. Clusters of documents in the semantic space can then be interpreted and transformed into meaningful topics by means of [Gaussian mixture modeling](https://link.springer.com/referenceworkentry/10.1007%2F978-1-4899-7488-4_196) (GMM). **Paragraph Vectors** expand the ideas of word and phrases representations, [Word2Vec](https://arxiv.org/abs/1310.4546/), to longer pieces of texts. The current version of **PVTM** focuses on Distributed Bag of Words (DBOW) methodology, where according to the ***Word2Vec-Skip-gram*** architecture the the whole document is conditioned on the words appearing in this document.

<h2 align="center">Highlights</h2>

-  :speech_balloon: **Find latent topics in the large text corpora effectively** 
-  :chart_with_upwards_trend: **Detect trends and measure the importance of certain topics** 
-  :bar_chart: **Assign new documents to trained topics** 
-  :telescope: **Apply it in different fields to get answers to various questions** 

<h2 align="center">Install</h2>

Install the module via `pip` command.

```
pip install pvtm 
```

<h2 align="center">Getting Started</h2>
<h3 align="center">Prerequisites: install required packages or will it be done automatically?</h3>
<h3 align="center">Importing & Preprocessing documents</h3>

Once you have installed the **pvtm** module, you can conduct analysis on your text documents stored in a *.txt* or *.csv* file.
The example below considers [reuters dataset](https://keras.io/datasets/#reuters-newswire-topics-classification) from `keras.datasets`.

```python
from keras.datasets import reuters
from sklearn.model_selection import train_test_split
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
texts_train, texts_test = train_test_split(input_texts, test_size=0.2, random_state=42)
```
After that, `PVTM` object should be created with the defined input texts.
Parameter `lemmatized` should be set to `False` when documents' texts should be lemmatized. However, take into account that this step could lead to improved results but also takes some time depending on the size of the document corpus. 
Set the parameter `preprocess=True` when the documents texts should be preprocessed, e.g. removal of special characters, number, currency symbols etc.
With the parameters `min_df` and `max_df` you set the thresholds for very rare/common words which should not be included in the corpus specific vocabulary. Further, you can also exclude language specific stopwords by importing your own stopwords list or using nlkt library as shown below.  

```python
from pvtm.pvtm import PVTM
from pvtm.pvtm import clean
# Load stopwords from nltk library
import nltk
nltk.download("stopwords")
stop_words = list(set(stopwords.words('english')))
# Create PVTM class object
pvtm = PVTM(texts_train, lemmatized = True, min_df = 0.005, max_df = 0.65, stopwords = stop_words)
```
<h2 align="center">Fitting the models</h3>

The next step includes training the Doc2Vec model and clustering of the resulted document vectors by means of GGM. For this, you only need to call the `pvtm.fit()` method and pass all the parameters needed for the Doc2Vec model training and GMM clustering. For more detailed description of the parameter see information provided by [gesim](https://radimrehurek.com/gensim/models/doc2vec.html)(Doc2Vec model) and [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)(GMM).

```python
pvtm.fit(vector_size = 100, # dimensionality of the feature vectors (Doc2Vec)
         hs = 0, # negative sampling will be used for model training (Doc2Vec)
         dbow_words = 1, # simultaneous training of word vectors and document vectors (Doc2Vec)
         dm = 0, # Distributed bag of words (=word2vec-Skip-Gram) (dm=0) OR distributed memory (=word2vec-cbow) (dm=1) (Doc2Vec)
         epochs = 1, # training epochs (Doc2Vec)
         window = 1, # window size (Doc2Vec)
         #seed = 123, # seed for the random number generator (Doc2Vec)
         min_count = 5, # minimal number of appearences for a word to be considered (Doc2Vec)
         workers = 1, # number workers (Doc2Vec)
         alpha = 0.025, # initial learning rate (Doc2Vec)
         min_alpha = 0.025, # doc2vec final learning rate. Learning rate will linearly drop to min_alpha as training progresses.
         #random_state = 123, # random seed (GMM)
         n_components = 5, # number of Gaussian mixture components, i.e. Topics (GMM)
         covariance_type = 'diag', # covariance type (GMM)
         verbose = 1, # verbosity (GMM)
         n_init = 1, # number of initializations (GMM)
         )
```

If you get the following warning message while fitting the model:

|:warning: **User Warning**: C extension not loaded, training will be slow. Install a C compiler and reinstall gensim for     fast training. "C extension not loaded, training will be slow."|
| --- |

just run:

```
conda install -c conda-forge gensim
```

`pvtm.topic_words`contains 100 frequent words from the texts which were assingned to single topics. 
`pvtm.wordcloud_df`contains all texts which were assingned to single topics. 

<h3 align="center">Assignment of new documents to resulted topics</h3>

You can easily assign a new document to the resulted topics by applying `.get_string_vector`(getting a vector from the input text) and `.get_topic_weights`(probability distribution over all topics) methods as shown below:  

```python
new_text = texts_test
new_vector = pvtm.get_string_vector([clean(new_text)])
pvtm.get_topic_weights(new_vector)
```

which returns:

```text
array([[0., 0., 0., 1., 0.]])
```

<h2 align="center">Example with dash app</h2>

You can also run the [example](example/reuters_with_dash.py) described above with a dash app extension 

```
python ".../path to the example file/reuters_with_dash.py"
```

and view all results in your browser: 

<img src="https://github.com/davidlenz/pvtm/blob/master/img/reuters_dash_demo.gif" width="600" height="400" />
