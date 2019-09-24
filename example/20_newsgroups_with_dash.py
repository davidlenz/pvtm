from pvtm.pvtm import PVTM
from pvtm.pvtm import clean
from sklearn.datasets import fetch_20newsgroups
import nltk
import argparse
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pybase64
import os
import errno
import glob
import flask
from dash.dependencies import Input, Output

newsgroups_train = fetch_20newsgroups(subset='train')
input_texts = newsgroups_train.data
print('There are', len(newsgroups_train.data), 'documents.')

nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = list(set(stopwords.words('english')))
len_texts = 500

# Create PVTM class object
pvtm = PVTM(input_texts[:len_texts], lemmatized = True, min_df = 0.0005, max_df = 0.65)
pvtm.fit(vector_size=100,
         hs=0,
         dbow_words=1, # train word vectors!
         dm=0, # Distributed bag of words (=word2vec-Skip-Gram) (dm=0) OR distributed memory (=word2vec-cbow) (dm=1)
         epochs=1, # doc2vec training epochs
         window=1, # doc2vec window size
         #seed=123, # doc3vec random seed
         #random_state = 123, # gmm random seed
         min_count=5, # minimal number of appearences for a word to be considered
         workers=1, # doc2vec num workers
         alpha=0.025, # doc2vec initial learning rate
         min_alpha=0.025, # doc2vec final learning rate. Learning rate will linearly drop to min_alpha as training progresses.
         n_components=10, # number of Gaussian mixture components, i.e. Topics
         covariance_type='diag', # gmm covariance type
         verbose=1, # verbosity
         n_init=1, # gmm number of initializations

         )
print("Finished fitting")
# You can get document vectors from the Doc2Vec model by calling the command
document_vectors = np.array(pvtm.model.docvecs.vectors_docs)

# You can get distribution of the document over the defined topics by calling the command
# Thereby, each row is a single document, each column is one topic. The entries within the matrix are probabilities.
document_topics = np.array(pvtm.gmm.predict_proba(np.array(pvtm.model.docvecs.vectors_docs)))

newsgroups_test = fetch_20newsgroups(subset='test')
new_vector = pvtm.get_string_vector([clean(newsgroups_test.data[0])])
print(pvtm.get_topic_weights(new_vector))



image_directory = 'Output/'
if not os.path.exists(os.path.dirname(image_directory)):
    try:
        os.makedirs(os.path.dirname(image_directory))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

for i in range(pvtm.gmm.n_components):
    pvtm.create_wordcloud_by_topic(i).to_file('Output/img_{}.png'.format(i))

def generate_table(dataframe, max_rows=20):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))],
        style = {'overflow': 'scroll', "background-color": "powderblue","width":"100%",
               'border': '1px solid black','border-collapse': 'collapse',
               'text-align': 'middle', 'border-spacing': '5px', 'font-size' : '20px'}
       )


app = dash.Dash()
app.scripts.config.serve_locally = True

app.layout = html.Div(children=[
    html.H1('PVTM Results', style={'textAlign': 'center', 'background-color': '#7FDBFF'}),
    dcc.Slider(
        id='input-value',
        marks={i: '{}'.format(1 * i) for i in range(10)},
        step=1,
        value=0,
        min=0,
        max=10-1
    ),
    html.Div(children=[
        html.Div([
            html.H3('Word Cloud', style={'textAlign': 'center', 'color': '#1C4E80'}),
            html.Img(id='wordcloud')
        ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top','textAlign': 'center'}, className="six columns"),
        html.Div([
            html.H3('Important Words', style={'textAlign': 'center', 'color': '#1C4E80'}),
            html.Div(id='table')
        ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'middle','textAlign': 'center'}, className='six columns')
    ], className="row")
])

@app.callback(Output(component_id='wordcloud', component_property='src'),
              [Input(component_id='input-value', component_property='value')]
              )
def update_img(value):
    try:
        image_filename = image_directory + 'img_{}.png'.format(
            value)
        encoded_image = pybase64.b64encode(open(image_filename, 'rb').read())#.decode('ascii')
        return 'data:image/png;base64,{}'.format(encoded_image.decode('ascii'))
    except Exception as e:
        with open(args['input'] + '/errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')

@app.callback(Output('table', 'children'), [Input('input-value', 'value')])
def display_table(value):
    #df = pd.DataFrame(pvtm.top_topic_center_words.iloc[value,:])
    #df = df.rename(columns={value:'top words for topic {}'.format(value)})
    text = pd.DataFrame(pvtm.model.wv.similar_by_vector(pvtm.cluster_center[value],
                                                        topn=100),
                        columns=['word', "similarity"])
    return generate_table(text, max_rows=15)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
