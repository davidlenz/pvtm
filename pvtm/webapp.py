import argparse
import joblib
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
from pvtm import PVTM, Documents

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

# general
ap.add_argument("-m", "--model", required=True,
                help="path to the trained PVTM model")

parsed_args = ap.parse_args()
args = vars(parsed_args)

data = joblib.load(args['model'])

image_directory = 'Output/'
if not os.path.exists(os.path.dirname(image_directory)):
    try:
        os.makedirs(os.path.dirname(image_directory))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

for i in range(data.gmm.n_components):
    data.wordcloud_by_topic(i).to_file('Output/img_{}.png'.format(i))

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
        marks={i: '{}'.format(1 * i) for i in range(data.gmm.n_components)},
        step=1,
        value=0,
        min=0,
        max= data.gmm.n_components - 1
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
        with open('errors.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')

@app.callback(Output('table', 'children'), [Input('input-value', 'value')])
def display_table(value):
    #df = pd.DataFrame(data.top_topic_center_words.iloc[value,:])
    #df = df.rename(columns={value:'top words for topic {}'.format(value)})
    text = pd.DataFrame(data.model.wv.similar_by_vector(data.cluster_center[value],
                                                        topn=100),
                        columns=['word', "similarity"])
    return generate_table(text, max_rows=15)

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
