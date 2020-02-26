import os
import pandas as pd

location = os.path.dirname(os.path.realpath(__file__))
my_file = os.path.join(location, 'data', 'sample_5000.csv')

sample_data = pd.read_csv(my_file)
example_texts = sample_data.text.values