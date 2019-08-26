from nltk.corpus import stopwords
import nltk
import pandas as pd
from pandas.tseries.offsets import MonthBegin, QuarterBegin, YearBegin, MonthEnd, Day

def get_stopwords(language="german"):

    numbers = ["zero", "one", "two", "three", "four", "five", "six", "seven",
                    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
                    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
                    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
                    "hundred", "thousand", "million", "billion", "trillion", "quadrillion",
                    "zwölf", "elf", "zehn", "neun", "acht", "sieben", "sechs", "fünf", "vier", "drei", "eins", "zwei",
                    "gajillion", "bazillion", "first", "second", "third", "fourth", "fifth", "sixth", "seventh"]

    other = ['reuter', '\x03', "mehr", "wurden", "konnen", "wurde","jedoch", "seien", "sei", "dass", "zwei", "weitere",
             "beim","allerdings","wegen","könne", "daher","zudem","schon", "denen", "dafür","geht","ging","bisher", "gibt", "bereit",
             "kommen", "lassen", "dabei", "schon", "bereits", "bislang", "erst", "wenig", "allerdings",  "außerdem", "lediglich", "etwa",
                "lässt", "ab", "euro", "dürfen", "geben", "gehen", "sollen", "sollen", "sollen", "gut", "derzeit", ]

    stop_words = set(nltk.corpus.stopwords.words(language) + numbers + other)
    return stop_words


def load_heise_data(nrows):
    print("Load heise data..")
    df = pd.read_csv("data/heise_archiv_lemmatized.csv", index_col="Unnamed: 0",  nrows=nrows)
    #df = df.sample(frac=frac, random_state=rndstate)
    print("Data:", df.shape)
    # add time information to aggregate on

    df['month'] = pd.to_datetime(df['date']).dt.date + MonthEnd(n=0) - MonthBegin(n=1)
    df['quarter'] = pd.to_datetime(df['date']).dt.date - QuarterBegin(n=1)
    df['year'] = pd.to_datetime(df['date']).dt.date - YearBegin(n=1)
    return df