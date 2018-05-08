import pandas as pd

import spacy

import datetime as dt
# from spacy import displacy

# from nltk import Tree

# import mongo_helpers as mh


class Archi(object):
    """Archi is a spacy nlp model trained with the following references:
        * 2015 IBC with Washington Amendments
        * ASCE 7-10
    """

    def __init__(self, nlp_model_name):
        """
            ARGS
            ----
            raw_data: dataframe; may need to change to different format when
                      deployed
            trained_data: dataframe; same as raw data
        """

        self._created_date = dt.datetime.today()
        self.raw_data = None  # dataframe
        self.trained_data = None
        self.nlp = spacy.load(nlp_model_name)

    def get_raw_data(self, path):
        """Get raw data from pickle files"""
        df = pd.read_pickle(path)
        df = df.drop(['_id', 'date_read'], axis=1)
        df = df.reset_index()
        df = df.drop('index', axis=1)
        if self.raw_data is None:
            self.raw_data = df
        else:
            self.raw_data = pd.concat([self.raw_data, df], axis=0)

    def fit_nlp(self, nlp):
        """
            ARGS
            ----
            nlp: spacy nlp model object
        """
        # self.trained_data = self.raw_data.copy()
        # self.trained_data = self.add_nlp_doc_col()
        pass

    def add_nlp_doc_col(self):
        """Add column with nlp doc object for code text
        """
        # doc = self.nlp(code_text)
        # return doc
        pass
