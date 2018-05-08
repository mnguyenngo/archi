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
        self.raw_data = None
        self.raw_nlp_data = None
        self.nlp = spacy.load(nlp_model_name)
        self.trained_ids = []

    def get_raw_data(self, path):
        """Get raw data from pickle files"""
        df = pd.read_pickle(path)
        df = df.drop(['date_read'], axis=1)
        df = df.reset_index()
        df = df.drop('index', axis=1)
        if self.raw_data is None:
            self.raw_data = df
        else:
            self.raw_data = pd.concat([self.raw_data, df],
                                      axis=0,
                                      ignore_index=True)

    def get_raw_nlp_data(self, path):
        """Get raw data from pickle files"""
        df = pd.read_pickle(path)
        if self.raw_nlp_data is None:
            self.raw_nlp_data = df
        else:
            self.raw_nlp_data = pd.concat([self.raw_nlp_data, df],
                                          axis=0,
                                          ignore_index=True)

    def pickle_raw_nlp(self, path):
        """Save the dataframe with nlp_doc to a pickle"""
        self.raw_nlp_data.to_pickle(path)

    def fit_nlp(self):
        """
            ARGS
            ----
            nlp: spacy nlp model object
        """
        self.raw_nlp_data = self.raw_data.copy()
        self.raw_nlp_data['nlp_doc'] = (self.raw_nlp_data['code']
                                        .apply(lambda x: self.add_nlp_doc(x)))

    def add_nlp_doc(self, code_text):
        """Add column with nlp doc object for code text
        """
        doc = self.nlp(code_text)
        return doc

    def random_sample(self, seed=290):
        """Returns a random row from the nlp_raw_data"""
        return pd.self.raw_nlp_data.sample(n=1,
                                           replace=False,
                                           random_state=seed)
