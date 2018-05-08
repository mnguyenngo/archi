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
        self.raw_ner_data = None
        self.ner_train_data = None

    def get_raw_data(self, path):
        """Get raw data from pickle files"""
        df = pd.read_pickle(path)
        df = df.drop(['date_read'], axis=1)
        df = df.reset_index()
        df = df.drop('index', axis=1)
        df['code'] = df['code'].apply(self.clean_newline)
        if self.raw_data is None:
            self.raw_data = df
        else:
            self.raw_data = pd.concat([self.raw_data, df],
                                      axis=0,
                                      ignore_index=True)

    def get_raw_nlp_data(self, path):
        """Get raw nlp data from pickle files"""
        df = pd.read_pickle(path)
        df['code'] = df['code'].apply(self.clean_newline)
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
                                        .apply(self.add_nlp_doc))

    def add_nlp_doc(self, code_text):
        """Add column with nlp doc object for code text
        """
        doc = self.nlp(code_text)
        return doc

    def random_sample(self, n=150, seed=290):
        """Returns a random row from the nlp_raw_data"""
        df = self.raw_nlp_data.sample(n=n,
                                      replace=False,
                                      random_state=seed)
        df = df.reset_index()
        df = df.drop('index', axis=1)
        return df

    def prep_ner_train(self):
        if self.raw_ner_data is None:
            self.raw_ner_data = self.random_sample()

        if len(self.raw_ner_data) == 0:
            print("No raw ner data left. Train the model with archi.train()")
            return None

        else:
            last_idx = len(self.raw_ner_data) - 1
            ner_data = self.raw_ner_data.loc[last_idx]
            self.raw_ner_data = self.raw_ner_data.drop(last_idx, axis=0)
            train_obj = self.format_ner_data(ner_data['nlp_doc'])
            return train_obj

    def format_ner_data(self, doc):
        """Serve reviewer a row from the random_sample of the data.
            Reviewer will make modifications to the entity results and send
            the object to the train dataset.

            Called by prep_ner_train()
        """
        # doc = row['nlp_doc'].values[0]
        text = doc.text
        ent_dict = {'entities': [(ent.start,
                                  ent.end,
                                  ent.label_) for ent in doc.ents],
                    'ent_word': [(ent.text,
                                  ent.start,
                                  ent.end,
                                  ent.label_) for ent in doc.ents]}
        train_obj = (text, ent_dict)
        return train_obj

    def submit_ner_train_data(self, ner_data):
        if self.ner_train_data is None:
            self.ner_train_data = [ner_data]
        else:
            self.ner_train_data.append(ner_data)

    def clean_newline(self, code_text):
        """Replace newline characters with a space"""
        if '\n' in code_text:
            code_text = code_text.replace('\n', ' ')
        return code_text


class NER_data(object):
    def __init__(self, ent_data):
        self.ent_data = ent_data
        # self.text = ent_data[0]
        # self.ent_dict = ent_data[1]

    def add_ent(self, idx, text, label, start, end):
        pass

    def del_ent(self, idx):
        pass

    def modify_ent(self, idx, text, label, start, end):
        pass
