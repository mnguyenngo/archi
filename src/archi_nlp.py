import pandas as pd
import numpy as np
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

    def __init__(self, nlp_model_name=None):
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
        if nlp_model_name is None:
            self.nlp = spacy.load('en')
        else:
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

    def clean_newline(self, code_text):
        """Replace newline characters with a space
        Called by get_raw_data
        """
        if '\n' in code_text:
            code_text = code_text.replace('\n', ' ')
        return code_text

    def pickle_raw_nlp(self, path):
        """Save the dataframe with nlp_doc to a pickle"""
        self.raw_nlp_data.to_pickle(path)

    def fit_nlp(self):
        """Copies the raw_data and calls add_nlp_doc to add an nlp_doc column
        """
        self.raw_nlp_data = self.raw_data.copy()
        self.raw_nlp_data['nlp_doc'] = (self.raw_nlp_data['code']
                                        .apply(self.add_nlp_doc))

    def add_nlp_doc(self, code_text):
        """Add column with nlp_doc object for code text
        """
        doc = self.nlp(code_text)
        return doc

    def add_keyword_cols(self, predict_df):
        """Add the subj and verb columns to the raw_nlp_data"""
        json_df = pd.DataFrame()
        json_df['nlp_doc'] = predict_df['nlp_doc']
        json_df['ROOT'] = (predict_df['nlp_doc']
                              .apply(lambda x:
                              self.get_root(x, dep='ROOT', lemma=True)))
        json_df['ROOT_TOKEN'] = (predict_df['nlp_doc']
                                    .apply(lambda x:
                                    self.get_root(x, dep='ROOT', lemma=False)))
        json_df['SUBJ'] = (predict_df['nlp_doc']
                              .apply(lambda x:
                              self.get_token_by_dep(x,
                                                    dep='nsubj', lemma=True)))
        json_df['SUBJ_TOKEN'] = (predict_df['nlp_doc']
                                    .apply(lambda x:
                                    self.get_token_by_dep(x,
                                                          dep='nsubj',
                                                          lemma=False)))
        json_df['CRIT'] = (predict_df['nlp_doc']
                              .apply(lambda x:
                              self.get_criteria(x,
                                                dep='criteria',
                                                lemma=True)))
        json_df['CRIT_TOKEN'] = (predict_df['nlp_doc']
                                    .apply(lambda x:
                                    self.get_criteria(x,
                                                      dep='criteria',
                                                      lemma=False)))
        json_df['NEG'] = (predict_df['nlp_doc']
                             .apply(self.is_root_negative))

        return json_df

    def get_root(self, doc, dep='ROOT', lemma=False):
        """Returns the root of the first sentence of the nlp doc"""
        if len(doc) > 0:
            first_sent = list(doc.sents)[0]
            if dep == 'ROOT':
                if lemma:
                    return first_sent.root.lemma_  # primary verb
                else:
                    return first_sent.root
            else:
                return None
        else:
            return None

    def get_token_by_dep(self, doc, dep='nsubj', lemma=False):
        """Returns the lemmatized token based on the dependency passed
        Only covers the first sentence of each provision as of 5/10.
        """

        if len(doc) > 0:
            # first_sent = list(doc.sents)[0]
            root = self.get_root(doc)
            if dep == 'nsubj':
                deps = ['nsubj', 'nsubjpass']  # nominal subjects
            matches = ([token for token in root.children
                       if token.dep_ in deps])
            if len(matches) > 0:
                token = matches[0]
                if lemma:
                    return token.lemma_
                else:
                    return token
            else:
                return None
        else:
            return None

    def get_criteria(self, doc, dep='criteria', lemma=False):
        """Returns the lemmatized token based on the dependency passed
        Currently, primarily for provisions with ROOT 'be'

        TODO:
        - [ ] Some provisions have multiple criteria. This method only
        retrieves the first occuring criteria.
        """
        if len(doc) > 0:
            if dep == 'criteria':
                # adj modifier (matches with ROOT 'be')
                deps = ['amod', 'prep']
            root = self.get_root(doc)  # dtype: spaCy token

            matches = [token for token in root.children if token.dep_ in deps]
            if len(matches) > 0:
                criteria = matches[0]  # dtype: spaCy token
                if criteria.dep_ == 'prep':
                    pobj = ([token for token in criteria.children
                             if token.dep_ == 'pobj'])
                    if len(pobj) > 0:
                        criteria = pobj[0]  # dtype: spaCy token
                        # match_chunk = []
                        # for chunk in doc.noun_chunks:
                        #     if criteria in chunk:
                        #         match_chunk.append(chunk)
                        # if len(match_chunk) > 0:
                        #     return match_chunk[-1].text
                if lemma:
                    return criteria.lemma_
                else:
                    return criteria
            else:
                return None
        else:
            return None

    def is_root_negative(self, doc):
        """Returns bool if ROOT is negative or not
        Only covers the first sentence of each provision as of 5/10.
        """

        if len(doc) > 0:
            # first_sent = list(doc.sents)[0]
            root = self.get_root(doc)
            matches = ([token for token in root.children
                       if token.dep_ == 'neg'])
            return len(matches) > 0

    def predict(self, query):
        """Returns top ten docs"""
        qdoc = self.nlp(query)
        top_ten = self.score_df(qdoc).sort_values(ascending=False)[:10].index
        top_ten_df = self.raw_nlp_data.iloc[top_ten]
        top_ten_df_json = self.add_keyword_cols(top_ten_df)
        return top_ten_df_json

    def score_df(self, qdoc):
        scores = self.raw_nlp_data['nlp_doc'].apply(
                 lambda x: self.cos_sim(qdoc.vector, x.vector))
        return scores

    def cos_sim(self, a, b):
        if len(a) == len(b):
            return (np.sum((a * b))
                    / (np.sqrt(np.sum((a ** 2))) * np.sqrt(np.sum((b ** 2)))))
        else:
            return 0
