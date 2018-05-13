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
            self.nlp = spacy.load('en_core_web_lg')
        else:
            self.nlp = spacy.load(nlp_model_name)

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
        """Save the dataframe with nlp_text to a pickle"""
        self.raw_nlp_data.to_pickle(path)

    def fit_nlp(self):
        """Copies the raw_data and calls add_nlp_text to add an nlp_text column
        """
        self.raw_nlp_data = self.raw_data.copy()
        self.raw_nlp_data['nlp_text'] = (self.raw_nlp_data['code']
                                        .apply(self.add_nlp_text))

    def add_nlp_text(self, code_text):
        """Add column with nlp_text object for code text
        """
        doc = self.nlp(code_text)
        return doc

    def add_keyword_cols(self, predict_df):
        """Add the subj and verb columns to the raw_nlp_data"""
        json_df = pd.DataFrame()
        # copy the title and nlp_text columns to json_df
        json_df['nlp_text'] = predict_df['nlp_text']
        json_df['title'] = predict_df['title']
        json_df['ROOT'] = (predict_df['nlp_text']
                           .apply(lambda x:
                           self.get_root(x, dep='ROOT', lemma=True)))
        json_df['ROOT_TOKEN'] = (predict_df['nlp_text']
                                 .apply(lambda x:
                                 self.get_root(x, dep='ROOT', lemma=False)))
        json_df['SUBJ'] = (predict_df['nlp_text']
                           .apply(lambda x:
                           self.get_token_by_dep(x, dep='nsubj', lemma=True)))
        json_df['SUBJ_TOKEN'] = (predict_df['nlp_text']
                                 .apply(lambda x:
                                 self.get_token_by_dep(x,
                                                       dep='nsubj',
                                                       lemma=False)))
        json_df['CRIT'] = (predict_df['nlp_text']
                           .apply(lambda x:
                           self.get_criteria(x, dep='criteria', lemma=True)))
        json_df['CRIT_TOKEN'] = (predict_df['nlp_text']
                                 .apply(lambda x:
                                 self.get_criteria(x,
                                                   dep='criteria',
                                                   lemma=False)))
        json_df['NEG'] = (predict_df['nlp_text']
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
        top_ten = self.score_df(qdoc).sort_values(ascending=False)[:10]
        top_ten_idx = top_ten.index
        top_ten_df = self.raw_nlp_data.iloc[top_ten_idx]
        top_ten_df_dp = self.add_keyword_cols(top_ten_df)  # dependecy parsed
        # top_ten_df_dp['scores'] = top_ten_df_dp.merge(top_ten, how='left',
        #                                               left_index=True)
        top_ten_df_dp['score'] = top_ten
        top_ten_kg = []  # empty knowledge graph object
        for row in top_ten_df_dp.iterrows():
            top_ten_kg.append(self.build_kg(row))
        return top_ten_kg

    def score_df(self, qdoc):
        """Return a pandas series with the cos_sim scores of the query vs
        the raw nlp docs"""
        scores = self.raw_nlp_data['nlp_text'].apply(
                 lambda x: self.cos_sim(qdoc.vector, x))

        return scores

    def cos_sim(self, query_vec, code_doc):
        """Calculates and returns the cosine similarity value

        Warning:
        For some reason the spaCy result and numpy dot product function does
        not return the same result as the one shown below. With the code below,
        the result falls between 0 and 1, which is expected.
        """
        code_vec = code_doc.vector
        if len(list(code_doc.sents)) > 1:
            code_first_sent = list(code_doc.sents)[0]
            code_vec = code_first_sent.vector
        if len(query_vec) == len(code_vec):
            return (np.sum((query_vec * code_vec))
                    / (np.sqrt(np.sum((query_vec ** 2)))
                       * np.sqrt(np.sum((code_vec ** 2)))))
        else:
            return 0

    def build_kg(self, row):
        provision = {"@context": "http://archi.codes/",  # url
                     "@type": "Provision",  # schema type
                     "title": row[1]['title'],  # provision title
                     "text": row[1]['nlp_text'].text,  # provision text
                     # "nlp_doc": [dict],  # dict with nlp docs for each sent
                     "about": row[1]['SUBJ'],  # subject
                     "criteria": row[1]['CRIT'],
                     "score": row[1]['score']}
        return provision
