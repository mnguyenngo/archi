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

    def random_sample(self, n=150):
        """Returns a random row from the nlp_raw_data"""
        df = self.raw_nlp_data.sample(n=n,
                                      replace=False)
        df = df.reset_index()
        df = df.drop('index', axis=1)
        return df

    def build_ner_train(self):
        """Initializes ner train data"""
        if self.raw_ner_data is None:
            self.raw_ner_data = self.random_sample()

    def review_ner_train(self):
        """Serves one ner object for user to review"""
        if len(self.raw_ner_data) == 0:
            print("No raw ner data left. Train the model with archi.train()")
            return None

        else:
            last_idx = len(self.raw_ner_data) - 1
            ner_data = self.raw_ner_data.loc[last_idx]
            self.raw_ner_data = self.raw_ner_data.drop(last_idx, axis=0)
            ent_data, ent_w_word = self.format_ner_data(ner_data['nlp_doc'])
            ner_data = NER_data(ent_data, ent_w_word)
            return ner_data

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
                                  ent.label_) for ent in doc.ents]}
        ent_formatted = {'entities': [(ent.text,
                                       ent.start,
                                       ent.end,
                                       ent.label_) for ent in doc.ents]}
        train_obj = (text, ent_dict)
        view_obj = (text, ent_formatted)
        return train_obj, view_obj

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
    def __init__(self, train_obj, view_obj):
        self.ent_data = train_obj
        self.view_data = view_obj
        # self.text = ent_data[0]
        # self.ent_dict = ent_data[1]

    def add_ent(self, text, label):
        if text in self.ent_data[0]:
            # if text is multiple words, extra check to see if index is correct
            if len(text.split()) > 1:
                check_1 = self.ent_data[0].split().index(text.split()[0])
                check_2 = self.ent_data[0].split().index(text.split()[1])
                if check_2 == check_1 + 1:
                    start = self.ent_data[0].split().index(text.split()[0])
                else:
                    raise ValueError
            else:
                start = self.ent_data[0].split().index(text)

            end = start + len(text.split())

            self._add_ent(text, start, end, label)

            print(self.view_data)
        else:
            raise ValueError

    def _add_ent(self, text, start, end, label):
        for attr in [self.ent_data, self.view_data]:
            if attr == self.ent_data:
                # add to ent_data
                attr[1]['entities'].append((start, end, label))
            else:
                # add to view_data
                attr[1]['entities'].append((text, start, end, label))
            attr[1]['entities'] = list(set(attr[1]['entities']))
            if attr == self.ent_data:
                attr[1]['entities'].sort()
            else:
                attr[1]['entities'].sort(key=lambda x: x[1])

    def modify_ent(self, text, start=None, end=None, label=None):
        for tup in self.view_data[1]['entities']:
            if text in tup:
                old_start = tup[1]
                old_end = tup[2]
                old_label = tup[3]

                # if start, end, label is not given by user,
                # set to existing value
                # for idx, attr in enumerate([start, end, label]):
                #     old_attr = [old_start, old_end, old_label]
                #     if attr is None:
                #         attr = old_attr[idx]

                if start is None:
                    start = old_start
                if end is None:
                    end = old_end
                if label is None:
                    label = old_label

                self._del_ent(text, old_start, old_end, old_label)
                self._add_ent(text, start, end, label)
        print(self.view_data)

    def del_ent(self, text):
        for tup in self.view_data[1]['entities']:
            if text in tup:
                start = tup[1]
                end = tup[2]
                label = tup[3]

                self._del_ent(text, start, end, label)
                print(self.view_data)
            else:
                raise ValueError

    def _del_ent(self, text, start, end, label):
        for attr in [self.ent_data, self.view_data]:
            if attr == self.ent_data:
                # add to ent_data
                attr[1]['entities'].remove((start, end, label))
            else:
                # add to view_data
                attr[1]['entities'].remove((text, start, end, label))
            # attr[1]['entities'] = list(set(attr[1]['entities']))
            # attr[1]['entities'].sort()
