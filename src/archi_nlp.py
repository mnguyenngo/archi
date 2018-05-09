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

    def random_sample(self, n=200):
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
        """Serves one NER object for user to review"""
        if len(self.raw_ner_data) == 0:
            print("No raw NER data left. Train the model with archi.train()")
            return None

        else:
            last_idx = len(self.raw_ner_data) - 1
            ner_data = self.raw_ner_data.loc[last_idx]
            self.raw_ner_data = self.raw_ner_data.drop(last_idx, axis=0)
            ent_data, ent_w_word = self.format_ner_data(ner_data['nlp_doc'])
            ner_data = NER_data(ent_data, ent_w_word, ner_data['nlp_doc'])
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

    def get_ner_train_data(self, path):
        """Get raw nlp data from pickle files"""
        df = pd.read_pickle(path)
        # df['code'] = df['code'].apply(self.clean_newline)
        if self.ner_train_data is None:
            self.ner_train_data = df
        else:
            self.ner_train_data = pd.concat([self.ner_train_data, df],
                                            axis=0,
                                            ignore_index=True)

    def submit_ner_train_data(self, ner_data):
        """Collects NER data for model training"""
        ent = ner_data.ent_data
        ent_dict = {'string': ent[0], 'ent': ent[1]}
        ent_df = pd.DataFrame(ent_dict)
        ent_df = ent_df.reset_index(drop=True)
        if self.ner_train_data is None:
            self.ner_train_data = ent_df
        else:
            self.ner_train_data = pd.concat([self.ner_train_data, ent_df],
                                            axis=0,
                                            ignore_index=True)

        path = 'data/ner/{}_ner.pkl'.format(self._created_date
                                            .strftime("%y-%m-%d"))
        self.ner_train_data.to_pickle(path)

    def ner_train(self, min_train_size=10):
        if len(self.ner_train_data) > min_train_size:
            pass
        pass


class NER_data(object):
    """Data for training a spacy nlp model in labeling named entities

    Attributes:
        train_obj (tuple(str, dict)): in the format for training according to
        spacy docs
        view_obj (tuple(str, dict)): for reviewer, includes text that is being
        labeled

    Returns:
        NER_data object
    """

    def __init__(self, train_obj, view_obj, doc):
        self.ent_data = train_obj
        self.view_data = view_obj
        self.doc = doc
        # self.text = ent_data[0]
        # self.ent_dict = ent_data[1]

    def add_ent(self, text, label):
        """Locates the text and applies the label to the text"""
        if text in self.ent_data[0]:
            # if text is multiple words, extra check to see if index is correct
            # start = 0  # init start
            if len(text.split()) > 1:
                check_1 = self.ent_data[0].split().index(text.split()[0])
                check_2 = self.ent_data[0].split().index(text.split()[1])
                if check_2 == check_1 + 1:
                    print("Warning: check if index is correct.")
                start = self.ent_data[0].split().index(text.split()[0])
                # else:
                    # raise ValueError
            else:
                start = self.ent_data[0].split().index(text)

            end = start + len(text.split())

            self._add_ent(text, start, end, label)

            print(self.view_data)
        else:
            raise ValueError

    def _add_ent(self, text, start, end, label):
        """Helper function for add_ent()"""
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

    def modify_label(self, text, label=None):
        """Locates the text and modifies the entity data"""
        for tup in self.view_data[1]['entities']:
            if text in tup:
                self.del_ent(text)
                self.add_ent(text, label)
        # print(self.view_data)

    def del_ent(self, text):
        """Locates the text in a doc and deletes entity label"""
        for tup in self.view_data[1]['entities']:
            if text in tup:
                start = tup[1]
                end = tup[2]
                label = tup[3]

                self._del_ent(text, start, end, label)
                print(self.view_data)
            # else:
                # raise ValueError

    def _del_ent(self, text, start, end, label):
        """Helper function for del_ent()"""
        for attr in [self.ent_data, self.view_data]:
            if attr == self.ent_data:
                # add to ent_data
                attr[1]['entities'].remove((start, end, label))
            else:
                # add to view_data
                attr[1]['entities'].remove((text, start, end, label))
            # attr[1]['entities'] = list(set(attr[1]['entities']))
            # attr[1]['entities'].sort()
