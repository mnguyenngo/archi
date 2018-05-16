import pandas as pd
import numpy as np
import spacy
from pymongo import MongoClient
import datetime as dt

from src.archi_graph import Node


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
        self.nlp_data = None
        if nlp_model_name is None:
            self.nlp = spacy.load('en_core_web_lg')
        else:
            self.nlp = spacy.load(nlp_model_name)

    def get_raw_data(self, path, default_process=True):
        """Get raw data from pickle files"""
        df = pd.read_pickle(path)
        df = df.drop(['date_read'], axis=1)
        df = df.reset_index(drop=True)
        df['code'] = df['code'].apply(self.clean_newline)
        # get source_doc and add as column
        df['source_doc'] = path
        df['source_doc'] = df['source_doc'].apply(self.get_source_doc)
        if self.raw_data is None:
            self.raw_data = df
        else:
            self.raw_data = pd.concat([self.raw_data, df],
                                      axis=0,
                                      ignore_index=True)
        if default_process is True:
            self.get_doc_data()
            self.fill_chapter_title()

    def get_source_doc(self, path):
        if 'ibc' in path.split('/')[-1]:
            source_doc = {'title': 'International Building Code',
                          'edition': 2015,
                          'withAmendedments': 'Washington'}

        elif 'asce' in path.split('/')[-1]:
            source_doc = {'title': 'ASCE 7-10'}
        else:
            source_doc = None
        return source_doc

    def get_doc_data(self, on='raw'):
        if on == 'raw':
            df = self.raw_data.copy()
            df = df.apply(self.parse_title, axis=1)
            self.raw_data = df
        if on == 'raw_nlp':
            df = self.nlp_data.copy()
            df = df.apply(self.parse_title, axis=1)
            self.nlp_data = df

    def parse_title(self, row):
        """Parse through section title and extract num and text"""
        title = row['title']
        source_doc_title = row['source_doc']['title']

        # parse_title
        title_list = title.split()
        if title_list[0].isalpha():
            if title_list[0] == 'Chapter':  # if the row is for a chapter
                section_num = None
                section_title = None
                chapter_title = " ".join(title_list[2:])
                chapter_num = title_list[1]
            # if the row is for a section title
            elif title_list[0] == 'Section':
                section_num = tuple(title_list[1].split('.'))
                section_title = " ".join(title_list[2:])
                if source_doc_title == 'International Building Code':
                    chapter_num = section_num[0]
                    if chapter_num.isdigit():
                        chapter_num = int(chapter_num)
                        chapter_num = chapter_num // 100
                        chapter_num = str(chapter_num)
                    elif chapter_num[0].isalpha():
                        chapter_num = chapter_num[0]
                else:
                    chapter_num = section_num[0]
                chapter_title = None

            elif title_list[0] == 'Appendix':  # if the row is for a chapter
                section_num = None
                section_title = None
                chapter_title = " ".join(title_list[2:])
                chapter_num = title_list[1]

            else:  # for unexpected conditions, just fill with Nones
                section_num = None
                section_title = None
                chapter_title = None
                chapter_num = None
        else:  # title is neither chapter or section title
            section_num = tuple(title_list[0].split('.'))
            section_title = " ".join(title_list[1:])
            chapter_num = None
            chapter_title = None

            # deal with IBC section number conventions
            if source_doc_title == 'International Building Code':
                chapter_num = section_num[0]
                if chapter_num.isdigit():
                    chapter_num = int(chapter_num)
                    chapter_num = chapter_num // 100
                    chapter_num = str(chapter_num)
                else:
                    chapter_num = chapter_num[0]
            else:
                chapter_num = section_num[0]
                if chapter_num[0].isalpha():
                    chapter_num = chapter_num[0]
                chapter_title = None

        # add title info to dataframe
        row['section_num'] = section_num
        row['section_title'] = section_title
        row['chapter_num'] = chapter_num
        row['chapter_title'] = chapter_title
        return row

    def fill_chapter_title(self):
        """Iterates through"""
        df = self.raw_data.copy()
        chapters = df[df['section_title'].isnull()]
        for row in chapters.iterrows():
            chapter_num = row[1]['chapter_num']
            chapter_title = row[1]['chapter_title']
            source_title = row[1]['source_doc']['title']
            mask = df['chapter_num'] == chapter_num
            for df_row in df[mask].iterrows():
                if df_row[1]['source_doc']['title'] == source_title:
                    df.loc[df_row[0], 'chapter_title'] = chapter_title
        self.raw_data = df

    def get_nlp_data(self, path):
        """Get raw nlp data from pickle files"""
        df = pd.read_pickle(path)
        df['code'] = df['code'].apply(self.clean_newline)
        if self.nlp_data is None:
            self.nlp_data = df
        else:
            self.nlp_data = pd.concat([self.nlp_data, df],
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
        self.nlp_data.to_pickle(path)

    def fit_nlp(self):
        """Copies the raw_data and calls add_nlp_text to add an nlp_text column
        """
        self.nlp_data = self.raw_data.copy()
        self.nlp_data['nlp_code_text'] = (self.nlp_data['code']
                                          .apply(self.add_nlp_text))
        self.nlp_data['nlp_section_title'] = (self.nlp_data['section_title']
                                              .apply(self.add_nlp_text))
        self.nlp_data['nlp_chapter_title'] = (self.nlp_data['chapter_title']
                                              .apply(self.add_nlp_text))

    def add_nlp_text(self, code_text):
        """Add column with nlp_text object for code text
        """
        if code_text is None:
            return None
        else:
            doc = self.nlp(code_text)
            return doc

    # def add_keyword_cols(self, predict_df):
    #     """Add the subj and verb columns to the nlp_data"""
    #     json_df = pd.DataFrame()
    #     # copy the title and nlp_text columns to json_df
    #     json_df['nlp_text'] = predict_df['nlp_text']
    #     json_df['title'] = predict_df['title']
    #     json_df['ROOT'] = (predict_df['nlp_text']
    #                        .apply(lambda x:
    #                        self.get_root(x, dep='ROOT', lemma=True)))
    #     json_df['ROOT_TOKEN'] = (predict_df['nlp_text']
    #                              .apply(lambda x:
    #                              self.get_root(x, dep='ROOT', lemma=False)))
    #     json_df['SUBJ'] = (predict_df['nlp_text']
    #                        .apply(lambda x:
    #                        self.get_token_by_dep(x, dep='nsubj', lemma=True)))
    #     json_df['SUBJ_TOKEN'] = (predict_df['nlp_text']
    #                              .apply(lambda x:
    #                              self.get_token_by_dep(x,
    #                                                    dep='nsubj',
    #                                                    lemma=False)))
    #     json_df['CRITICAL'] = (predict_df['nlp_text']
    #                            .apply(lambda x:
    #                            self.get_criteria(x,
    #                                              dep='criteria',
    #                                              lemma=True)))
    #     json_df['CRITICAL_TOKEN'] = (predict_df['nlp_text']
    #                                  .apply(lambda x:
    #                                  self.get_criteria(x,
    #                                                    dep='criteria',
    #                                                    lemma=False)))
    #     json_df['NEG'] = (predict_df['nlp_text']
    #                       .apply(self.is_root_negative))
    #
    #     return json_df

    def predict(self, query):
        """Returns top ten docs"""
        qdoc = self.nlp(query)
        top_ten = self.score_df(qdoc).sort_values(ascending=False)[:10]
        top_ten_idx = top_ten.index
        top_ten_df = self.nlp_data.iloc[top_ten_idx]
        # top_ten_df_dp = self.add_keyword_cols(top_ten_df)  # dependecy parsed
        # top_ten_df_dp['scores'] = top_ten_df_dp.merge(top_ten, how='left',
        #                                               left_index=True)
        top_ten_df['score'] = top_ten
        # top_ten_kg = []  # empty knowledge graph object
        # for row in top_ten_df.iterrows():
        #     top_ten_kg.append(self.build_kg(row))
        return top_ten_df

    def score_df(self, qdoc):
        """Return a pandas series with the cos_sim scores of the query vs
        the raw nlp docs"""
        code_text_scores = self.nlp_data['nlp_code_text'].apply(
                           lambda x: self.cos_sim(qdoc.vector, x))
        sec_title_scores = self.nlp_data['nlp_section_title'].apply(
                           lambda x: self.cos_sim(qdoc.vector, x))
        chap_title_scores = self.nlp_data['nlp_chapter_title'].apply(
                            lambda x: self.cos_sim(qdoc.vector, x))
        scores = pd.concat([code_text_scores, sec_title_scores, chap_title_scores], axis=1)
        scores['total'] = scores.mean(axis=1)
        # print(scores)
        return scores['total']

    def cos_sim(self, query_vec, code_doc):
        """Calculates and returns the cosine similarity value

        Warning:
        For some reason the spaCy result and numpy dot product function does
        not return the same result as the one shown below. With the code below,
        the result falls between 0 and 1, which is expected.
        """
        if code_doc is not None:
            if len(code_doc) > 0:
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
        else:
            return 0

    def build_node(self, row):
        document_info = self.package_document_info(row)
        text_nlp = row['nlp_code_text']
        section_nlp = row['nlp_section_title']
        chapter_nlp = row['nlp_chapter_title']
        node = Node(node_type='provision', document_info=document_info,
                    text_nlp=text_nlp, section_nlp=section_nlp,
                    chapter_nlp=chapter_nlp)
        return node

    def package_document_info(self, row):
        """Packages the data in a dataframe row into a python dictionary"""
        chapter = {'chapter_num': row['chapter_num'],
                   'chapter_title': row['chapter_title']}
        section = {'section_num': row['section_num'],
                   'section_title': row['section_title']}
        source_doc = row['source_doc']
        document_info = {'chapter': chapter,
                         'section': section,
                         'source_doc': source_doc}
        return document_info

    def find_edges(self, node):
        edges = node.create_edges()
        return edges

    def build_db(self, coll_name=None):
        # if coll_name is None, create new db
        if coll_name is None:
            todays_date = self._created_date.strftime('%y%m%d')
            coll_name = f"archi_{todays_date}"
            print(coll_name)
        self.nlp_data.apply(lambda x: self.build_db_pipeline(x, coll_name),
                            axis=1)

    def build_db_pipeline(self, row, coll_name):
        client = MongoClient()
        db = client['archi']
        coll = db[coll_name]

        node = self.build_node(row)  # dtype: node object
        edges = node.create_edges()  # dtype: list of edge objects
        if type(node.node) == dict:
            print('foonode')
            coll.insert_one(node.node)
        if edges is not None:
            for edge in edges:
                if type(edge.edge) == dict:
                    print('fooedge')
                    coll.insert_one(edge.edge)
