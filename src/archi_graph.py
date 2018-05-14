# import pandas as pd
# import numpy as np
# import spacy
# import datetime as dt


class ArchiNode(object):
    def __init__(self, node_type='provision', doc_info=None, nlp_obj=None):
        self.node_type = node_type
        self.doc_info = doc_info
        self.nlp_obj = nlp_obj

    def build_node(self):
        if self.node_type == 'provision':
            provision_node = ({
                '@context': 'http://archi.codes/',
                '@type': 'provision',
                'docInfo': self.doc_info})

            provision_node = self.add_provision_data(provision_node)
            return provision_node

    def add_provision_data(self, provision_node):
        provision_node["text"] = self.doc_info,  # provision text; nlp doc
        about_base, criteria, about = self.parse_nlp_doc(self.doc_info)
        provision_node["about"] = about  # primary subject
        provision_node["criteria"] = criteria  # primary criteria
        provision_node["aboutBase"] = about_base  # primary subject lemma
        return provision_node

    def parse_nlp_doc(self, doc):
        root = self.get_root(doc, dep='ROOT', lemma=False)
        if root.text is 'be':
            print('the root is be')
            about_base, criteria, about = None, None, None
            return about_base, criteria, about

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
