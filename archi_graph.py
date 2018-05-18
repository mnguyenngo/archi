# import pandas as pd
# import numpy as np
# import spacy
# import datetime as dt


class Node(object):
    """Node for an AEC knowledge graph
    Args:
        node_type (str): 'provision' or 'component'
        document_info (dict): information about source document (provisions
                              only)
        text_nlp (spacy doc): provision nlp obj (provisions only)
    Returns:
        Node class object
        (use .node to see the json object or python dict)
    """

    def __init__(self, node_type='provision', document_info=None,
                 text_nlp=None, section_nlp=None, chapter_nlp=None,
                 component=None):
        self.node_type = node_type
        self.document_info = document_info
        self.text_nlp = text_nlp
        self.section_nlp = section_nlp
        self.chapter_nlp = chapter_nlp
        self.component = component
        self.ancilliary_nodes = list()
        self.node = self.build_node()

    def build_node(self):
        """Creates node object; Called by __init__
        """
        if self.node_type == 'provision':
            provision_node = ({
                '@context': 'http://archi.codes/',
                '@type': 'provision'})
            if self.document_info is not None:
                provision_node['documentInfo'] = self.document_info
            if self.text_nlp is not None:
                provision_node = self.add_provision_nlp_data(provision_node)
            # returns base node if additional info is None
            return provision_node

        # Placeholder for component node_type
        elif self.node_type == 'component':
            component_node = ({
                '@context': 'http://archi.codes/',
                '@type': 'component',
                'name': self.component})

            return component_node
        else:
            return None

    def add_provision_nlp_data(self, provision_node):
        """Returns parsed provision nlp data
        Args:
            provision_node (dict): base node object
        Returns:
            provision_node (dict): node object with nlp items for the provision
        """
        # provision_node = self.node
        if self.text_nlp is not None:
            if len(self.text_nlp) > 0:
                provision_node["text"] = self.text_nlp.text  # provision text
                first_sent = list(self.text_nlp.sents)[0]
                provision_node["text_nlp_vector"] = (
                    first_sent.vector.tolist())
            else:
                provision_node["text"] = None
                provision_node["text_nlp_vector"] = None
            # parse the nlp obj and extract the root and its objects and
            # subject
            about_base, criteria, about, neg_root = (
                self.parse_nlp_doc(self.text_nlp))
            if about is not None:
                provision_node["about"] = about.text  # primary subject
            provision_node["criteria"] = criteria  # primary criteria
            provision_node["aboutBase"] = about_base  # primary subject lemma
            provision_node["negRoot"] = neg_root  # bool for negative root

            if self.section_nlp is not None:
                provision_node["section_nlp_vector"] = (
                    self.section_nlp.vector.tolist())

            if self.section_nlp is not None:
                provision_node["chapter_nlp_vector"] = (
                    self.chapter_nlp.vector.tolist())
        return provision_node
        # self.node = provision_node

    def parse_nlp_doc(self, doc):
        root = self.get_root(doc, dep='ROOT', lemma=False)
        if root is not None:
            # if root.lemma_ is 'be':
            """Parse nlp_doc and return the following tokens
            Return:
                about_base (lemma): primary subj for categorization,
                ex. wall
                criteria (pobj): object of 'be'
                about (token as read): primary subject, ex. walls
            """
            about_base = self.get_token_by_dep(doc, lemma=True)
            criteria = self.get_criteria(doc, lemma=False)
            about = self.get_token_by_dep(doc, lemma=False)
            neg_root = self.is_root_negative(doc)

            # create nodes for about_base, aka components
            # if node already exists, do nothing
            about_node = Node(node_type='component', component=about_base)
            self.ancilliary_nodes.append(about_node)
            noun_chunk = self.get_nc_with_nsubj(doc, about)
            if noun_chunk is not None:
                type_node = Node(node_type='component',
                                 component=" ".join(noun_chunk[-2:]))
                self.ancilliary_nodes.append(type_node)
            # create edges to link provision to component nodes

            # Placeholder for comply root
            # elif root.lemma_ is 'comply':
            #     print("the root is 'comply'")
            #     about_base, criteria, about = None, None, None

            # else:
            #     about_base = self.get_token_by_dep(doc, lemma=True)
            #     criteria = self.get_criteria(doc, lemma=False)
            #     about = self.get_token_by_dep(doc, lemma=False)
            #     neg_root = self.is_root_negative(doc)
            return about_base, criteria, about, neg_root
        else:
            return None, None, None, None

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
                    return criteria.text
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

    def get_nc_with_nsubj(self, doc, subj):
        chunks = list(doc.noun_chunks)
        if len(chunks) > 0:
            for chunk in chunks:
                if subj in chunk:
                    lemma_chunk = ([word.lemma_ for word in chunk
                                   if word.dep_ not in
                                   ['conj', 'det', 'punct', 'cc']])
                    return lemma_chunk

    def create_edges(self):
        """Create edge object that links to the current base node to a branch
           node.
        """
        doc = self.text_nlp
        if doc is not None:
            base_node_info = self.node['documentInfo']
            sents = list(doc.sents)
            edges = []
            for sent in sents:
                lems = sent.lemma_.split()
                if 'section' in lems:
                    sec_list = self.extract_provision_name(lems,
                                                           keyword='section')
                    for sec_num in sec_list:
                        branch_node = self.package_branch_node(sec_num)
                        e = Edge(base_node_info, branch_node)
                        edges.append(e)
                if 'chapter' in lems:
                    chap_list = self.extract_provision_name(lems,
                                                            keyword='chapter')
                    for chap_num in chap_list:
                        branch_node = self.package_branch_node(
                                      chap_num,
                                      branch_provision_type='chapter')
                        e = Edge(base_node_info, branch_node)
                        e.add_property_related_to()
                        edges.append(e)

            for anc_node in self.ancilliary_nodes:
                # name_of_comp = anc_node.node['name']
                e = Edge(base_node_info, anc_node.node['name'])
                e.add_property_applies_to()
                edges.append(e)

            if len(self.ancilliary_nodes) > 1:
                base_comp_node_name = self.ancilliary_nodes[0].node['name']
                for anc_node in self.ancilliary_nodes[1:]:
                    e = Edge(base_comp_node_name, anc_node.node['name'])
                    e.add_property_type_of()
                    edges.append(e)
            return edges
        else:
            return None
        # insert e into mongodb; return e for now

    def extract_provision_name(self, lemma_list, keyword='section'):
        """Extracts the chapters and sections referenced in a list of lemmas
        Returns:
            prov_names (list): ex. [section_num 16.2, chapter 3]
        """
        prov_names = []
        for idx, word in enumerate(lemma_list):
            if word == keyword:
                if idx != len(lemma_list) - 1:  # ignore the last word
                    next_word = lemma_list[idx+1]
                    ref_num = next_word.split('.')
                    if ref_num[0].isdigit():
                        prov_names.append(tuple(ref_num))
        prov_names = set(prov_names)  # convert to set to remove duplicates
        prov_names = list(prov_names)  # convert back to list
        return prov_names

    def package_branch_node(self, sec_num, branch_provision_type='section'):
        # check if provision node exists or not
        # if exists, update the provision node with the rel edge
        # if does not exist, create new provision node
        source_doc = self.node['documentInfo']['source_doc']
        branch_node = {'source_doc': source_doc}
        chapter = {}

        if branch_provision_type == 'section':
            section = {}
            section['section_num'] = sec_num
            # get chapter to include with section number
            if len(sec_num[0]) > 2:  # for ibc sections and their number format
                actual_chapter = str(int(sec_num[0]) // 100)
                chapter['chapter_num'] = actual_chapter
            else:
                chapter['chapter_num'] = sec_num[0]
            branch_node['chapter'] = chapter
            branch_node['section'] = section
        if branch_provision_type == 'chapter':
            chapter['chapter'] = sec_num[0]
            branch_node['chapter'] = chapter
        return branch_node


class Edge(object):
    """Node for an AEC knowledge graph
    Args:
        node_type (str): 'provision' or 'component'
        document_info (dict): information about source document (provisions
                              only)
        text_nlp (spacy doc): provision nlp obj (provisions only)
    Returns:
    """

    def __init__(self, base_node, branch_node):
        self.base_node = base_node
        self.branch_node = branch_node
        self.edge = self.build_edge()

    def build_edge(self):
        e = {"@context": "http://archi.codes/",  # url
             "@type": "edge"}  # wikidata prop code for 'related to'

        e['base_node'] = self.base_node
        e['branch_node'] = self.branch_node
        return e

    def add_property_related_to(self):
        self.edge['@property'] = 'P1628'

    def add_property_applies_to(self):
        self.edge['@property'] = 'P518'

    def add_property_type_of(self):
        self.edge['@property'] = 'P31'
