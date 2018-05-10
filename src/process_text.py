import re
import pandas as pd
from pymongo import MongoClient
import mongo_helpers as mh


def get_collection_names(db_name, host='localhost', address=27017):
    """Returns a list of the names of the collections at the specified database.

        Args:
            db_name (str): name of database
            host (str): 'localhost' (default) for mongodb on local computer
            address (int): '27017' (default) for mongodb on local computer

        Returns:
            list of collections names
    """
    client = MongoClient(host, address)
    db = client[db_name]
    return db.collection_names()


def db_to_pickle(db_name):
    """Converts mongo collection to df to pkl"""
    colls = get_collection_names(db_name)
    if 'raw_html' in colls:
        colls.remove('raw_html')
    for idx, chapter in enumerate(colls):
        if idx == 0:
            db_first = mh.convert_db_to_df(db_name, chapter)
        else:
            db_add = mh.convert_db_to_df(db_name, chapter)
            db_first = pd.concat([db_first, db_add], axis=0)

    path_name = 'data/df/{}.pkl'.format(db_name)
    db_first.to_pickle(path_name)
    print(path_name)
    # return db_first


"""
source of code:
https://stackoverflow.com/questions/4576077/python-split-text-on-sentences
modified from source
"""
# for split_into_sentences
caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms+" "+starters, "\\1<stop> \\2", text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(caps + "[.]" + caps + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" "+suffixes+"[.] "+starters, " \\1<stop> \\2", text)
    text = re.sub(" "+suffixes+"[.]", " \\1<prd>", text)
    text = re.sub(" " + caps + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    if '\"' in text:
        text = text.replace("\"", " \" ")
    text = text.replace(". ", ".<stop> ")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
