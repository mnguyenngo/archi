from flask import Flask, render_template, request, jsonify, Markup
from archi_nlp import Archi
from render_text import render_text
# import pandas as pd
# from pymongo import MongoClient
# import datetime as dt

app = Flask(__name__)

# initiate archi
archi = Archi('en_core_web_lg')
archi.start_mongo()


@app.route('/', methods=['GET'])
def index(results=None):
    # if results=None:
    return render_template('archi-codes.html')
    # else:
    #     return render_template('archi-codes.html', results)


@app.route('/predict', methods=['POST'])
def predict():
    """Returns the nlp prediction given the user's query"""
    user_query = request.json
    # predict returns results and the nlp output of the user query
    results, query_doc = archi.predict(user_query, data_on="mongo")
    rendered_query = render_text(query_doc)
    uq_annotated = render_template('annotate_text.html',
                                   data=Markup(rendered_query))
    table = render_template('cards.html', data=results)
    return jsonify({'user_query': uq_annotated,
                    'table': table})


@app.route('/provision/<variable>', methods=['GET'])
def provision_page(variable):
    """Returns the information for the provision from mongo database"""
    section_num = variable.split('_')

    results = list(archi.mongo_coll.find(
        {'documentInfo.section.section_num.0': {'$eq': section_num[0]}}))
    sub = []

    for prov in results:
        sn = prov['documentInfo']['section']['section_num']

        if len(section_num) > 1 and len(sn) > 1:
            if sn[1] == section_num[1]:
                sub.append(prov)
    if len(sub) > 0:
        results = sub
    return render_template("provision.html", data=results)


def _check_section_num(docInfo, check):
    """Helper function for provision_page()"""
    tup = docInfo['section']['section_num']
    if tup is not None and len(check) == 1:
        return tup[0] == check[0]
    elif tup is not None and len(check) > 1 and len(tup) > 1:
        return tup[0] == check[0] and tup[1] == check[1]
    else:
        return False


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
