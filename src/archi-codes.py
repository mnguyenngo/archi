from flask import Flask, render_template, request, jsonify, Markup
from archi_nlp import Archi
from render_text import render_text, find_components
# import pandas as pd
# from pymongo import MongoClient
# import datetime as dt

app = Flask(__name__)

# initiate archi
archi = Archi('en_core_web_lg')
archi.start_mongo(db_name='archi', collection_name='newest')


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

    prov_results = render_template('prov_cards.html', data=results)

    """Return related provisions for each component in user query"""
    possible_comps = find_components(query_doc)

    # find provisions for components in the user query
    comp_results = {}
    wiki_results = {}

    for comp in possible_comps:
        prov_edges = list(archi.mongo_coll.find(
            {'@property': 'P518', 'branch_node': comp}))
        if len(prov_edges) > 0:
            comp_results[comp] = prov_edges
        comp_node = list(archi.mongo_coll.find(
            {'@type': 'component', 'name': comp}))
        if len(comp_node) > 0:
            # return only one component, only one is expected to exist in db
            comp_node = comp_node[0]
            wiki_results[comp] = comp_node

    prov_for_comps = render_template('comp_cards.html', data=comp_results)

    rendered_query = render_text(query_doc, wiki_results)
    uq_annotated = render_template('annotate_text.html',
                                   data=Markup(rendered_query))

    return jsonify({'user_query': uq_annotated,
                    'provisions': prov_results,
                    'components': prov_for_comps})


@app.route('/provision/<prov_num>', methods=['GET'])
def provision_page(prov_num):
    """Returns the information for the provision from mongo database"""
    section_num = prov_num.split('_')

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


@app.route('/component/<comp>', methods=['GET'])
def component_page(comp):
    """Returns the information for the provision from mongo database"""
    rel_prov = list(archi.mongo_coll.find(
        {'@property': 'P518', 'branch_node': comp}))

    rel_comp = list(archi.mongo_coll.find(
        {'@property': 'P31', 'base_node': comp}))

    comp_node = list(archi.mongo_coll.find(
        {'@type': 'component', 'name': comp}))
    comp_node = comp_node[0]
    print(comp_node)

    base_type = list(archi.mongo_coll.find(
        {'@property': 'P31', 'branch_node': comp}))

    if len(base_type) > 0:
        base_type = base_type[0]
    else:
        base_type = None

    return render_template("component.html", provisions=rel_prov,
                           comp_name=comp, components=rel_comp,
                           base_type=base_type, comp_node=comp_node)


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
