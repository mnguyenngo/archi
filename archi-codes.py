from flask import Flask, render_template, request, jsonify, Markup
from archi_nlp import Archi
from render_text import render_text
# import datetime as dts

app = Flask(__name__)

# initiate archi
archi = Archi('en_core_web_lg')

# load nlp data if available
archi.get_nlp_data('data/nlp_df/nlp_0514.pkl')

# otherwise, load raw data
# archi.get_raw_data('data/raw_df/ibc.pkl')
# archi.get_raw_data('data/raw_df/asce7.pkl')

# and fit with archi nlp model
# archi.fit_nlp()

# and pickle the nlp data
# today = dt.datetime.today().strftime('%y%m%d')
# archi.pickle_raw_nlp(f'data/nlp_df/nlp_{today}.pkl')

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
    results, query_doc = archi.predict(user_query)
    data = [(result[1]['nlp_chapter_title'].text,  # chapter title text
             result[1]['source_doc']['title'],  # book title
             result[1]['title'],  # section title text
             result[1]['nlp_code_text'].text,  # provision text
             # score value rounded to the two decimal points
             round(result[1]['score'], 2),
             # url for provision section
             "_".join(result[1]['section_num'][:2])) for result in results.iterrows()]
    rendered_query = render_text(query_doc)
    uq_annotated = render_template('annotate_text.html',
                                   data=Markup(rendered_query))
    table = render_template('cards.html', data=data)
    return jsonify({'user_query': uq_annotated,
                    'table': table})


@app.route('/provision/<variable>', methods=['GET'])
def provision_page(variable):
    """Returns the information for the provision"""
    section_num = variable.split('_')
    results = (archi.nlp_data.loc[archi.nlp_data['section_num']
               .apply(lambda x: _check_section_num(x, section_num))])
    data = [(result[1]['nlp_chapter_title'].text,  # chapter title text
             result[1]['source_doc']['title'],  # book title
             result[1]['title'],  # section title text
             result[1]['nlp_code_text'].text,  # provision text
             # url for provision section
             "_".join(result[1]['section_num'][:2])) for result in results.iterrows()]
    chapter = data[0][0]
    source_doc = data[0][1]
    return render_template("provision.html", data=data, chapter=chapter,
                           source_doc=source_doc)


def _check_section_num(tup, check):
    """Helper function for provision_page()"""
    if tup is not None and len(check) == 1:
        return tup[0] == check[0]
    elif tup is not None and len(check) > 1 and len(tup) > 1:
        return tup[0] == check[0] and tup[1] == check[1]
    else:
        return False


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
