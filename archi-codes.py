from flask import Flask, render_template, request, jsonify, Markup
from archi_nlp import Archi
from render_text import render_text
import datetime as dt

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


@app.route('/solve', methods=['POST'])
def solve():
    user_query = request.json
    print(user_query)
    # predict returns results and the nlp output of the user query
    results, query_doc = archi.predict(user_query)
    print(type(results))
    data = [(result[1]['nlp_chapter_title'].text,
             result[1]['source_doc']['title'],
             result[1]['title'],
             result[1]['nlp_code_text'].text,
             round(result[1]['score'], 2)) for result in results.iterrows()]
    rendered_query = render_text(query_doc)
    uq_annotated = render_template('annotate_text.html', data=Markup(rendered_query))
    table = render_template('cards.html', data=data)
    return jsonify({'user_query': uq_annotated,
                    'table': table})
    # index(results=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
