from flask import Flask, render_template, request, jsonify
from archi_nlp import Archi
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
    user_data = request.json
    print(user_data)
    results = archi.predict(user_data)
    print(type(results))
    data = [(result[1]['title'],
             result[1]['code'],
             round(result[1]['score'], 2)) for result in results.iterrows()]
    table = render_template('cards.html', data=data)
    return jsonify({'user_query': user_data,
                    # 'results': results,
                    'table': table})
    # index(results=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
