from flask import Flask, render_template, request, jsonify
from src.archi_nlp import Archi
app = Flask(__name__)


archi = Archi('en_core_web_lg')
archi.get_raw_nlp_data('data/nlp_df/raw_nlp_05-11.pkl')


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
    # print(results)
    data = [(result['title'], result['text']) for result in results]
    table = render_template('table.html', data=data)
    return jsonify({'user_query': user_data,
                    'results': results,
                    'table': table})
    # index(results=results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
