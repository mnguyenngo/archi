from flask import Flask, render_template, request, jsonify
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('archi-codes.html')


@app.route('/solve', methods=['POST'])
def solve():
    user_data = request.json
    q = user_data
    return jsonify({'user_query': q})


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
