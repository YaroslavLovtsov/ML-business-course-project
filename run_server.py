from flask import Flask, render_template, url_for
from requests import request
# import  learn_model
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/api/make-prediction')
def make_prediction():
    # print(request)
    return "Hello World"


if __name__ == '__main__':
    app.run(debug=True)
