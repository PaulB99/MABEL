from flask import Flask, request, render_template
from gevent.pywsgi import WSGIServer
import torch 
import sys
import time
sys.path.insert(0, '../')
from joint.run import runner
app = Flask(__name__)

@app.route('/', methods=['GET'])
def my_form():
    taggers = ['base_model', 'large_model']
    neutralisers = ['bart']
    return render_template('index.html', taggers=taggers,neutralisers=neutralisers)

@app.route('/', methods=['POST'])
def my_form_post():
    variable = request.form['variable']
    return variable

#@app.route('/', methods=['GET'])
#def dropdown_1():
    #taggers = ['base_model', 'Large_model']
    #return render_template('index.html', taggers=taggers)

#@app.route('/', methods=['GET'])
#def dropdown_2():
    #neutralisers = ['bart']
    #return render_template('index.html', neutralisers=neutralisers)

if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    print('Hosting on localhost:5000 ......')
    http_server.serve_forever()