from flask import Flask, request, render_template
from gevent.pywsgi import WSGIServer
import torch 
import sys
import time
import os
sys.path.insert(0, '../')
from joint.run import runner
app = Flask(__name__)
taggers = ['base_model', 'large_model']
neutralisers = ['bart']

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html', taggers=taggers,neutralisers=neutralisers, error="")

@app.route('/', methods=['GET','POST'])
def get_models():
    error=""
    if request.method == 'POST':   
        tagger = request.form.get('taggers')
        neutraliser = request.form.get('neutralisers')
        if tagger is not None and neutraliser is not None:
            if not os.path.exists('../../cache/taggers/' + tagger +'.pt'):
                error+="{} has not been trained yet! Please run the relevant testing script and try again.\n".format(tagger)
            if not os.path.exists('../../cache/neutralisers/' + neutraliser + '.pt'):
                error+="{} has not been trained yet! Please run the relevant testing script and try again.\n".format(neutraliser)      
            if error != "":
                return render_template('index.html', taggers=taggers,neutralisers=neutralisers, error=error)
            else:
                return render_template('runner.html', tagger=tagger, neutraliser=neutraliser)
        else:
            error="Error! Please select a tagger and neutraliser"
    return render_template('index.html', taggers=taggers,neutralisers=neutralisers, error=error)
    
#@app.route('/', methods=['POST'])
#def my_form_post():
  #  variable = request.form['variable']
   # return variable

if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    print('Hosting on localhost:5000 ......')
    http_server.serve_forever()