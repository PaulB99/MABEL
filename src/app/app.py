from flask import Flask, request, render_template, redirect, url_for
from gevent.pywsgi import WSGIServer
import torch 
import sys
import time
import os
sys.path.insert(0, '../')
from joint.run import runner
app = Flask(__name__)
taggers = ['base_model', 'large_model', 'lexi']
neutralisers = ['bart', 'roberta', 'parrot']
tagger=''
neutraliser=''

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
            if not os.path.exists('../../cache/taggers/' + tagger +'.pt') and tagger!='lexi':
                error+="{} has not been trained yet! Please run the relevant testing script and try again.<br>".format(tagger)
            if not os.path.exists('../../cache/neutralisers/' + neutraliser + '.pt') and neutraliser!='parrot':
                error+="{} has not been trained yet! Please run the relevant testing script and try again.<br>".format(neutraliser)      
            if error != "":
                return render_template('index.html', taggers=taggers,neutralisers=neutralisers, error=error)
            else:
                global app_runner
                app_runner = runner(tagger, neutraliser)
                return redirect(url_for('get_text'))
        else:
            error="Error! Please select a tagger and neutraliser"
    return render_template('index.html', taggers=taggers,neutralisers=neutralisers, error=error)
    
@app.route('/run', methods=['GET'])
def get_text():
    return render_template('runner.html', tagger=tagger, neutraliser=neutraliser, out_text='')

@app.route('/run', methods=['GET', 'POST'])
def run_model():
    if request.method == 'POST':
        text = request.form.get('textinput')
        processed_text = app_runner.pipeline(text)
        print('Processed text: ' + processed_text)
        return render_template('runner.html', tagger=tagger,neutraliser=neutraliser,out_text=processed_text)
    
    
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