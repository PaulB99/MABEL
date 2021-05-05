from flask import Flask, request, render_template, redirect, url_for
from gevent.pywsgi import WSGIServer
import torch 
import sys
import time
import os
sys.path.insert(0, '../')
from joint.run import runner
import transformers
transformers.logging.set_verbosity_error()

app = Flask(__name__)
taggers = ['base_model', 'large_model', 'distilbert', 'lexi']
neutralisers = ['bart', 'roberta', 'seq2seq', 'miniseq2seq', 'parrot', 'lexi_swap']
tagger=None
neutraliser=None

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html', taggers=taggers,neutralisers=neutralisers, error="")


@app.route('/', methods=['GET','POST'])
def get_models():
    error=""
    if request.method == 'POST':  
        # Update tagger and neutraliser names
        global tagger
        tagger = request.form.get('taggers')
        global neutraliser
        neutraliser = request.form.get('neutralisers')
        if tagger is not None and neutraliser is not None:
            if not os.path.exists('../../cache/taggers/' + tagger +'.pt') and tagger!='lexi':
                error+="{} has not been trained yet! Please run the relevant testing script and try again.<br>".format(tagger)
            if not os.path.exists('../../cache/neutralisers/' + neutraliser + '.pt') and (neutraliser!='parrot' and neutraliser!='lexi_swap'):
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
    return render_template('runner.html', tagger=tagger, neutraliser=neutraliser, out_text='', biased='')


@app.route('/run', methods=['GET', 'POST'])
def run_model():
    if request.method == 'POST':
        if 'run' in request.form:
            text = request.form.get('textinput')
            processed_text, biased = app_runner.pipeline(text)
            if biased:
                print('Biased!')
                b_text = 'Biased!'
            else:
                print('Unbiased!')
                b_text = 'Unbiased!'
            print('Processed text: ' + processed_text)
            return render_template('runner.html',
                                   tagger=tagger,
                                   neutraliser=neutraliser,
                                   out_text=processed_text, 
                                   biased=b_text,
                                   residual_text=text)
        elif 'return' in request.form:
            return redirect(url_for('main'))
    
@app.route('/models')
def show_models():
    return render_template('models.html')

@app.route('/models', methods=['POST'])
def return_to_selection():
    if request.method == 'POST':
        return redirect(url_for('main'))

if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    print('Hosting on localhost:5000 ......')
    http_server.serve_forever()