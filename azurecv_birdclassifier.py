# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:56:58 2020

@author: kemtay
"""

import time
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from bc_forms import SelectFileForm
from bird_classifications import classify_bird, display_bird

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    
    file_form = SelectFileForm()
    
    if file_form.validate_on_submit() and request.method == 'POST':
        #img_file = request.form['img_file']
        img_file = file_form.img_file.data
        rand_num = int(time.time())
        
        if img_file != '':
            return render_template('classification_output.html',
                                   fig_1=display_bird(img_file),
                                   fig_2=classify_bird(img_file),
                                   rand_num=rand_num)
                                   
    return render_template('classification_index.html', file_form=file_form)

@app.route('/intro')
def intro():
    return render_template('intro.html')

app.run(host='127.0.0.1', port=50001)