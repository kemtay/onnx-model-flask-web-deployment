# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:40:18 2020

@author: kemtay
"""

from flask_wtf import FlaskForm
from wtforms import SubmitField
from flask_wtf.file import FileField, FileRequired, FileAllowed

class SelectFileForm(FlaskForm):   
    
    img_file = FileField('Please select the image file:', validators=[FileRequired()])
    
    submit = SubmitField('Submit')
