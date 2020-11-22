import os
from flask_wtf import FlaskForm
from wtforms import SelectField, FileField

class ModelForm(FlaskForm):
    custom_image = FileField( id='custom_image')
    model = SelectField(id = 'model', choices = None)
    network = SelectField(id = 'network', choices = [("Mobilenet","Mobilenet"),("VGG","BGG"), ("Densenet10","Densenet10")])

