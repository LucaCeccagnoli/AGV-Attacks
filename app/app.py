import os 
import sys
import json
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename

# project modules
sys.path.append(os.path.abspath('../flask_attacks'))
from run_attack import attack
from forms import ModelForm

# global variables
MODEL_DIRECTORY = os.path.join(os.getcwd(), 'static/models/') 

app = Flask(__name__)  

# flask setting 
UPLOAD_FOLDER = 'static/images'     # folder for images uploaded to the server
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}   # allowed extensions
SECRET_KEY = os.urandom(32)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = SECRET_KEY

# homepage
@app.route('/')
def index():
    # automatically loads models from MODEL_DIRECTORY
    form = ModelForm()
    form.model.choices = tuple_list_from_dict(get_models(MODEL_DIRECTORY))

    return render_template('flask-attacks.html', form = form)

# execute attack
@app.route('/runattack/', methods = ['POST'])
def run_attack():
    if request.method == 'POST':

        # read the image as bytes
        in_img = request.files['image']

        # read additional json data
        # data['model'] contains the string name of the model
        data = json.loads(request.form['jsonData'])
    
        # run attack on the image, returns the modified image as a numpy array
        # arguments: input image as array of bytes,
        #            name of the model to run the attack, 
        #            folder to save the modified image
        #            filename for the modified image
        img_path = os.path.join(UPLOAD_FOLDER, secure_filename(in_img.filename))
        model_path =  os.path.join(MODEL_DIRECTORY, data['model'])
        attack(in_img.read(), model_path , in_img.filename, img_path)

        # get the modified image
        return send_from_directory(app.config['UPLOAD_FOLDER'], secure_filename("modified.jpg"), as_attachment=True)

#returns all json files from the input path as attack models
def get_models(path):
    choices = {}
    choices[''] = 'Choose a model'
    for jsonFile in os.listdir(path):
        if jsonFile.endswith(".json"):
            fname = jsonFile.rsplit( ".", 1 )[ 0 ]
            print(fname)
            choices[jsonFile] = fname

    return choices

# converts a dictionary in a list of tuples (key, value)
def tuple_list_from_dict(in_dict):
    out_list = []
    for entry in in_dict:
        tp = (entry, in_dict[entry])
        out_list.append(tp)

    return out_list