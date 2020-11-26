import os 
import sys
import json
from PIL import Image
from flask import Flask, render_template, request, jsonify, make_response
from werkzeug.utils import secure_filename

# project modules
sys.path.append(os.path.abspath('../flask_attacks'))
from forms import ModelForm
from networks import get_predictions
from run_attack import attack
from filters import _file_to_cv

# global variables
MODEL_DIRECTORY = os.path.join(os.getcwd(), 'static/models/') 

app = Flask(__name__)  

# flask setting 
UPLOAD_FOLDER = 'static/images'     # folder for images uploaded to the server
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}   # allowed extensions
SECRET_KEY = os.urandom(32)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = SECRET_KEY
app.config['FLASK_DEBUG'] = 1

# homepage
@app.route('/')
def index():
    # automatically loads models from MODEL_DIRECTORY
    form = ModelForm()
    form.model.choices = tuple_list_from_dict(get_models(MODEL_DIRECTORY))

    return render_template('flask-attacks.html', form = form)

@app.route('/predict/', methods = ['POST'])
def run_prediction():
    # get input
    in_img = request.files['image']
    data = json.loads(request.form['jsonData'])
    network_name = data['network']

    # predict the image class
    prediction = get_predictions(_file_to_cv(in_img), network_name)
    # TODO: salvare l'immagine convertita nel server?

    # build and return the json response
    response = {
        "prediction": prediction, 
    }

    print(response)
    return jsonify(response)

# run attack
@app.route('/runattack/', methods = ['POST'])
def run_attack():
    if request.method == 'POST':

        # read the image as bytes
        in_img = request.files['image']

        # read additional json data
        # data['model'] contains the string name of the model
        data = json.loads(request.form['jsonData'])
        model_path =  os.path.join(MODEL_DIRECTORY, data['model'])
        network = data['network']
    
        # run attack on the image, returns the modified image as a numpy array
        # arguments: input image as array of bytes, path to the model to run the attack, 
        # returns a jpg image in base 64 which can be sent via json
        mod_img_np, mod_img_b64 = attack(_file_to_cv(in_img), model_path)  
        mod_class = get_predictions(mod_img_np, network)

        # build and return the json response
        response = {
            "encoding": "data:image/jpeg;base64,", 
            "img_base64": mod_img_b64,
            "modified_class": mod_class
        }

        return jsonify(response)

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