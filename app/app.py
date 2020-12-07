import os 
import sys
import json
import glob
from PIL import Image
from flask import Flask, render_template, request, jsonify, make_response
from werkzeug.utils import secure_filename

# project modules
sys.path.append(os.path.abspath('../flask_attacks'))
from forms import ModelForm
from networks import NNModels
from run_attack import attack
from filters import _file_to_cv, _cv_to_base64

# global variables
MODEL_DIRECTORY = os.path.join(os.getcwd(), 'static/models/') 
IMAGE_DIRECTORY = os.path.join(os.getcwd(), 'static/images/')  

app = Flask(__name__)  
nnmodels = NNModels()

# flask setting   
ALLOWED_EXTENSIONS = ('png','jpg','jpeg')   # allowed extensions
SECRET_KEY = os.urandom(32)

app.config['SECRET_KEY'] = SECRET_KEY
app.config['FLASK_DEBUG'] = 1

# homepage
@app.route('/')
def index():
    # automatically loads models from MODEL_DIRECTORY
    form = ModelForm()
    form.imagenet_image.choices = tuple_list_from_dict(get_images(IMAGE_DIRECTORY + "imagenet/"))
    form.model.choices = tuple_list_from_dict(get_models(MODEL_DIRECTORY))

    return render_template('index.html', form = form)

@app.route('/predict/', methods = ['POST'])
def run_prediction():
    # get input
    in_img = request.files['image']
    data = json.loads(request.form['jsonData'])
    network_name = data['network']
    print(network_name)

    # predict the image class
    class_name, class_code = nnmodels.get_predictions(_file_to_cv(in_img), network_name)
    print(class_name)
    print(class_code)

    # build and return the json response
    response = {
        "class_name": class_name, 
        "class_code": class_code, 
    }
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
        mod_class_name, mod_class_code = nnmodels.get_predictions(mod_img_np, network)

        # build and return the json response
        response = {
            "encoding": "data:image/jpeg;base64,", 
            "img_base64": mod_img_b64,
            "mod_class_name": mod_class_name,
            "mod_class_code": mod_class_code
        }

        return jsonify(response)

#returns all json files from the input path as attack models
def get_models(path):
    choices = {}
    choices[''] = 'Choose a model'
    for jsonFile in os.listdir(path):
        if jsonFile.endswith(".json"):
            fname = jsonFile.rsplit( ".", 1 )[ 0 ]
            choices[jsonFile] = fname

    return choices

def get_images(path):
    choices = {}
    choices[''] = 'Choose from ImageNet...'
    for f in os.listdir(path):
        if f.endswith(ALLOWED_EXTENSIONS):
            # encode file in base 64, return filename and encoding
            b64 = ''
            with open(path + f, mode = 'rb') as img_bytes:
                b64 = _cv_to_base64(_file_to_cv(img_bytes))
            if(b64 != ''):
                fname = f.rsplit( ".", 1 )[ 0 ]
                choices[b64] = fname
    return choices

# converts a dictionary in a list of tuples (key, value)
def tuple_list_from_dict(in_dict):
    out_list = []
    for entry in in_dict:
        tp = (entry, in_dict[entry])
        out_list.append(tp)

    return out_list