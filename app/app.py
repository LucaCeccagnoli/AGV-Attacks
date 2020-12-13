import os 
import sys
import json
import glob
from PIL import Image
from flask import Flask, render_template, request, jsonify, make_response, Response
from werkzeug.utils import secure_filename

# project modules
sys.path.append(os.path.abspath('../flask_attacks'))
from forms import ModelForm
from NNModels import NNModels
from run_attack import attack
from filters import _file_to_cv, _cv_to_base64

# global variables
MODEL_DIRECTORY = os.path.join(os.getcwd(), 'static/models/') 
IMAGE_DIRECTORY = os.path.join(os.getcwd(), 'static/images/')  

app = Flask(__name__)  
nnmodels = NNModels()

# flask setting   
ALLOWED_EXTENSIONS = ('png','jpg','jpeg',"JPEG")   # allowed extensions
SECRET_KEY = os.urandom(32)

app.config['SECRET_KEY'] = SECRET_KEY
app.config['FLASK_DEBUG'] = 1

# homepage
@app.route('/')
def index():
    # automatically loads models from MODEL_DIRECTORY
    form = ModelForm()
    images = get_images(IMAGE_DIRECTORY + "imagenet/")
    images_data = []

    # json object for the images: uses the filename as key and contains the class code and base 64 encoding

    for image in images:
        images_data.append(
            {
                "name": image[0].rsplit( ".", 1 )[ 0 ],
                "class_code": image[1],
                "b64data": image[2]
            }
        )

    form.model.choices = tuple_list_from_dict(get_models(MODEL_DIRECTORY))

    return render_template('index.html', form = form, images_data = images_data)

@app.route('/predict/', methods = ['POST'])
def run_prediction():
    # get input
    in_img = request.files['image']
    data = json.loads(request.form['jsonData'])
    print(data)
    network_name = data['network']
    top = data['top']

    # predict the image class
    predictions, class_codes = nnmodels.get_predictions(_file_to_cv(in_img), network_name, top)

    # build and return the json response
    response = {}
    for i in range(len(class_codes)):
        response[class_codes[i]] = {
            "snippet": predictions[i][0],
            "name": predictions[i][1],
            "probability": round(predictions[i][2] * 100,2)
        }

    print(response)

    return jsonify(response)

# run attack
@app.route('/runattack/', methods = ['POST'])
def run_attack():
    # read the image as bytes
    in_img = request.files['image']

    # read additional json data
    # data['model'] contains the string name of the model
    data = json.loads(request.form['jsonData'])
    model_path =  os.path.join(MODEL_DIRECTORY, data['model'])
    network = data['network']
    top = data['top']

    # run attack on the image, returns the modified image as a numpy array
    # arguments: input image as array of bytes, path to the model to run the attack, 
    # returns a jpg image in base 64 which can be sent via json
    mod_img_np, mod_img_b64 = attack(_file_to_cv(in_img), model_path)  
    predictions, class_codes = nnmodels.get_predictions(mod_img_np, network, top)

    # build and return the json response
    response = {
        "encoding": "data:image/jpeg;base64,", 
        "img_base64": mod_img_b64,
        "mod_class_name": predictions[0][1],
        "mod_class_code": class_codes[0]
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
    images = []     # [image name, ground truth code, base64 encoding]  

    # ground truth file
    ground_truth = open((path + "caffe_clsloc_validation_ground_truth.txt"),'r').readlines()
    for fname in os.listdir(path):
        if fname.endswith(ALLOWED_EXTENSIONS):
            #images.append([(f.rsplit( ".", 1 )[ 0 ]), f])
            images.append([fname]) 


    if(len(images) > 0):
        # read ground truth files, when an image matches one in images, get its code
        # O(n) but only works the entries in images are ordered like the ones in the folder
        current = 0
        for row in ground_truth:
            split = row.split()
            if split[0] == images[current][0]:   # if image names match
                images[current].append(split[1]) # append the ground_truth code
                current += 1
                if(current > len(images)-1):
                    break

        # encode every image in base64 to send it to the client
        for image in images:
            with open(path + image[0], mode = 'rb') as byte_image:
                # file name without extension as the label
                label = image[0].rsplit( ".", 1 )[ 0 ]

                # substitute the b64 encoding to the original file
                b64_image = _cv_to_base64(_file_to_cv(byte_image))
                image.append(b64_image)
        
        return images
    else:
        return None      


# converts a dictionary in a list of tuples (key, value)
def tuple_list_from_dict(in_dict):
    out_list = []
    for entry in in_dict:
        tp = (entry, in_dict[entry])
        out_list.append(tp)

    return out_list