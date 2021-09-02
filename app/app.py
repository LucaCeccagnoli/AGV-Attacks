import os 
import sys
import json
import glob
from PIL import Image
from flask import Flask, render_template, request, jsonify, make_response, Response
from werkzeug.utils import secure_filename

print(os.getcwd())

# project modules
sys.path.append(os.path.abspath('flask_attacks'))
from forms import ModelForm
from attacks import attack, custom_attack
from filters import _file_to_cv, _cv_to_base64
from NNModels import NNModels

# global variables
MODEL_DIRECTORY = os.path.join(os.getcwd(), 'app/static/models/') 
IMAGE_DIRECTORY = os.path.join(os.getcwd(), 'app/static/images/')  

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
    model_form = ModelForm()
    images = get_images(IMAGE_DIRECTORY + "imagenet/")
    images_data = []

    # json object for the images: uses the filename as key and contains the class code and base 64 encoding

    for image in images:
        images_data.append(
            {
                "file_name": image[0].rsplit( ".", 1 )[ 0 ],
                "class_code": image[1],
                "class_name": image[2],
                "b64data": image[3]
            }
        )

    model_form.model.choices = tuple_list_from_dict(get_models(MODEL_DIRECTORY))

    return render_template('index.html', 
                            model_form = model_form, 
                            images_data = images_data)

@app.route('/predict/', methods = ['POST'])
def run_prediction():
    # get input
    in_img = request.files['image']
    data = json.loads(request.form['jsonData'])
    network_name = data['network']
    top = data['top']

    # predict the image class
    predictions, class_codes = nnmodels.get_predictions(_file_to_cv(in_img), network_name, top)

    # build and return the json response
    response = {}
    for i in range(len(class_codes)):
        response[int(class_codes[i])] = {
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
    network = data['network']
    top = data['top']

    response = {}
    if (data['model'].endswith(".json")):
        model_path =  os.path.join(MODEL_DIRECTORY, data['model'])

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
            "mod_class_code": int(class_codes[0])
        }
    else:
        mod_img_np, mod_img_b64 = custom_attack(_file_to_cv(in_img), json.loads(data['model']))
        predictions, class_codes = nnmodels.get_predictions(mod_img_np, network, top)
        response = {
            "encoding": "data:image/jpeg;base64,", 
            "img_base64": mod_img_b64,
            "mod_class_name": predictions[0][1],
            "mod_class_code": int(class_codes[0])
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
    images = []     # [image name, ground truth code, ground truth name, base64 encoding]  

    # read the ground truth file inside path
    ground_truth = open((path + "caffe_clsloc_validation_ground_truth.txt"),'r').readlines()


    # get file names of the images inside path
    # The images need to be named in the format: ILSVRC2012_val_XXXXXXXX.JPEG to be sorted correctly
    for fname in sorted(os.listdir(path)):
        if fname.endswith(ALLOWED_EXTENSIONS):
            images.append([fname]) 


    if(len(images) > 0):
        # Ground Truth Codes
        # read the ground truth file, when a code in images matches one from the ground truth, append its code
        # O(n) but only if works the entries in images are ordered like the ones in the ground truth file
        current = 0

        for row in ground_truth:
            split = row.split()
            if split[0] == images[current][0]:   # if image names match
                images[current].append(int(split[1])) # append the ground_truth code
                current += 1
                if(current > len(images)-1):
                    break

        # Ground Truth Names
        # sort the images according to their ground truth code and find their names in the imagenet class index
        images.sort(key = lambda x: x[1])
        class_indexes =  json.load(open(path + "imagenet_class_index.json"))
        current = 0
        for key, value in class_indexes.items():
            if key == str(images[current][1]):
                images[current].append(value[1])
                current += 1
            if(current > len(images)-1):
                break

        images.sort(key = lambda x: x[0]) # sort the images again by their names

        # encode every image in base64 to send it to the client
        for image in images:
            with open(path + image[0], mode = 'rb') as byte_image:
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