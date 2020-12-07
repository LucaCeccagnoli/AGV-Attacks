import json
import pickle
import copy
import base64
import random
import cv2, base64
import PIL
import numpy 
import os

from filters import _to_cv_image, _cv_to_array, _cv_to_base64
from agv_model_loader import ModelLoader
from agv_optimizer import Individual

def attack(in_img, model_path):

    # load the model and apply
    model = ModelLoader().load(model_path)
    in_img = _cv_to_array(in_img)
    mod_img_np = model.apply(in_img)

    # return modified np image and its base64 encoding
    return (mod_img_np, _cv_to_base64(_to_cv_image(mod_img_np)))


if __name__ == "__main__":
    attack()