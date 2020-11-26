import cv2
import numpy
import os
import keras
from filters import _cv_to_array, _to_cv_image

# input: numpy or cv2 image
def get_predictions(in_img, network_name):
    # network model selection
    model = None
    if network_name == 'VGG':
        from tensorflow.keras.applications.vgg19 import VGG19, decode_predictions
        model = VGG19(weights='imagenet')
    else:
        print("no valid model selected")
        return

    # preprocess input
    print(type(in_img))
    # if input is a numpy image, convert it to a cv image
    if( isinstance( in_img, numpy.ndarray)):
        in_img = _to_cv_image(in_img)

    img = cv2.resize(in_img, dsize=(244,244), interpolation = cv2.INTER_CUBIC)
    img = _cv_to_array(img)
    img = numpy.expand_dims(img, axis = 0)

    # predict model and return top prediction
    features = model.predict(img)
    return decode_predictions(features, top=1)[0][0][1]   # ottiene nome della classe predetta

if __name__ == '__main__':
    img = cv2.imread('dolce.jpg')
    get_predictions(img, 'VGG')