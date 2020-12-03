import cv2
import numpy
import os
import keras
from filters import _cv_to_array, _to_cv_image

class NNModels(object):
    def __init__(self):
        self.MODEL_VGG19 = None
        self.MODEL_DENSENET201 = None
        self.MODEL_MOBILENET = None
        self.MODEL_INCEPTIONV3 = None
        self.MODEL_RESNET50 = None
        

# input: numpy or cv2 image
    def get_predictions(self, in_img, network_name):
        # network model selection
        model = None
        size = (244,244)
        if network_name == 'VGG19':
            if self.MODEL_VGG19 is None:
                from tensorflow.keras.applications.vgg19 import VGG19
                self.MODEL_VGG19 = VGG19(weights='imagenet')
            model = self.MODEL_VGG19
            from tensorflow.keras.applications.vgg19 import decode_predictions

        elif network_name == 'DenseNet201':
            if self.MODEL_DENSENET201 is None:
                from tensorflow.python.keras.applications.densenet import DenseNet201
                self.MODEL_DENSENET201 = DenseNet201(weights='imagenet')
            model = self.MODEL_DENSENET201
            from tensorflow.keras.applications.densenet import decode_predictions

        elif network_name == 'MobileNetV2':
            if self.MODEL_MOBILENET is None:
                from tensorflow.keras.applications.mobilenet import MobileNet # (244,244)
                self.MODEL_MOBILENET = MobileNet(weights='imagenet')
            model = self.MODEL_MOBILENET
            from tensorflow.keras.applications.mobilenet import decode_predictions

        elif network_name == 'InceptionV3':
            if self.MODEL_INCEPTIONV3 is None:
                from tensorflow.keras.applications.inception_v3 import InceptionV3 # (299,299)
                self.MODEL_INCEPTIONV3 = InceptionV3(weights='imagenet')
            model = self.MODEL_INCEPTIONV3
            size = (299, 299)
            from tensorflow.keras.applications.inception_v3 import decode_predictions

        elif network_name == 'ResNet50':
            if self.MODEL_RESNET50 is None:
                from tensorflow.python.keras.applications.resnet import ResNet50 # (244,244)
                self.MODEL_RESNET50 = ResNet50(weights='imagenet')
            model = self.MODEL_RESNET50
            from tensorflow.keras.applications.resnet import decode_predictions

        else:
            print("no valid model selected")
            return


        # if input is a numpy image, convert it to a cv image
        if( isinstance( in_img, numpy.ndarray)):
            in_img = _to_cv_image(in_img)

        img = cv2.resize(in_img, dsize=size, interpolation = cv2.INTER_CUBIC)
        img = _cv_to_array(img)
        img = numpy.expand_dims(img, axis = 0)

        # predict model and return top prediction
        features = model.predict(img)
        name = decode_predictions(features, top=1)[0][0][1]
        code = int(numpy.argmax(features))
        return (name, code)   # ottiene nome della classe predetta

if __name__ == '__main__':
    img = cv2.imread('dolce.jpg')
    nnmodel = NNModels()
    print(nnmodel.get_predictions(img, 'VGG19'))
    print(nnmodel.get_predictions(img, 'VGG19'))