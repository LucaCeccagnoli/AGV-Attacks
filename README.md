# AGV Attacks
An Adversarial Machine Learning web application that applies Instagram-like filters to images and uses them as adversarial examples on different neural networks.
It was developed for my Bachelor's thesis at the Univeristy of Perugia.

Developed using Flask, Keras and Tensorflow.
The filters are written using OpenCV.
The frontend is written using Bootstrap and jQuery
The predefined images are taken from the ImageNet dataset: https://www.image-net.org/

You can read the full thesis [here](/thesis/AGV_thesis.pdf). 

## Features
* Run adversarial attacks on 5 different convolutional neural networks, using predefined or custom images.
* See how accurately the original image was predicted.
* Use two predefined image filters or create your own.

## Usage
Move to the project directory and run the following commands:
```
  FLASK_APP=app/app:app
```
```
  flask run
```
Then navigate to http://127.0.0.1:5000/ .

Select a preset image or upload one from your computer, along with a CNN model.  

.
![form 1](/thesis/images/form1-preset-image.PNG)  

Press "Predict class" and wait for the prediction to end (will take some time whenever using a model for the first time), then note the prediction scores on the top three classes. The highest one will be used as reference when running the attack.  

![prediction-correct](/thesis/images/predictions-correct.PNG)  

Some models won't be able to correctly classify all images.  
![prediction-wrong](/thesis/images/predictions-failed.PNG)  
 
If you uploaded a custom image, you will be able to choose the reference class for the attack.
![class-1](/thesis/images/ground-truth-1.PNG)
![class-2](/thesis/images/ground-truth-2.PNG)
 
Next, choose one between the two preset attacks.
![attacks](/thesis/images/form2-empty.PNG)

Or create a custom attack. You can add infinite combinations of 5 image filters, setting their intensity and alpha value. Clicking "create" will add it to the previous list of available attacks with the name you specified.

![custom-attack](/thesis/images/attack-editor.PNG)

Finally, run the attack. If successful, the model will have classified the filtered image differently from the original.
![attack-success](/thesis/images/form2-preset-attack.PNG)


