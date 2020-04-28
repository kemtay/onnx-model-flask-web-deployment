# onnx-model-flask-web-deployment
To deploy a ONNX model as a web application with Python Flask.

## Prerequisites
The below Python modules are required:

1. Python Flask packages- the micro web application framework
- from flask import Flask, render_template, request
- from flask_bootstrap import Bootstrap
- from flask_wtf import FlaskForm
- from wtforms import SubmitField
- from flask_wtf.file import FileField, FileRequired, FileAllowed 

2. Python libraries for image processing: cv2, PIL
3. Python libraries for data visualization: matplotlib.pyplot, seaborn 
4. Python libraries for ONNX runtime: onnxruntime

## Python files/folders explained:
1. azurecv_birdclassifier.py - the main python program to run the web application (python azurecv_birdclassifier.py)
2. bc_forms.py - the form for user input an image file for classification
3. bird_classifications.py - contains the main functions (display_bird() and classify_bird()) for producing outputs
4. folders 
- static: consists of favicon image file, img folder and css folder (this is a standard structure for a flask web application)
- templates: consists of HTML templates to be rendered by Flask (this is a standard structure for a flask web application)
- models: consists of the ONNX models and labels.txt (both downloaded from Microsoft CV services)
- plots: placeholder for the output images

## Model
- The ONNX is a deep learning model trained by Microsoft Azure Custom Vision services for image classification.
- The train/test dataset are a private image collection of birds seen in Singapore.

## The python program explained:
After running the main python program with 'python azurecv_birdclassifier.py':
- the web application will be running on 'http://127.0.0.1:50001/' (localhost)
- open a web browser and place the url at the address bar
- a web GUI will be displayed
![mainpage-imgfile](/images/mainpage.png)
- click on 'Choose File' button to select an image file from your local disk
- click on 'submit' button when ready
- the image and a bar chart with the Top-5 classification will be displayed
![outputpage-imgfile](/images/outputpage.png)

## Results Summary
- The result of the ONNX model is not satisfying mainly due to a relatively small and unbalanced dataset. 
- The aim of this simple project is to demostrate the ONNX model deployment using Flask.

## Reference
1. https://onnx.ai/ 
2. https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/home
3. Book: Flask Web Development, Developing Web Applications with Python (https://flaskbook.com/) 
4. https://wtforms.readthedocs.io/en/stable/
