from flask import Flask, request, g
import time
from flask_restful  import Resource, Api, reqparse
from flask_cors import CORS, cross_origin
from json import dumps
from flask_jsonpify import jsonify


from collections import namedtuple
import json




import os

from torchvision import  transforms
import matplotlib.pyplot as plt
import cv2    
from PIL import Image
import torch
import torch.nn as nn
import pandas as pd    

from vgg_pytorch import VGG
from resnet_pytorch import ResNet
  
############################################### REST API ######################################################### 


app = Flask(__name__)
api = Api(app)

CORS(app)








parser = reqparse.RequestParser()


@app.route('/')
def index():
    return "<h1>Welcome to Ian's Dog breed Classifier RestApi !!</h1>"



class Post_Prediction(Resource):  
    def post(self):
        
        imageurl=request.files['imageurl']

        result = run_app(imageurl)

        return jsonify(result)    









api.add_resource(Post_Prediction,'/api/v1/predict')







if __name__ == '__main__':
    app.run(threaded=True,port=8000)











############################################ MODEL PIPELINE ###################################################



# define VGG16 model
VGG16 = VGG.from_pretrained("vgg16")
# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")





def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    img = Image.open(img_path)

    if dog_detector(img_path) is True:
        prediction = predict_breed_transfer(img_path)
        return {"message": "Dogs is Detected! in image. It looks like a {0}".format(prediction), "prediction": prediction} 
    elif face_detector(img_path) > 0:
        prediction = predict_breed_transfer(img_path)
        return {"message": "Hey, human! If you were a dog..You may look like a {0}".format(prediction), "prediction": prediction }
    else:
        return {"message": "Error! Can't detect anything in image"}



# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    pt = os.path.abspath(img_path.filename)
    img = plt.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0



### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    
    predict =VGG16_predict(img_path)
    
    return predict >= 151 and predict <=268 # true/false

class_names = pd.read_csv('prediction_classes.csv')

def predict_breed_transfer(img_path):

    model_transfer =load_model('model_transfer.pt')
    # load the image and return the predicted breed
    image = Image.open(img_path).convert('RGB')
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = prediction_transform(image)[:3,:,:].unsqueeze(0) 

    model = model_transfer
    model.eval()
    idx = torch.argmax(model(image))
    idx = idx.detach().numpy()
    return class_names.iloc[idx]['class_name']   



def load_model(path):
    """ this function loads the saved model
    Input: it takes in the path to the saved checkpoint
               
    Return: the saved model
    
    """
    # load checkpoint
    model = VGG16
    checkpoint = torch.load(path,map_location=torch.device('cpu'))

    



    # define new layer for model
    input_feat = model.classifier[6].in_features
    new_layer = nn.Linear(input_feat, 133)
    model.classifier[6] = new_layer
    

        
    model.load_state_dict(checkpoint)
 
    print("Model successfuly loaded")
    return model



def VGG16_predict(img_path):
    '''
    Use pre-trained resnet50 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to resnet50 model's prediction
    '''
    

    image = Image.open(img_path).convert('RGB')
    # resize to (244, 244) because VGG16 accept this shape
    vgg16_transform = transforms.Compose([
                        transforms.Resize(size=(244, 244)),
                        transforms.ToTensor()]) # normalizaiton parameters from pytorch doc.

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = vgg16_transform(image)[:3,:,:].unsqueeze(0)
    
    

    
    VGG16.eval()
    output = VGG16(image)
    
    # Reverse the log function in our output
    predict = torch.max(output,1)[1].item()
    


    
    return predict # predicted class index


