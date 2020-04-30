import numpy as np
from glob import glob
import cv2  
import torch
from PIL import Image
import pandas as pd
import torchvision.models as models
import torch.nn as nn
import os
from torchvision import  transforms
# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")





def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    img = Image.open(img_path)

    if dog_detector(img_path) is True:
        prediction = predict_breed_transfer(img_path)
        print("Dogs is Detected! in image \nIt looks like a {0}".format(prediction))  
    elif face_detector(img_path) > 0:
        prediction = predict_breed_transfer(img_path)
        print("Hey, human!\nIf you were a dog..You may look like a {0}".format(prediction))
    else:
        print("Error! Can't detect anything in image")



# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0



### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    
    predict =VGG16_predict(img_path)
    
    return predict >= 151 and predict <=268 # true/false

class_names = pd.read_csv('web\backend\src\prediction_classes.csv')

def predict_breed_transfer(img_path):
    model_path = os.path.join('models','model_transfer.pt')
    model_transfer =load_model(model_path)
    # load the image and return the predicted breed
    image = Image.open(img_path).convert('RGB')
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = prediction_transform(image)[:3,:,:].unsqueeze(0) 
    image = image.cuda()
    model = model_transfer.cuda()
    model.eval()
    idx = torch.argmax(model(image))

    return class_names.iloc[idx - 1]['class_name']   



def load_model(path):
    """ this function loads the saved model
    Input: it takes in the path to the saved checkpoint
               
    Return: the saved model
    
    """
    # load checkpoint
    checkpoint = torch.load(path)

    model = models.resnet50(pretrained=True)
    model.eval()
    # define new layer for model
    input_feat = model.fc.in_features
    new_layer = nn.Linear(input_feat, 133)
    model.fc = new_layer
    
        
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
    
    # define VGG16 model
    VGG16 = models.vgg16(pretrained=True)
    VGG16.eval()
    image = Image.open(img_path).convert('RGB')
    # resize to (244, 244) because VGG16 accept this shape
    vgg16_transform = transforms.Compose([
                        transforms.Resize(size=(244, 244)),
                        transforms.ToTensor()]) # normalizaiton parameters from pytorch doc.

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = vgg16_transform(image)[:3,:,:].unsqueeze(0)
    
    

    
    
    output = VGG16(image)
    
    # Reverse the log function in our output
    predict = torch.max(output,1)[1].item()
    


    
    return predict # predicted class index