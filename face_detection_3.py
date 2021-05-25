# 
"""
Created on Wed Ja

@author: ULVI PC
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
# load in color image for face detection
image = cv2.imread('P1_Facial_Keypoints-master/images/obamas.jpg')

# switch red and blue color channels 
# --> by default OpenCV assumes BLUE comes first, not RED as in many images
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

'''# plot the image
fig = plt.figure(figsize=(9,9))
plt.imshow(image)'''

# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('P1_Facial_Keypoints-master/detector_architectures/haarcascade_frontalface_default.xml')

# run the detector
# the output here is an array of detections; the corners of each detection box
# if necessary, modify these parameters until you successfully identify every face in a given image
faces = face_cascade.detectMultiScale(image, 2.5, 2)

# make a copy of the original image to plot detections on
image_with_detections = image.copy()

# loop over the detected faces, mark the image where each face is found
for (x,y,w,h) in faces:
    # draw a rectangle around each detected face
    # you may also need to change the width of the rectangle drawn depending on image resolution
    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3) 

fig = plt.figure(figsize=(9,9))

plt.imshow(image_with_detections)

import torch
from models import Net
from torch.autograd import Variable

net = Net()

net.load_state_dict(torch.load('C:/Users/ULVI PC/Desktop/Programming/Neural Network/saved_models/keypoints_model_110.pt'))

## print out your net and prepare it for testing (uncomment the line below)
net.eval()

image_copy = np.copy(image)

# loop over the detected faces from your haar cascade
def showpoints(image,keypoints):
    
    plt.figure()
    
    keypoints = keypoints.data.numpy()
    keypoints = keypoints * 60.0 + 68
    keypoints = np.reshape(keypoints, (68, -1))
    
    plt.imshow(image, cmap='gray')
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=50, marker='.', c='r')
    


from torch.autograd import Variable
image_copy = np.copy(image)


# loop over the detected faces from your haar cascade
for (x,y,w,h) in faces:
    
    # Select the region of interest that is the face in the image 
    roi = image_copy[y:y+h,x:x+w]

    ## TODO: Convert the face region from RGB to grayscale
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    image = roi

    ## TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    roi = roi/255.0
    
    ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    roi = cv2.resize(roi, (224,224))
    
    ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    roi = np.expand_dims(roi, 0)
    roi = np.expand_dims(roi, 0)
    
    ## TODO: Make facial keypoint predictions using your loaded, trained network 
    roi_torch = Variable(torch.from_numpy(roi))
    
    roi_torch = roi_torch.type(torch.FloatTensor)
    keypoints = net(roi_torch)

    ## TODO: Display each detected face and the corresponding keypoints        
    showpoints(image,keypoints)
    
