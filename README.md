# Building Object Localisation and Detection AI
The University of Helsinki course featuring grow with google certificate practice python programs for building seamless AI programs in small chunks to execute applications


# Building Object Localisation and Detection AI

My Final project for the Building AI course

## Summary

This project features the OpenCV, Mediapipe, TensorFlow and pygame packages with pycharm IDE. It focuses on localizing and detecting real life objects in real time with spot on movements and its frame rate. The idea for this is vast as to be able to develop a AI system that could detect objects, face and hands as in a complete package with proper real time email or messaging alerts if there are any changes undesirable set in the controller. This helps in automation of smart cameras into public sectors for 24/7 monitoring sites for desired optimum security benefits. Usually applicable for security / policing operation or crowd management sectors. 


## Background

Which problems does your idea solve? How common or frequent is this problem? What is your personal motivation? Why is this topic important or interesting?

The problems it solves is: 
* Crowd Management
* Remote Security and Monitoring
* Casual In-house monitoring solutions


## How is it used?

Describe the process of using the solution. In what kind situations is the solution needed (environment, time, etc.)? Who are the users, what kinds of needs should be taken into account?

Images will make your README look nice!
Once you upload an image to your repository, you can link link to it like this (replace the URL with file path, if you've uploaded an image to Github.)
![Cat](https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg)

If you need to resize images, you have to use an HTML tag, like this:
![imgdetection](https://user-images.githubusercontent.com/71119638/131369737-3befdda1-8514-42ed-87d6-bdf73add767b.jpg)
![holo2-detected](https://user-images.githubusercontent.com/71119638/131369768-f8ad5651-ff54-4fde-9aa9-f5e27f778966.jpg)
![imgdetection](https://user-images.githubusercontent.com/71119638/131369796-b7ae279d-0591-4e3b-bf28-d86d22b0c413.jpg)



##Sample Source Code for training.py

!wget https://github.com/hfg-gmuend/openmoji/releases/latest/download/openmoji-72x72-color.zip
!mkdir emojis
!unzip -q openmoji-72x72-color.zip -d ./emojis

!wget https://github.com/hfg-gmuend/openmoji/releases/latest/download/openmoji-72x72-color.zip
!mkdir emojis
!unzip -q openmoji-72x72-color.zip -d ./emojis

%matplotlib inline

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image, ImageDraw
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout

print('Using TensorFlow version', tf.__version__)

emojis = {
    0: {'name': 'happy', 'file': '1F642.png'},
    1: {'name': 'laughing', 'file': '1F602.png'},
    2: {'name': 'skeptical', 'file': '1F928.png'},
    3: {'name': 'sad', 'file': '1F630.png'},
    4: {'name': 'cool', 'file': '1F60E.png'},
    5: {'name': 'whoa', 'file': '1F62F.png'},
    6: {'name': 'crying', 'file': '1F62D.png'},
    7: {'name': 'puking', 'file': '1F92E.png'},
    8: {'name': 'nervous', 'file': '1F62C.png'}
}

plt.figure(figsize=(9, 9))

for i, (j, e) in enumerate(emojis.items()):
    plt.subplot(3, 3, i + 1)
    plt.imshow(plt.imread(os.path.join('emojis', e['file'])))
    plt.xlabel(e['name'])
    plt.xticks([])
    plt.yticks([])
plt.show()

for class_id, values in emojis.items():
    png_file = Image.open(os.path.join('emojis', values['file'])).convert('RGBA')
    png_file.load()
    new_file = Image.new("RGB", png_file.size, (255, 255, 255))
    new_file.paste(png_file, mask=png_file.split()[3])
    emojis[class_id]['image'] = new_file
    
    


## Data sources and AI methods
It will be needing pictures of hands and feet or objects which i will grab from opensource data sets website and feed it to the systems for processing, it will not take more than 100 to 200 pictures of around 250 mb storage of a repository created inside the project folder, however it will be working perfectly as i have a prototype ready with quater of the functions ok tested.

## Challenges

The main challenges is to make a object oriented system which can detect objects, a large number of dataset is prevalent in the web but somehow it is still not enough for a system like this to be working for commercial industries or organisation. I am fully certain that a practice project is already ok tested, like hands and palm recognition is working as per the data i have provided to it but i will need to surpass the boundary of current data sets and need some more practice in to making this more effective system. 

## What next?

I am starting my own small marketing campaign for like minded individuals who are concerned and want to develop projects that they can really be proud of in this AI field, also will ask them to onboard into the Elements of AI course before starting any contributions in this project. I am looking forward to completing the whole project within May 2022. And i will be looking for online resources and academia for tools that have been in practice which is similar to my project and learn from it about how to over come the challenges i am facing at present.   


## Acknowledgments
https://engibex.com/foot-gesture-recognition/
https://www.resna.org/sites/default/files/conference/2015/wheeled_mobility/student_scientific/lyons.html
