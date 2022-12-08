import os
import sys
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import random 
import json
import plotly
import numpy as np
from flask import Flask,redirect, url_for
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import cv2                
import matplotlib.pyplot as plt    
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image             
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions  
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense

from extract_bottleneck_features import *
from tqdm import tqdm
from glob import glob


app = Flask(__name__,static_url_path='/static')

#======================================================================
#1. LOAD DATA
#======================================================================
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

def load_data():
    # ===================Load Dog Images=============================
    # load train, test, and validation datasets
    train_files, train_targets = load_dataset('../data/dog_images/train')
    valid_files, valid_targets = load_dataset('../data/dog_images/valid')
    test_files, test_targets = load_dataset('../data/dog_images/test')
    dog_names = [item[20:-1] for item in sorted(glob("../data/dog_images/train/*/"))]


    # print statistics about the dataset
    print('There are %d total dog categories.' % len(dog_names))
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.'% len(test_files))
    
    return train_files,train_targets,valid_files,valid_targets,test_files, test_targets,dog_names

#======================================================================
#2. BUILD MODEL
#======================================================================
def build_model():
    ### TODO: Obtain bottleneck features from another pre-trained CNN.
    bottleneck_features = np.load('../data/bottleneck_features/DogResnet50Data.npz')
    train_Resnet50 = bottleneck_features['train']
    valid_Resnet50 = bottleneck_features['valid']
    test_Resnet50 = bottleneck_features['test']
    
    Resnet50_model = Sequential()
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=(train_Resnet50.shape[1:])))
    Resnet50_model.add(Dense(133, activation='softmax'))
    Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return train_Resnet50,valid_Resnet50,test_Resnet50, Resnet50_model

#======================================================================
#3. HUMAN DETECT
#======================================================================
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('../data/haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

#======================================================================
#4. DOG DETECT
#======================================================================
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    ResNet50_model = ResNet50(weights='imagenet')
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)].split('.')[1]

def algorithm(img_path):
    isHuman = face_detector(img_path)
    isDog = dog_detector(img_path)
    
    if isHuman:
        return "Human", Resnet50_predict_breed(img_path)
    
    if isDog:
        return "Dog", Resnet50_predict_breed(img_path)
    
    return "Cannot_Detect", ""

def CheckAlgorithm(img_path):
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Identification, DogBreed = algorithm(img_path)
    temp = img_path.replace('\\','/')
    title = "Filename :{0};\Identification: {1}; Dog breed: {2};\n".format(temp.split('/')[-1],Identification, DogBreed)
    plt.imshow(cv_rgb)
    plt.title(title);
    plt.show()
    
def CheckAlgorithmSaveResult(img_path,filename):
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Identification, DogBreed = algorithm(img_path)
    title = "Identification: {0}; Dog breed: {1};\n".format(Identification, DogBreed)
    plt.imshow(cv_rgb)
    plt.title(title);
    out_img_path = img_path.replace(filename,'ResultOut_'+filename)
    print(out_img_path)
    plt.savefig(out_img_path)
    return 'ResultOut_'+filename,Identification, DogBreed
    
def IsImage(input):
    output = False
    if '.' in input:
        if input.split('.')[1].lower() in image_extensions:
            output = True
    return output

#======================================================================
#5. MAIN
#======================================================================
int_epochs=20
image_extensions = 'jpg,jpeg,png'

IMAGE_FOLDER = 'static\\images'

print('Loading data...')
train_files,train_targets,valid_files,valid_targets,test_files, test_targets,dog_names = load_data()
        
print('Building model...')
train_Resnet50,valid_Resnet50,test_Resnet50, Resnet50_model = build_model()
        
print('Epochs= {0}, Training model... '.format(int_epochs))
checkpointer = ModelCheckpoint(filepath='../models/weights.best.Resnet50.hdf5', 
    verbose=1, save_best_only=True)

Resnet50_model.fit(train_Resnet50, train_targets, 
    validation_data=(valid_Resnet50, valid_targets),
    epochs=int_epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

print('Trained model saved!')

#load models
Resnet50_model.load_weights('../models/weights.best.Resnet50.hdf5')

# get index of predicted dog breed for each image in test set
Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]

# report test accuracy
test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
str_test_accuracy = "{0:.2f} %".format(test_accuracy)
print('Test accuracy: ' + str_test_accuracy)

# #check
# input_files = np.array(glob("../data/Test/*"))
# print('There are %d total input images.' % len(input_files))
# for path in input_files:
#     CheckAlgorithm(path)
#     break

#======================================================================
#6. WEB
#======================================================================
@app.route('/')
def index():
    return render_template('master.html',
                           status = '',
                           result_image='',
                           Identification='',
                           DogBreed='',
                           EpochsNum=int_epochs,
                           TestAccuracy=str_test_accuracy)

@app.route('/display_image/<filename>')
def display_image(filename):
	return redirect(url_for('static',filename='images/' + filename), code=301)

@app.route('/go' , methods = ['GET' , 'POST'])
def go():
    status = ''
    result_image = ''
    Identification=''
    DogBreed=''
    # save user input in query
    if request.method == 'POST':
        print(request)
        if (request.files):
            image_input = request.files['imageFile']
            print(image_input)
            if image_input:
                if IsImage(image_input.filename):
                    print(image_input.filename)
                    img_path = os.path.join(IMAGE_FOLDER , image_input.filename)
                    print(img_path)
                    image_input.save(img_path)
                    result_image,Identification, DogBreed = CheckAlgorithmSaveResult(img_path,image_input.filename)
                else:
                    status = 'Please upload image file with jpg,jpeg,png format then try again !'
            else:
                status = 'Can not upload your image file. Please try again !'
        else:
            status = 'Can not upload your image file. Please try again !'
            
        if len(status)==0:
            return render_template('master.html',
                                   status = '',
                                   result_image=result_image,
                                   Identification=Identification,
                                   DogBreed=DogBreed,
                                   EpochsNum=int_epochs,
                                   TestAccuracy=str_test_accuracy)
        else:
            return render_template('master.html',
                                   status = status,
                                   result_image='',
                                   Identification='',
                                   DogBreed='',
                                   EpochsNum=int_epochs,
                                   TestAccuracy=str_test_accuracy)
            
    else:
        return render_template('master.html',
                               status = '',
                               result_image='',
                               Identification='',
                               DogBreed='',
                               EpochsNum=int_epochs,
                               TestAccuracy=str_test_accuracy)
    
def main():
   app.run(host='0.0.0.0', port=4000, debug=True)
    

if __name__ == '__main__':
    main()