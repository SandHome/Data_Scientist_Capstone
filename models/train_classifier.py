import sys
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import random 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
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

    # print statistics about the dataset
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    
    return train_files,train_targets,valid_files,valid_targets

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
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
    Resnet50_model.add(Dense(133, activation='softmax'))
    Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    return train_Resnet50,valid_Resnet50,test_Resnet50, Resnet50_model


def main():
    print(len(sys.argv))
    if len(sys.argv) == 3:
        input_epochs, model_name = sys.argv[1:]
        int_epochs = 0
        try:
            int_epochs=int(input_epochs)
        except:
            print('Please input the epochs value as number !')
            
        if(int_epochs>0):
            print('Loading data...')
            train_files,train_targets,valid_files,valid_targets = load_data()
                    
            print('Building model...')
            train_Resnet50,valid_Resnet50,test_Resnet50, Resnet50_model = build_model()
                    
            print('Training model...')
            checkpointer = ModelCheckpoint(filepath='../models/'+model_name, 
                verbose=1, save_best_only=True)

            Resnet50_model.fit(train_Resnet50, train_targets, 
                validation_data=(valid_Resnet50, valid_targets),
                epochs=int_epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

            print('Trained model saved!')
    
    else:
        print('Please provide the epochs as the first argument '\
              'and the models name as the second argument.\n\nExample: python '\
              'train_classifier.py 20 weights.best.Resnet50.hdf5')


if __name__ == '__main__':
    main()