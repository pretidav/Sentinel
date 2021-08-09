
#define VGG deep learning face recognition topology.

from keras import Sequential 
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

from keras.models import model_from_json
model.load_weights('./VGGnet/vgg_face_weights.h5')

vgg_face_descriptor =Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

#img1_representation = vgg_face_descriptor.predict(preprocess_image('DSC_1584_FINAL.JPG'))[0,:]
#img2_representation = vgg_face_descriptor.predict(preprocess_image('DSC_1584_FINAL.JPG'))[0,:]

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
 
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

epsilon = 0.40 #cosine similarity
#epsilon = 120 #euclidean distance
 
def verifyFace(img1, img2):
 img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0,:]
 img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0,:]
 
 cosine_similarity = findCosineDistance(img1_representation, img2_representation)
 euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
 
 if(cosine_similarity < epsilon):
  print("verified... they are same person")
 else:
  print("unverified! they are not same person!")    

#PREDICTIONS
print('################################')
verifyFace('./photoset/DAVID_face.jpg','./face.jpg')
print('################################')