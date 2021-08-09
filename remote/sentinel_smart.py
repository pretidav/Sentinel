
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
  return True, cosine_similarity
 else:
  return False, cosine_similarity    


import cv2


# Connects to your computer's default camera
cap = cv2.VideoCapture(0)
count = 0


# Automatically grab width and height from video feed
# (returns float which we need to convert to integer for later on!)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#tracker
tracker = cv2.TrackerMedianFlow_create()
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

ret, frame = cap.read()
frame_orig = frame.copy()

people = {'DAVID':False,'ROBERTA':False,'MARIO':False,'MATTEO':False,'SIMONE':False,'STEFANO':False}


reps = []
for key, values in people.items():
    reps.append(vgg_face_descriptor.predict(preprocess_image('./faces/' + key + '_face.jpg'))[0,:])

print(np.shape(reps))
roi = face_cascade.detectMultiScale(frame,scaleFactor=1.2, minNeighbors=5) 
for (x,y,w,h) in roi: 
    tracker.init(frame,(x,y,w,h))
    face = frame[y:y+h, x:x+w]
    cv2.imwrite("face.jpg", face)
    iterator = -1
    new_rep = vgg_face_descriptor.predict(preprocess_image('./face.jpg'))[0,:]
    for key, value in people.items():
        iterator +=1
        cosine_similarity = findCosineDistance(reps[iterator], new_rep)    
        if cosine_similarity < epsilon:
            cv2.putText(frame, key, (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
            break
        else: 
            cv2.putText(frame, "Unknown", (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
entered = False

while True:
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    count += int(ret)
    
    # Update tracker
    success, roi = tracker.update(frame)
    (x,y,w,h) = tuple(map(int,roi))
    if success:
        # Tracking success
        p1 = (x, y)
        p2 = (x+w, y+h)
        if entered == False:
            for key, value in people.items():
                people[key]=False

            face = frame[y:y+h, x:x+w]
            cv2.imwrite("face.jpg", face)    
            entered = True    
            iterator = -1
            new_rep = vgg_face_descriptor.predict(preprocess_image('./face.jpg'))[0,:]
            for key, value in people.items():
                iterator +=1
                cosine_similarity = findCosineDistance(reps[iterator], new_rep)  
                if cosine_similarity < epsilon:
                    people[key] = True
                    
        for key, value in people.items():
            if value == True:        
                cv2.putText(frame, key, (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                #print(key)
                break
            
        if (not any(list(people.values()))) == True :
                cv2.putText(frame, "Unknown", (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
        
        cv2.rectangle(frame, p1, p2, (0,255,0), 3)
    else :
        # Tracking failure
        entered = False
        cv2.putText(frame, "ALERT: Detection Failure", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
        if count%5 == 0:
            tracker = cv2.TrackerMedianFlow_create()
            roi = face_cascade.detectMultiScale(frame,scaleFactor=1.2, minNeighbors=5) 
            for (x,y,w,h) in roi: 
                tracker.init(frame,(x,y,w,h))

    # Display the resulting frame
    cv2.imshow('frame',frame)
    
    # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()