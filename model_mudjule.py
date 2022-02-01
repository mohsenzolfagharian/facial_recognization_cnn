from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing import image
import time
import preprocessing_module
import cv2




def model(OutputNeurons, training_set, test_set):
    classifier = Sequential()
    classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(64, activation='relu'))
    classifier.add(Dense(OutputNeurons, activation='softmax'))
    classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])
    StartTime=time.time()
    classifier.fit_generator(
                        training_set,
                        steps_per_epoch=9,
                        epochs=10,
                        validation_data=test_set,
                        validation_steps=10)
    EndTime=time.time()
    classifier.save('model.h5')
    print("###### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ######')
    return classifier


def load_pickle():
    ResultMap = pickle.load(open("ResultsMap.pkl", "rb" ))
    return ResultMap


def load_custom_model():
    model = load_model('model.h5')
    return model

def verify(model, ResultMap):
    ImagePath='/home/mohsen/Desktop/project_uni/input.jpg'
    test_image=image.load_img(ImagePath,target_size=(64, 64))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result = model.predict(test_image,verbose=0)
 
    # print('####'*10)
    print('Prediction is: ',ResultMap[np.argmax(result)])
    return ResultMap[np.argmax(result)]


# train model
# data = preprocessing_module.run_module()
# trained_model = model(*data)
# loaded_pickle = load_pickle()

def live_detection():

    trained_model = load_custom_model()
    loaded_pickle = load_pickle()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, frame = cap.read()
        frame = frame[150: 150+350, 200:200+350, :]
        if cv2.waitKey(10) == ord('s'):
            cv2.imwrite('input.jpg', frame)
            cv2.putText(frame,'wait for 3s',(200,200), cv2.FONT_HERSHEY_SIMPLEX, .5,(255,255,255),2,cv2.LINE_AA)
            time.sleep(3)
        if cv2.waitKey(10) == ord('q'):
            break
        name = verify(trained_model, loaded_pickle)
        cv2.putText(frame,f'press "s" for save and "q" for quit.',(0,20), cv2.FONT_HERSHEY_SIMPLEX, .5,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame,f'You are {name}',(0,40), cv2.FONT_HERSHEY_SIMPLEX, .5,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('camera', frame)
    cap.release()
    cv2.destroyAllWindows()
    verify(trained_model, loaded_pickle)


live_detection()

# mod = load_custom_model()

# print(mod.summary())
