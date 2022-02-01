from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

path_train_imgs = '/home/mohsen/Desktop/project_uni/Face Images/Final Training Images'

def data_generator():
    train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True) 
        
    return train_datagen


# Generating the Training Data
def training_set_func(train_data_gen, Training_Image_Path):
        training_set = train_data_gen.flow_from_directory(
            Training_Image_Path,
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')
        return training_set



def test_set_func(test_datagen, Training_Image_Path):
    test_set = test_datagen.flow_from_directory(
        Training_Image_Path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
    return test_set


def make_pickle(TrainClasses):

    ResultMap={}
    for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
        ResultMap[faceValue]=faceName

    with open("ResultsMap.pkl", 'wb') as fileWriteStream:
        pickle.dump(ResultMap, fileWriteStream)

    print("Mapping of Face and its ID",ResultMap)
    OutputNeurons=len(ResultMap)
    
    print('\n The Number of output neurons: ', OutputNeurons)
    return OutputNeurons

def run_module():
    train_datagen = data_generator()
    train_set = training_set_func(train_datagen, path_train_imgs)
    test_set = test_set_func(train_datagen, path_train_imgs)
    pickle_file = make_pickle(train_set.class_indices)
    return (pickle_file, train_set, test_set)

x = run_module()
print(x)
