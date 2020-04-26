from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   validation_split=0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory(r'C:\Users\roshan.gupta\Downloads\Family\train',
                                                 target_size = (224, 224),
                                                 batch_size = 8,
                                                 subset="training",
                                                 class_mode = 'categorical')

validation_set = train_datagen.flow_from_directory(r'C:\Users\roshan.gupta\Downloads\Family\train',
                                                 target_size = (224, 224),
                                                 batch_size = 8,
                                                subset="validation",
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r'C:\Users\roshan.gupta\Downloads\Family\test',
                                            target_size = (224, 224),
                                            batch_size = 8,
                                            class_mode = 'categorical')

#color_mode = "grayscale"


STEP_SIZE_TRAIN=training_set.n//training_set.batch_size
STEP_SIZE_VALID=validation_set.n//validation_set.batch_size
STEP_SIZE_TEST=test_set.n//test_set.batch_size