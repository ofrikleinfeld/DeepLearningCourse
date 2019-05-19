import pandas as pd
import gzip
import pickle
import numpy as np

def crop_and_augment():
    train = pd.read_csv('data/train.csv', header = None)
    train_data = train.iloc[:, 1:].values
    train_label = train.iloc[:, 0].values
    validation = pd.read_csv('data/validate.csv', header = None)
    validation_data = validation.iloc[:, 1:].values
    validation_label = validation.iloc[:, 0].values
    test = pd.read_csv('data/test.csv', header = None)
    test_data = test.loc[:, 1:].values

    cropped_train_mirrored = []
    for i in range(train_data.shape[0]):
        rgb = train_data[i]
        img = rgb.reshape(3,32,32).transpose([1, 2, 0])
        img = img[2:30, 2:30]
        img = np.fliplr(img)
        rgb = img.transpose([2, 0, 1]).reshape(2352, )
        cropped_train_mirrored.append(rgb)

    cropped_train= []
    for i in range(train_data.shape[0]):
        rgb = train_data[i]
        img = rgb.reshape(3,32,32).transpose([1, 2, 0])
        img = img[2:30, 2:30]
        rgb = img.transpose([2, 0, 1]).reshape(2352, )
        cropped_train.append(rgb)


    cropped_valid = []
    for i in range(validation_data.shape[0]):
        rgb = validation_data[i]
        img = rgb.reshape(3,32,32).transpose([1, 2, 0])
        img = img[2:30, 2:30]
        rgb = img.transpose([2, 0, 1]).reshape(2352, )
        cropped_valid.append(rgb)

    cropped_test = []
    for i in range(test_data.shape[0]):
        rgb = test_data[i]
        img = rgb.reshape(3,32,32).transpose([1, 2, 0])
        img = img[2:30, 2:30]
        rgb = img.transpose([2, 0, 1]).reshape(2352, )
        cropped_test.append(rgb)

    cropped_train_mirrored = np.array(cropped_train_mirrored)
    cropped_train_mirrored = np.hstack((train.iloc[:, 0].values.reshape(-1,1), cropped_train_mirrored))

    cropped_train = np.array(cropped_train)
    cropped_train = np.hstack((train.iloc[:, 0].values.reshape(-1,1), cropped_train))

    cropped_valid = np.array(cropped_valid)
    cropped_valid = np.hstack((validation.iloc[:, 0].values.reshape(-1,1), cropped_valid))

    cropped_test = np.array(cropped_test)
    cropped_test = np.hstack((test.iloc[:, 0].values.reshape(-1,1), cropped_test))

    pd.DataFrame(cropped_train_mirrored).to_csv(index = False, 
                                     header = False, path_or_buf='data/train_mirrored_cropped_smaller.csv')
    pd.DataFrame(cropped_train).to_csv(index = False, 
                                     header = False, path_or_buf='data/train_cropped_smaller.csv')
    pd.DataFrame(cropped_valid).to_csv(index = False, 
                                     header = False, path_or_buf='data/valid_cropped_smaller.csv')
    pd.DataFrame(cropped_test).to_csv(index = False, 
                                     header = False, path_or_buf='data/test_cropped_smaller.csv')