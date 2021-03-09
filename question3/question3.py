import re
import numpy as np
from sklearn.neural_network import MLPRegressor

INPUT_FILE_1 = 'train_data.txt'
INPUT_FILE_2 = 'train_truth.txt'
INPUT_FILE_3 = 'test_data.txt'
OUT_FILE = 'test_predicted.txt'


def read_data(file_name):
    """
    Read the features from the file
    :param file_name: name of the file
    :return: a 2D-list with float
    """
    matrix = []
    is_title = True
    file = open(file_name)
    for line in file.readlines():
        if is_title:
            is_title = False
            continue
        # delete the '\n' and space at the beginning and the end of the string
        data = line.strip()
        # split the data into pieces
        data = re.split('[\t ]', data)
        data = [float(x) for x in data]  # convert the string to numbers
        matrix.append(data)
    return matrix


def read_truth(file_name):
    """
    Read the truth from the file
    :param file_name: name of the file
    :return: a 2D-list with float
    """
    matrix = []
    is_title = True
    file = open(file_name)
    for line in file.readlines():
        if is_title:
            is_title = False
            continue
        # delete the '\n' and space at the beginning and the end of the string
        data = line.strip()
        data = float(data)  # convert the string to numbers
        matrix.append(data)
    return matrix


train_x = read_data(INPUT_FILE_1)  # read train features from file
test_x = read_data(INPUT_FILE_3)  # read test features from file
y = read_truth(INPUT_FILE_2)  # read y from file
X_train, X_test, y = np.array(train_x), np.array(test_x), np.array(
    y)  # convert list to array
y = y.ravel()  # make a copy for y

model = MLPRegressor(hidden_layer_sizes=130, learning_rate="adaptive")
model.fit(X_train, y.astype('float'))  # train with the train data

y_predict = model.predict(X_test)  # predict according to the test features
# write the predicted result to the file
file = open(OUT_FILE, 'w')
file.write('y\n')
for y in y_predict:
    file.write(str('%e'%y) + '\n')
file.close()

