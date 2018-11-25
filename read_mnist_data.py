import _pickle
import gzip
import  numpy as np

def load_data():
    file = gzip.open('mnist.pkl.gz','rb')
    training_data, validation_data, test_data = _pickle.load(file, encoding='bytes')
    file.close()
    return training_data, validation_data, test_data

def prep_data():

    train, validate, test = load_data()

    train_inputs = [np.reshape(x, (784, 1)) for x in train[0]]
    train_outputs = [calculate_op_vect(y) for y in train[1]]
    training_data = zip(train_inputs, train_outputs)

    validate_inputs = [np.reshape(x, (784, 1)) for x in validate[0]]
    validation_data = zip(validate_inputs, validate[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in test[0]]
    testing_data = zip(test_inputs, test[1])

    return training_data,validation_data,testing_data



def calculate_op_vect(correct_op):

    vectorized_correct_op = np.zeros((10,1))
    vectorized_correct_op[correct_op] = 1.0

    return vectorized_correct_op


load_data()
prep_data()