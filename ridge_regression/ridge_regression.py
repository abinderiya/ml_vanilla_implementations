import numpy as np
import seaborn as sns
import scipy.io as sio
import urllib.request
import matplotlib.pyplot as plt

def download_data(download=False):
	if download:
		url = 'https://duvenaud.github.io/sta414/dataset.mat'
		file_name = 'dataset.mat'
		urllib.request.urlretrieve(url, file_name)


def shuffle_data(data):
    new_indices = np.random.permutation(data.shape[0])
    return np.take(data, new_indices, axis = 0)
    

def split_data(data, num_fold, fold):
    n_obsv_fold = np.floor(data.shape[0]/num_fold)
    if fold == num_fold:
        start = (num_fold - 1) * n_obsv_fold
        end = data.shape[0]
    else:
        start = (num_fold - 1) * n_obsv_fold
        end = num_fold * n_obsv_fold
    fold_ = np.arange(start, end, dtype = int)
    rest = list(set(np.arange(data.shape[0])) - set(fold_))
    return np.take(data, fold_, axis = 0), np.take(data, rest, axis = 0)

def train_model(data, lambd):
    y = data[:,0]
    X = data[:,1:]
    design_mat =  np.dot(X.T, X) + np.identity(X.shape[1]) * lambd
    left_hs = np.dot(X.T, y)
    return np.linalg.solve(design_mat, left_hs)

def predict(data, model):
    X = data[:, 1:]
    return np.dot(X, model).reshape((data.shape[0],))

def loss(data, model):
    target = data[:, 1]
    predictions = predict(data, model)
    residual = target - predictions
    return np.dot(residual.T, residual)/data.shape[0]

def cross_validation(data, num_fold, lambd_seq):
    data = shuffle_data(data)
    cv_error = []
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        temp_error = 0
        for j in range(1, num_fold + 1):
            fold, rest = split_data(data, num_fold, j)
            model = train_model(rest, lambd)
            temp_error += loss(data, model)
        cv_error.append(temp_error/float(num_fold))
    return cv_error

def train_test_error(train_data, test_data, lambd_seq):
    train_error = []
    test_error = []
    for i in range(len(lambd_seq)):
        model = train_model(train_data, lambd_seq[i])
        train_error.append(loss(train_data, model))
        test_error.append(loss(test_data, model))
    return train_error, test_error


def main():
    dataset = sio.loadmat('./dataset.mat')
    data_train_X = dataset['data_train_X']
    data_train_y = dataset['data_train_y'][0]
    data_test_X = dataset['data_test_X']
    data_test_y = dataset['data_test_y'][0]    
    train_data = np.concatenate((data_train_y.reshape((-1,1)), data_train_X), axis=1)
    test_data = np.concatenate((data_test_y.reshape((-1,1)), data_test_X), axis = 1)
    lambd_seq = np.linspace(0.02, 1.5)
    train_error, test_error = train_test_error(train_data, test_data, lambd_seq)
    cv_5 = cross_validation(train_data, 5, lambd_seq)
    cv_10 = cross_validation(train_data, 10, lambd_seq)
    fig, ax = plt.subplots(2, 2)
    plt.subplot(2, 2, 1)
    plt.scatter(lambd_seq, cv_5)
    plt.subplot(2, 2, 2)
    plt.scatter(lambd_seq, cv_10)
    plt.subplot(2, 2, 3)
    plt.scatter(lambd_seq, train_error)
    plt.subplot(2, 2, 4)
    plt.scatter(lambd_seq, train_error)
    plt.show()


if __name__ == '__main__':
	main()








