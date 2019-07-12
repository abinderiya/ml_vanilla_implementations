from loadMNIST import load_mnist, save_images
import numpy as np
from scipy.misc import logsumexp


def binarize(x):
    return np.where(x < 0.5, 0, 1)


def grad_log_posterior(x_train, y_train, w):
    grad = np.zeros(w.shape)
    for i in range(x_train.shape[0]):
        sums = np.exp(np.dot(w, x_train[i,]))
        sums /= sum(sums)
        sums *= -1
        grad_x = np.dot(sums.reshape((len(sums), -1)), x_train[i,].reshape((-1, len(x_train[i,]))))
        grad_x[np.argmax(y_train[i,]),] += x_train[i,]
        grad += grad_x
    return grad/x_train.shape[0]


def grad_ascent(x_train, y_train, step_size = 0.5, max_iter = 500, error = 0.0003):
    w_old = np.zeros((y_train.shape[1], x_train.shape[1]))
    w_new = w_old + step_size * grad_log_posterior(x_train, y_train, w_old)
    itr = 1
    while np.max(np.dot(np.abs(w_old - w_new), np.ones(w_old.shape[1]))) > error and itr < max_iter:
        itr += 1
        w_old = w_new
        w_new = w_old + step_size * grad_log_posterior(x_train, y_train, w_old)
    return w_new, itr


def avg_log_posterior(x, y, w):
    posterior_matrix = np.dot(x, w.T)
    posterior_matrix /= logsumexp(posterior_matrix, axis = 1)[:, None]
    log_posterior = posterior_matrix[y.astype(bool)]
    return sum(log_posterior)/len(log_posterior)


def accuracy(x, y, w):
    prob = np.exp(np.dot(w, x.T))
    sums = np.dot(np.ones((1, prob.shape[0])), prob)
    prob /= sums
    truth = np.argmax(y, axis = 1)
    prediction = np.argmax(prob, axis = 0)
    return sum(truth == prediction)/len(prediction)

if __name__ == '__main__':
	n, x_train, y_train, x_test, y_test = load_mnist()
	x_train = binarize(x_train[:10000,])
	y_train = y_train[:10000,]
	x_test = binarize(x_test)
	w, itr = grad_ascent(x_train, y_train)
	test_acc = accuracy(x_test, y_test, w)
	train_acc = accuracy(x_train, y_train, w)
	test_l = avg_log_likelihood(x_test, y_test, w)
	train_l = avg_log_likelihood(x_train, y_train, w)
	

	