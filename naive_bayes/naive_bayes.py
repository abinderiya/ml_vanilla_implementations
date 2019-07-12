from loadMNIST import load_mnist, save_images
import numpy as np
from scipy.misc import logsumexp


def binarize(x):
    return np.where(x < 0.5, 0, 1)

def map_estimate(x_train, y_train):
    thetas = np.zeros((y_train.shape[1], x_train.shape[1]))
    class_count = np.dot(np.ones((y_train.shape[0],)),y_train) + 2 * np.ones(y_train.shape[1])
    for i in range(x_train.shape[0]):
        index = np.argmax(y_train[i,])
        thetas[index,] += x_train[i,]
    thetas += np.ones(thetas.shape)
    return thetas / class_count[:,None]

def log_posterior(predictors, thetas):
    posterior = np.zeros((predictors.shape[0], thetas.shape[0]))
    for j in range(thetas.shape[0]):
        theta = thetas[j,]
        theta = np.repeat(theta.reshape((-1, len(theta))), predictors.shape[0], axis = 0)
        log_likelihood = np.zeros(predictors.shape)
        log_likelihood[np.where(predictors<0.5)] = np.log(1 - theta[np.where(predictors<0.5)])
        log_likelihood[np.where(predictors>0.5)] = np.log(theta[np.where(predictors>0.5)])
        posterior[:,j] = np.sum(log_likelihood, axis = 1)
    normalize = logsumexp(posterior, axis = 1)
    return posterior - normalize[:,None]

def avg_log_likelihood(predictors, targets, thetas):
    avg = 0
    posterior = log_posterior(predictors, thetas)
    for i in range(targets.shape[0]):
        avg += posterior[i, np.argmax(targets[i,])]
    return avg / targets.shape[0]

def predict(predictors, thetas):
    posterior = log_posterior(predictors, thetas)
    return np.argmax(posterior, axis=1)

def accuracy(predictors, targets, thetas):
    predictions = predict(predictors, thetas)
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == np.argmax(targets[i,]):
            accuracy += 1
    return accuracy / len(predictions)

def generate_random_images(thetas, n = 10):
    random_images = np.zeros((n, thetas.shape[1]))
    for i in range(n):
        c = np.argmax(np.random.multinomial(1, [1/10.]*10))
        theta = thetas[c,]
        for j in range(thetas.shape[1]):
            pixel = np.random.binomial(1, theta[j])
            if pixel == 1:
                random_images[i, j] = pixel
    save_images(random_images, 'sample_images.png')

def complete_images(incomplete_images, thetas):
    bottom_thetas = thetas[:, incomplete_images.shape[1]:]
    posterior = np.exp(log_posterior(incomplete_images, thetas))
    bottom = np.dot(posterior, bottom_thetas)
    images = np.concatenate((incomplete_images, bottom), axis = 1)
    save_images(images, 'completed.png')

if __name__ == '__main__':
	n, x_train, y_train, x_test, y_test = load_mnist()
	x_train = binarize(x_train[:10000,])
	y_train = y_train[:10000,]
	x_test = binarize(x_test)
	thetas = map_estimate(x_train, y_train)
	generate_random_images(thetas)
	incomplete_images = x_train[:20, :int(x_train.shape[1]/2)]
	complete_images(incomplete_images, thetas)
	test_acc = accuracy(x_test, y_test, thetas)
	train_acc = accuracy(x_train, y_train, thetas)
	test_l = avg_log_likelihood(x_test, y_test, thetas)
	train_l = avg_log_likelihood(x_train, y_train, thetas)

