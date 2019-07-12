from loadMNIST import load_mnist, save_images
import numpy as np
from scipy.misc import logsumexp
import matplotlib.pyplot as plt


def cost(assignments, clusters, data):
    c_1 = np.dot(data - clusters[0,][None, :], (data - clusters[0,][None, :]).T).diagonal()
    c_2 = np.dot(data - clusters[1,][None, :], (data - clusters[1,][None, :]).T).diagonal()
    distance = np.concatenate((c_1[:,None], c_2[:,None]), axis = 1)
    return sum(np.dot(distance, assignments.T).diagonal())


def km_e_step(clusters, data):
    c_1 = np.dot(data - clusters[0,][None, :], (data - clusters[0,][None, :]).T).diagonal()
    c_2 = np.dot(data - clusters[1,][None, :], (data - clusters[1,][None, :]).T).diagonal()
    distance = np.concatenate((c_1[:,None], c_2[:,None]), axis = 1)
    return (distance == np.min(distance, axis = 1)[:,None]).astype(int)


def km_m_step(assignments, data):
    clusters = np.dot(assignments.T, data)
    return clusters / np.sum(assignments, axis = 0)[:,None]

def train_km(data):
    costs = []
    clusters = np.array([[0 ,0], [1.0, 1.0]])
    assignments = km_e_step(clusters, data)
    old_cost = cost(assignments, clusters, data)
    costs.append(old_cost)
    clusters = km_m_step(assignments, data)
    assignments = km_e_step(clusters, data)
    new_cost = cost(assignments, clusters, data)
    while new_cost != old_cost:
        old_cost = new_cost
        costs.append(old_cost)
        clusters = km_m_step(assignments, data)
        assignments = km_e_step(clusters, data)
        new_cost = cost(assignments, clusters, data)
    costs.append(new_cost)
    return assignments, clusters, costs

def accuracy(truth, assignments):
    return sum(truth[:,0] == assignments[:,0])/len(truth[:,0])

def normal_density(x, mean, cov):
    n_d = []
    for k in range(mean.shape[0]):
        diff = x - mean[k,]
        dist = np.dot(np.dot(diff, np.linalg.inv(cov[k,])), diff.T)
        dist *= -0.5
        n_d.append(dist)
    n_d = np.array(n_d)
    dist = np.exp(n_d)
    denom = np.pi**(len(x)/2)*np.linalg.det(cov)**(1/2)
    return dist/denom

def log_l(x, mean, cov, prior):
    log_l = 0
    for i in range(x.shape[0]):
        s = np.dot(prior, normal_density(x[i,], mean, cov))
        log_l += np.log(s)
    return log_l


def em_e_step(x, mean, cov, prior):
    responsibilities = np.zeros((x.shape[0], cov.shape[0]))
    for i in range(x.shape[0]):
        num = prior * normal_density(x[i,], mean, cov)
        num /= sum(num)
        responsibilities[i,] = num
    return responsibilities


def get_cov(x, mean, r):
    cov = np.zeros((x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        diff = x[i,] - mean
        cov += np.dot(diff[:,None], diff[None,:]) * r[i]
    return cov


def em_m_step(responsibilities, x):
    effective_count = np.sum(responsibilities, axis = 0)
    mean = np.dot(responsibilities.T, x)/effective_count[:,None]
    cov = np.zeros((responsibilities.shape[1], x.shape[1], x.shape[1]))
    for k in range(cov.shape[0]):
        r = responsibilities[:,k]
        cov[k,] = get_cov(x, mean[k,], r)/effective_count[k]
    prior = effective_count/sum(effective_count)
    return mean, cov, prior

def train_em(data):
    lst_l = []
    mean = np.array([[0 ,0], [1.0, 1.0]])
    cov = np.array([[[1,0],[0,1]],[[1,0],[0,1]]])
    prior = np.array([0.5, 0.5])
    old_l = log_l(data, mean, cov, prior)
    lst_l.append(old_l)
    responsibilities = em_e_step(data, mean, cov, prior)
    mean, cov, prior = em_m_step(responsibilities, data)
    new_l = log_l(data, mean, cov, prior)
    while old_l != new_l:
        old_l = new_l
        lst_l.append(old_l)
        responsibilities = em_e_step(data, mean, cov, prior)
        mean, cov, prior = em_m_step(responsibilities, data)
        new_l = log_l(data, mean, cov, prior)
    lst_l.append(new_l)
    return lst_l, responsibilities, mean, cov

if __name__ == '__main__':
	cov = np.array([[10, 7],[7, 10]])
	mean_1 = np.array([0.1, 0.1])
	mean_2 = np.array([6.0, 0.1])
	samples_1 = np.random.multivariate_normal(mean_1, cov, 200)
	samples_2 = np.random.multivariate_normal(mean_2, cov, 200)
	data = np.concatenate((samples_1, samples_2), axis = 0)
	x_1 = samples_1[:,0]
	y_1 = samples_1[:,1]
	x_2 = samples_2[:,0]
	y_2 = samples_2[:,1]
	plt.plot(x_1, y_1, 'x')
	plt.plot(x_2, y_2, 'o')
	plt.xlabel('x coordinate')
	plt.ylabel('y coordinate')
	plt.savefig('real.png')
	plt.show()
	l, r, mean, cov = train_em(data)
	assignments, clusters, costs = train_km(data) 