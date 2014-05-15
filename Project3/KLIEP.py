from __future__ import division
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist


class KLIEP:

    def __init__(self, sigma, init_b=100, max_iter=100, epsilons=10**np.asarray(range(3, -3, -1))):
        self.sigma = sigma
        self.init_b = init_b   # initial number of kernels
        self.max_iter = max_iter
        self.epsilons = epsilons

    def KG(self, X_l, M_l):
        basis_distances = cdist(X_l, M_l, 'euclidean')
        K = np.exp(-basis_distances ** 2 / (2 * self.sigma**2))
        return K

    def fit(self, x_tr, x_te, rand_index=None):
        # generate a random permutation of centers
        # possible improvement - permute so that the points are
        # sorted according to their distance to x_tr?
        # (k-nearest neighbor approach)
        if rand_index is None:
            rand_index = np.random.permutation(x_te.shape[0])
        # get the minimum b
        self.b = min(self.init_b, x_te.shape[0])
        # parse the centers (for the basis functions kernels)
        k_cent = x_te[rand_index[0:self.b], :]

        # try also different kernels?
        A_jl = self.KG(x_te, k_cent)
        tmp_X = self.KG(x_tr, k_cent)
        b_l = np.mean(tmp_X, axis=0).reshape((self.b, 1))

        c = np.dot(b_l.T, b_l)

        # for predicting later
        self.centers = k_cent.copy()

        # initial alphas
        alpha = np.ones((self.b, 1))
        score_old = - np.inf

        for epsilon in self.epsilons:
            for i in xrange(self.max_iter):
                tmp_alpha = alpha.copy()
                alpha = alpha + epsilon*np.dot(A_jl.T, np.dot(A_jl, alpha) ** -1)
                # we use pinv since we cannot always compute the inverse
                alpha = alpha + b_l * (1 - np.dot(b_l.T, alpha))*np.linalg.pinv(c)
                alpha = np.maximum(np.zeros(alpha.shape), alpha)
                alpha = np.dot(alpha, np.linalg.pinv(np.dot(b_l.T, alpha)))
                score = np.mean(np.log(np.dot(A_jl, alpha)))

                if score - score_old <= 0:
                    self.alpha = tmp_alpha
                    break

                print 'Score at iter', i+1, ':', score
                score_old = score

        print 'Final score:', score_old
        # self.alpha = alpha

    def predict(self, X):
        X_de = self.KG(X, self.centers)
        w = np.dot(X_de, self.alpha).T
        return w


def main():
    np.random.seed(seed=0)

    # mat = loadmat('kliep.mat')
    # x_de = mat['x_de'].T
    # x_nu = mat['x_nu'].T
    # rand_index = mat['rand_index'].T.ravel() - 1
    # kliep = KLIEP(0.09998, init_b=20)
    # kliep.fit(x_de, x_nu, rand_index=rand_index)
    # w = kliep.predict(x_de)

    x_de = np.random.multivariate_normal([0, 0], np.eye(2), 100)
    x_nu = np.random.multivariate_normal([0, 0], 0.5*np.eye(2), 100)

    kliep = KLIEP(1, init_b=100)
    kliep.fit(x_de, x_nu)

    # w = kliep.predict(x_de)
    # print w


if __name__ == '__main__':
    main()
