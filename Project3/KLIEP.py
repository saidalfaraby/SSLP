from __future__ import division
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist


class KLIEP:

    def __init__(self, sigma=1, init_b=100, max_iter=100, seed=None):
        self.sigma = sigma
        self.init_b = init_b   # initial number of kernels
        self.max_iter = max_iter
        if seed is not None:
            np.random.seed(seed=seed)

    def KG(self, X, M, sigma):
        basis_distances = cdist(X, M, 'euclidean')
        K = np.exp(-basis_distances ** 2 / (2 * sigma**2))
        return K

    def learning(self, b_l, A_jl):
        c = np.dot(b_l.T, b_l)
        # initial alphas
        alpha = np.ones((self.b, 1))
        score_old = -np.inf

        for epsilon in 10**np.asarray(range(3, -4, -1), dtype=float):
            for i in xrange(self.max_iter):
                tmp_alpha = alpha.copy()
                alpha = alpha + epsilon*np.dot(A_jl.T, np.dot(A_jl, alpha) ** -1)
                # we use pinv since we cannot always compute the inverse
                alpha = alpha + b_l * (1 - np.dot(b_l.T, alpha))*np.linalg.pinv(c)
                alpha = np.maximum(np.zeros(alpha.shape), alpha)
                alpha = np.dot(alpha, np.linalg.pinv(np.dot(b_l.T, alpha)))
                score = np.mean(np.log(np.dot(A_jl, alpha)))

                if score - score_old <= 0:
                    final_alpha = tmp_alpha
                    break

                # print 'Score at iter', i+1, ':', score
                score_old = score

        # print 'Final score:', score_old
        return final_alpha

    # simple fit without optimizing for sigma
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
        # for predicting later
        self.centers = k_cent.copy()

        # try also different kernels?
        A_jl = self.KG(x_te, k_cent, self.sigma)
        tmp_X = self.KG(x_tr, k_cent, self.sigma)
        b_l = np.mean(tmp_X, axis=0).reshape((self.b, 1))

        self.alpha = self.learning(b_l, A_jl)

    # cross-validation to optimize sigma
    def fit_CV(self, x_tr, x_te, rand_index=None, n_folds=5):
        if rand_index is None:
            rand_index = np.random.permutation(x_te.shape[0])
        # get the minimum b
        self.b = min(self.init_b, x_te.shape[0])
        # parse the centers (for the basis functions kernels)
        k_cent = x_te[rand_index[0:self.b], :]
        # for predicting later
        self.centers = k_cent.copy()

        sigma, score = 10, -np.inf
        for e in np.arange(np.log10(sigma) - 1, -6, -1):
            for i in xrange(9):
                sigma_new = sigma - 10 ** e
                score_new = 0
                #np.random.shuffle(x_te)
                x_te = x_te[np.random.permutation(x_te.shape[0]), :]
                tmp_X = self.KG(x_tr, k_cent, sigma_new)
                b_l = np.mean(tmp_X, axis=0).reshape((self.b, 1))
                splits = np.array_split(x_te, n_folds)
                for fold in xrange(len(splits)):
                    used = np.vstack([splits[k] for k in xrange(len(splits)) if k != fold])
                    A_jl = self.KG(used, k_cent, sigma_new)
                    alpha_cv = self.learning(b_l, A_jl)
                    wh_cv = np.dot(self.KG(splits[fold], k_cent, sigma_new), alpha_cv)
                    score_new = score_new + np.mean(np.log(wh_cv))/(n_folds)

                if score_new - score <= 0:
                    break
                score = score_new
                sigma = sigma_new
                print 'score:', score, ', sigma:', sigma

        print 'Best sigma:', sigma
        self.sigma = sigma
        tmp_X = self.KG(x_tr, k_cent, sigma)
        b_l = np.mean(tmp_X, axis=0).reshape((self.b, 1))
        A_jl = self.KG(x_te, k_cent, sigma)
        self.alpha = self.learning(b_l, A_jl)

    def predict(self, X):
        X_de = self.KG(X, self.centers, self.sigma)
        w = np.dot(X_de, self.alpha).T
        return w


def main(case):

    if case == 1:
        mat = loadmat('kliep.mat')
        x_de = mat['x_de'].T
        x_nu = mat['x_nu'].T
        rand_index = mat['rand_index'].T.ravel() - 1

        kliep = KLIEP(seed=0)
        kliep.fit_CV(x_de, x_nu, rand_index=rand_index)

        w = kliep.predict(x_de)
        print w.sum()  # needs to be 100
    elif case == 2:
        x_de = np.random.multivariate_normal([0, 0], np.eye(2), 100)
        x_nu = np.random.multivariate_normal([0, 0], 0.5*np.eye(2), 100)

        kliep = KLIEP(seed=0)
        kliep.fit_CV(x_de, x_nu)

        w = kliep.predict(x_de)
        print w.sum()  # needs to be 100


if __name__ == '__main__':
    print 'kliep data...'
    main(1)
    print
    print 'dummy gaussians...'
    main(2)
