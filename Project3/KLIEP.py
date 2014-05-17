from __future__ import division
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from sklearn.cross_validation import KFold
from sklearn.neighbors import BallTree


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

    def KC(self, X, M):
        return cdist(X, M, 'cosine')

    def learning(self, b_l, A_jl, verbose=False):
        c = np.dot(b_l.T, b_l)
        # initial alphas
        alpha = np.ones((self.b, 1))
        score_old = -np.inf

        final_alpha = np.empty_like(alpha)
        tmp_alpha = np.empty_like(alpha)
        for epsilon in 10**np.asarray(range(3, -4, -1), dtype=float):
            for i in xrange(self.max_iter):
                np.copyto(tmp_alpha, alpha)
                # gradient ascent
                alpha = alpha + epsilon*np.dot(A_jl.T, np.dot(A_jl, alpha) ** -1)
                # we use pinv since we cannot always compute the inverse
                # enfore constraints
                alpha = alpha + b_l * (1 - np.dot(b_l.T, alpha))*np.linalg.pinv(c)
                alpha = np.maximum(np.zeros(alpha.shape), alpha)
                alpha = np.dot(alpha, np.linalg.pinv(np.dot(b_l.T, alpha)))
                score = np.mean(np.log(np.dot(A_jl, alpha)))

                if score - score_old <= 0:
                    np.copyto(final_alpha, tmp_alpha)
                    break

                if verbose:
                    print 'Score at iter', i+1, ':', score
                score_old = score

        if verbose:
            print 'Final score:', score_old
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
        b_l = np.mean(self.KG(x_tr, k_cent, self.sigma), axis=0).reshape((self.b, 1))

        self.alpha = self.learning(b_l, A_jl, verbose=True)

    # use cosine kernel
    def fit_cosine(self, x_tr, x_te, rand_index=None):
        if rand_index is None:
            rand_index = np.random.permutation(x_te.shape[0])
        # get the minimum b
        self.b = min(self.init_b, x_te.shape[0])
        # parse the centers (for the basis functions kernels)
        k_cent = x_te[rand_index[0:self.b], :]
        # for predicting later
        self.centers = k_cent.copy()

        # try also different kernels?
        A_jl = self.KC(x_te, k_cent)
        b_l = np.mean(self.KC(x_tr, k_cent), axis=0).reshape((self.b, 1))
        self.alpha = self.learning(b_l, A_jl, verbose=True)

    # unfinished
    def find_NN_basis(x_tr, x_te, leaf_size):
        bt = BallTree(x_tr, leaf_size=30, metric='euclidean')
        indices = bt.query(x_te, k=1, return_distance=False)
        uniq_val = np.unique(indices)
        return x_te[uniq_val, :]

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

                b_l = np.mean(self.KG(x_tr, k_cent, sigma_new), axis=0).reshape((self.b, 1))

                kf = KFold(x_te.shape[0], n_folds=n_folds, shuffle=True, indices=False)
                for train, test in kf:
                    A_jl = self.KG(x_te[train], k_cent, sigma_new)
                    alpha_cv = self.learning(b_l, A_jl)
                    wh_cv = np.dot(self.KG(x_te[test], k_cent, sigma_new), alpha_cv)
                    score_new = score_new + np.mean(np.log(wh_cv))/(n_folds)

                if score_new - score <= 0:
                    break
                score = score_new
                sigma = sigma_new
                print 'score:', score, ', sigma:', sigma

        print 'Best sigma:', sigma
        self.sigma = sigma
        b_l = np.mean(self.KG(x_tr, k_cent, sigma), axis=0).reshape((self.b, 1))
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
        print w.shape
    elif case == 2:
        x_de = np.random.multivariate_normal([0, 0], np.eye(2), 100)
        x_nu = np.random.multivariate_normal([0, 0], 0.5*np.eye(2), 100)

        kliep = KLIEP(seed=0)
        kliep.fit_CV(x_de, x_nu, n_folds=5)

        w = kliep.predict(x_de)
        print w.sum()  # needs to be 100
        print w.shape


if __name__ == '__main__':
    print 'kliep data...'
    main(1)
    print
    print 'dummy gaussians...'
    main(2)
