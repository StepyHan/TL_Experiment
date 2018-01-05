from JDA import myJDA
import numpy as np
from sklearn import svm




# x_0HP, y_0HP = CWRU_Data.get_data('CWRU_smaller/0HP', one_hot=False)
# x_3HP, y_3HP = CWRU_Data.get_data('CWRU_smaller/3HP', one_hot=False)
#
# x_0HP = x_0HP[0:2400, :]
# y_0HP = y_0HP[0:2400, :]
# x_3HP = x_3HP[0:2400, :]
# y_3HP = y_3HP[0:2400, :]
#
#
# X_src = x_0HP        # ns * n_feature
# Y_src = y_0HP        # ns * 1
# X_tar = x_3HP        # nt * n_feature
# Y_tar = y_3HP        # nt * 1
Xs = np.array([[1,2,3,7], [2,1,3,7],[4,1,3,7], [3,2,1,7], [88,99,66,7], [44,33,66,7],[44,66,88,7]])
Xt = np.array([[99,2,3,3], [153,9,3,64],[123,4,3,86], [123,2,1,3], [88,99,66,97], [44,33,66,42],[44,66,88,86]])

ys = np.array([[1],[1],[1],[1],[2],[2],[2]])
yt = np.array([[2],[2],[2],[2],[1],[1],[1]])

class options(object):
    def __init__(self, kernel, gamma, lam, dim, epoch):
        self.kernel = kernel
        self.gamma = gamma
        self.lam = lam
        self.dim = dim
        self.epoch = epoch
options = options('rbf', 0.1, 1, 3, 100)
A, _ = myJDA(Xs, ys, Xt, yt, options)

# X = np.matrix(np.hstack((Xs.T, Xt.T)))
# # kernel_jda('rbf', np.matrix(Xs), 1)
# n1sq = np.sum(np.square(X), axis=0)
# n1 = X.shape[1]
# D = np.tile(n1sq, (n1, 1)) + np.tile(n1sq, (n1, 1)).T - 2 * X.T * X
# K = np.exp(-1 * D)

# import numpy as np
# from scipy.linalg import norm
# from scipy.linalg import eig
# from sklearn import svm
#
# D, V = np.linalg.eig(np.dot(Xt.I, Xs))
# eig_values = D.reshape(len(D), 1)
# eig_values_sorted = np.sort(eig_values, axis=0)
# index_sorted = np.argsort(eig_values, axis=0)
# V = V[:, index_sorted].reshape
# Z = np.dot(V.T, X)
