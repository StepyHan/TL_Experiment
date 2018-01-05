import numpy as np
from scipy.linalg import norm
from numpy.linalg import eig
from sklearn import svm
from sklearn import neighbors

'''
X_src: ns行*n列features
X_tar: nt行*n列features
Y_src: ns * 1
Y_tar_pseudo: nt * 1
'''
# dim = 2
# lam = 2
#
# X_src = np.array([[1,2,3], [2,3,4], [3,4,5], [5,6,7]])        # ns == 4  n_features == 3
# X_tar = np.array([[7,8,9], [9,10,11], [11,12,14], [17,31,41]])        # nt == 4  n_features == 3
# Y_src = np.array([[1], [2], [1],[2]])
# Y_tar_pseudo = np.array([[1], [1], [1], [2]])
# X_src.size

def JDA_core(X_src, Y_src, X_tar, Y_tar_pseudo, options):
    '''
    :param X_src: ns行 * n_features列
    :param Y_src: ns * 1
    :param X_tar: nt行 * n_features列
    :param Y_tar_pseudo: nt * 1
    :param options: class contain kernel, lambda, dim, gamma for rbf
    :return: A: mapping matrix (prime_shape=[n_feature*dim], kernel_shape=[ns+nt*dim]);
             Z: feature space after mapping (shape=[dim*ns+nt])
    '''

    lam = options.lam
    dim = options.dim
    kernel = options.kernel
    gamma = options.gamma

    X = np.hstack((X_src.T, X_tar.T))
    m, n = X.shape      # n == ns + nt, m == n_features
    ns = X_src.shape[0]
    nt = X_tar.shape[0]
    e = np.vstack((1 / ns * np.ones([ns, 1]), - 1 / nt * np.ones([nt, 1])))
    C = np.unique(Y_src).shape[0]
    M = np.dot(e, e.T) * C      # M0

    # Mc
    N = 0
    if Y_tar_pseudo.size != 0:
        for c in np.unique(Y_src):
            e = np.zeros((n, 1))
            if Y_src[Y_src == c].shape[0] != 0:
                e[0: ns][Y_src == c] = 1 / Y_src[Y_src == c].shape[0]

            if Y_tar_pseudo[Y_tar_pseudo == c].shape[0] != 0:
                e[ns:][Y_tar_pseudo == c] = -1 / Y_tar_pseudo[Y_tar_pseudo == c].shape[0]

            N = N + np.dot(e, e.T)

    M = M + N
    M = M / norm(M, 'fro')
    H = np.eye(n) - 1 / n * np.ones((n, n))

    H = np.mat(H)
    M = np.mat(M)
    X = np.mat(X)

    if kernel == 'primal':
        D, V = eig((X*H*(X.T)).I * (X*M*(X.T)+lam*np.eye(m)))      # TODO REFER TO TCA
        eig_values = D.reshape(len(D), 1)
        index_sorted = np.argsort(eig_values, axis=0)   # ASCENDING
        V = V[:, index_sorted]
        V = V.reshape((V.shape[0], V.shape[1]))
        A = V[:, 0: dim]
        Z = np.dot(A.T, X)
    else:
        K = kernel_jda(kernel, X, gamma)
        D, V = eig((K*H*(K.T)).I * (K*M*(K.T)+lam*np.eye(n)))     # TODO REFER TO TCA
        eig_values = D.reshape(len(D), 1)
        index_sorted = np.argsort(eig_values, axis=0)
        V = V[:, index_sorted]
        V = V.reshape((V.shape[0], V.shape[1]))
        A = V[:, 0: dim]
        Z = np.dot(A.T, K)

    return A, Z


def kernel_jda(kernel, X, gamma):
    '''
    :param kernel: string, contain 'linear' and 'rbf'
    :param X: n_features*ns+nt
    :param gamma: gamma for rbf
    :return: kernel matrix
    '''
    if kernel == 'linear':
        K = X.T * X
    elif kernel == 'rbf':
        n1sq = np.sum(np.square(X), axis=0)
        n1 = X.shape[1]
        D = np.tile(n1sq, (n1, 1)) + np.tile(n1sq, (n1, 1)).T - 2 * X.T * X
        K = np.exp(-gamma * D)
    else:
        raise Exception('Not support kernel: %s' % kernel)

    return K

def myJDA(X_src, Y_src, X_tar, Y_tar, options):
    '''
    :param X_src: ns * n_feature
    :param Y_src: ns * 1
    :param X_tar: nt * n_feature
    :param Y_tar: nt * 1
    :param options: class options
    :return: A: n_feature*m, each column of A is eigenvector(m smallest). A.T*X is the result: m*(ns+nt)==>primal
    '''
    epoch = options.epoch
    Y_tar_pseudo = np.array([])

    for i in range(epoch):
        A, Z = JDA_core(X_src, Y_src, X_tar, Y_tar_pseudo, options)
        Z = Z * np.diag(np.array(1 / np.sum(np.square(Z), axis=0)).reshape(-1))
        Zs = Z[:, 0: X_src.shape[0]]    # dim * ns
        Zt = Z[:, X_src.shape[0]: ]     # dim * nt

        # Train classifier: SVM
        # clf = svm.SVC()
        # clf.fit(Zs.T, Y_src.reshape(-1))
        clf = neighbors.KNeighborsClassifier(n_neighbors=1)
        clf.fit(Zs.T, Y_src.reshape(-1))
        Y_src_pseudo = clf.predict(Zs.T).reshape((-1, 1))
        acc = Y_src[Y_src == Y_src_pseudo].size / Y_src.size
        print('SVM accuracy is %.4f' % acc)

        Y_tar_pseudo = clf.predict(Zt.T).reshape((-1, 1))
        acc = Y_tar[Y_tar == Y_tar_pseudo].size / Y_tar.size
        print('After %d steps, SVM & JDA`s accuracy is %.4f' % (i+1, acc))
    return A, Z