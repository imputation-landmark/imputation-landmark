import numpy as np
import dataloader
from utils import *
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

class SMFL():

    def __init__(self, X, k, alpha, beta, lmd, iterations, p=5, cluster=True):

        self.X = X
        self.num_samples, self.num_features = X.shape
        self.X_ = self.replace_nan(np.zeros(self.X.shape))
        self.X2_ = self.X_[:,2:]
        self.p = p
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.lmd = lmd
        self.D = np.zeros((self.num_samples, self.num_samples))
        self.W = np.zeros((self.num_samples, self.num_samples))
        # preprocessing
        spatial_mat = self.X[:, :2]
        nbrs = NearestNeighbors(n_neighbors=p, algorithm='ball_tree').fit(spatial_mat)
        _, indices = nbrs.kneighbors(spatial_mat)
        for idx in range(self.num_samples):
            self.D[idx, indices[idx]] = 1
            self.D[indices[idx], idx] = 1

        for i in range(self.num_samples):
            self.W[i][i] = np.sum(self.D[i, :])
        self.L = self.W - self.D
        self.not_nan_index = (np.isnan(self.X) == False)
        self.r_omega = self.not_nan_index
        self.cluster = cluster
        if self.cluster:
            kmeans = KMeans(n_clusters=self.k, random_state=1)
            kmeans.fit(self.X[:,:2])
            self.cluster_centers = kmeans.cluster_centers_
        self.loss_record = []

    def geo_distance(self, i, j):
        dis_x = (self.X[i][0] - self.X[j][0]) ** 2
        dis_y = (self.X[i][1] - self.X[j][1]) ** 2
        return np.sqrt(dis_x + dis_y)

    def error(self):
        X_hat = np.matmul(self.U, self.V)
        res = np.nansum((self.X - X_hat) ** 2) + self.lmd * np.trace(np.matmul(np.matmul(self.U.T, self.L), self.U))
        return res

    def train(self, method=1):
        # Initialize factorization matrix U and V
        self.U = np.random.rand(self.num_samples, self.k)
        self.V = np.random.rand(self.k, self.num_features)
        self.V[:, :2] = self.cluster_centers
        self.V_ = self.V[:, 2:]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            # total square error
            e = self.error()
            training_process.append((i, e))
            if method == 1:
                self.update()
            elif method == 2:
                self.update2()
        return training_process

    def update(self):
        'sgd'
        a1 = np.matmul(self.X_, self.V.T)
        a2 = np.matmul(self.r_omega * np.matmul(self.U, self.V), self.V.T)
        a3 = self.lmd * np.matmul(self.L, self.U)
        U_new = self.U - self.alpha * (-2 * a1 +
                                    2 * a2 + 2 * a3)
        V_new = self.V - self.beta * (-2 * np.matmul(self.U.T, self.X_) +
                                   2 * np.matmul(self.U.T, self.r_omega * np.matmul(self.U, self.V)))
        self.U = U_new
        self.V[:, 2:] = V_new[:, 2:]

    def update2(self):
        'iteration'
        R_UV = self.r_omega * np.matmul(self.U, self.V)

        u1 = np.matmul(self.X_, self.V.T) + self.lmd * np.matmul(self.D, self.U)
        u2 = np.matmul(R_UV, self.V.T) + self.lmd * np.matmul(self.W, self.U)
        U_new = self.U * u1 / u2

        v1 = np.matmul(self.U.T, self.X2_)
        v2 = np.matmul(self.U.T, R_UV[:,2:])
        self.V_ = self.V_ * v1 / v2
        self.U = U_new
        self.V[:, 2:] = self.V_


    def get_x(self, i, j):
        """
        Get the predicted x of sample i and feature j
        """
        prediction = self.b + self.b_u[i] + self.b_v[j] + self.U[i, :].dot(self.V[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, U and V
        """
        return np.matmul(self.U, self.V) * (1-self.r_omega) + self.X_

    def replace_nan(self, X_hat):
        """
        Replace np.nan of X with the corresponding value of X_hat
        """
        X = np.copy(self.X)
        for i in range(self.num_samples):
            for j in range(self.num_features):
                if np.isnan(X[i, j]):
                    X[i, j] = X_hat[i, j]
        return X
