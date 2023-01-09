import SMFL
from utils import *
import time

parameters = {
        'economic': {
            'k': 11,
            'lmd': 0.3,
            'iter': 300,
            'p': 5,
        },
        'farm': {
            'k': 10,
            'lmd': 0.95,
            'iter': 700,
            'p': 5,
        },
        'lake': {
            'k': 8,
            'lmd': 0.8,
            'iter': 300,
            'p': 5,
        },
        'vehicle': {
            'k': 9,
            'lmd': 0.07,
            'iter': 500,
            'p': 5,
        },
    }

def rmse(mat, imputed_mat, mask):
    error = 0
    count = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mask[i, j]:
                error += pow(mat[i, j] - imputed_mat[i, j], 2)
                count += 1

    if count == 0:
        return np.nan
    return np.sqrt(error / count)


def geo_distance(X, i, j):
    dis_x = (X[i][0] - X[j][0]) ** 2
    dis_y = (X[i][1] - X[j][1]) ** 2
    return np.sqrt(dis_x + dis_y)

def sparseList(mat,threshold, num):
    n = mat.shape[0]
    points = []
    for i in range(0, n):
        count = 0
        for j in range(0, n):
            dis = geo_distance(mat, i, j)
            if dis < threshold:
                 count += 1
        if count < num:
            points.append(i)
    return points

def SMFTest(mat, mat_, lmd, landmarkrow=[], method=1, k=6, iter=500, p=5, return_v= False):
    smf = SMF.SMF(mat_, k=k, alpha=0.001, beta=0.001, iterations=iter, lmd=lmd, p=p)
    start = time.time()
    training_process = smf.train(method=method)
    end = time.time()
    X_hat = smf.full_matrix()
    X_comp = smf.replace_nan(X_hat)
    mask = np.isnan(mat_)
    if return_v:
        return rmse(X_comp, mat, mask), X_comp, training_process, (end - start) * 1000, smfl.V
    return rmse(X_comp, mat, mask), X_comp, training_process, (end-start) * 1000

def SMFLTest(mat, mat_, lmd, landmarkrow=[], method=1, k=6, iter=500, cluster=False, p=5, return_v= False):
    smfl = SMFL.SMFL(mat_, k=k, alpha=0.001, beta=0.001, iterations=iter, lmd=lmd, cluster=cluster, p=p)
    start = time.time()
    training_process = smfl.train(method=method)
    end = time.time()
    X_hat = smfl.full_matrix()
    X_comp = smfl.replace_nan(X_hat)
    mask = np.isnan(mat_)
    if return_v:
        return rmse(X_comp, mat, mask), X_comp, training_process, (end - start) * 1000, smfl.V
    return rmse(X_comp, mat, mask), X_comp, training_process, (end-start) * 1000

