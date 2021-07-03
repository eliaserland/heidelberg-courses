import numpy as np


def kalmanfilter(x, mu0, L0, A, B, Gamma, Sigma, C=None, u=None):
    """
    :param x: observed variables (p x n, with p= number obs., n= number time steps)
    :param mu0: initial values
    :param L0: initial values
    :param A: transition matrix
    :param B: observation matrix
    :param Gamma: observation covariance
    :param Sigma: transition covariance
    :param C: control variable matrix
    :param u: control variables
    :return: mu, V, K
    """
    p = Sigma.shape[0]
    q = x.shape[0]
    n = x.shape[1]

    # are control variables entered? if not, set to 0
    if C is None and u is None:
        C = np.zeros((p, q))
        u = np.zeros((q, n))

    # initialize variables for filter and smoother
    L = np.zeros((p, p, n))  # measurement covariance matrix
    L[:, :, 0] = L0  # prior covariance
    mu_p = np.zeros((p, n))  # predicted expected value
    mu_p[:, 0] = np.squeeze(mu0)  # prior expected value
    mu = np.zeros((p, n))  # filter expected value
    V = np.zeros((p, p, n))  # filter covariance matrix
    K = np.zeros((p, q, n))  # Kalman Gain

    # first step
    K[:, :, 0] = L[:, :, 0] @ B.T @ np.linalg.inv(B @ L[:, :, 0] @ B.T + Gamma)  # Kalman gain
    mu[:, 0] = mu_p[:, 0] + K[:, :, 0] @ (x[:, 0] - B @ mu_p[:, 0])
    V[:, :, 0] = (np.eye(p) - K[:, :, 0] @ B) @ L[:, :, 0]

    # go forwards
    for t in range(1, n):
        L[:, :, t] = A @ V[:, :, t - 1] @ A.T + Sigma
        K[:, :, t] = L[:, :, t] @ B.T @ np.linalg.inv(B @ L[:, :, t] @ B.T + Gamma)  # Kalman gain
        mu_p[:, t] = A @ mu[:, t - 1] + C @ u[:, t]  # model prediction
        mu[:, t] = mu_p[:, t] + K[:, :, t] @ (x[:, t] - B @ mu_p[:, t])  # filtered state
        V[:, :, t] = (np.eye(p) - K[:, :, t] @ B) @ L[:, :, t]  # filtered covariance
    return mu, V, K


# code originally in Matlab from:
# (c) 2017, Georgia Koppe, Dept. Theoretical Neuroscience, CIMH, Heidelberg University

# translation to Python:
# (c) 2019, Leonard Bereska, Dept. Theoretical Neuroscience, CIMH, Heidelberg University
# for comments, questions, errors, please contact Leonard.Bereska@zi-mannhiem.de
