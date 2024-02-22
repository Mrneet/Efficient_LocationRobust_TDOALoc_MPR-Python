import numpy as np
from math import sin, cos, pi

#
# Estimation of the source location in MPR by the GTRS method, s1 not
# necessarily in the origin.
#
# Input:
#   senPos:	    (Dim x M), postions of reciveing sensors, each column is a sensor position
#               and the first column is the reference sensor location for TDOA.
#   rd:         ((M-1)x1), TDOA measurement vector.
#   Q:          ((M-1)x(M-1)), covariance matrix of TDOAs.
#
# Output:
#   varargout: containing
#     **Dim == 2:
#       theta:	(1x2), DOA estimation, including 1st & 2nd stages.
#       g:      (1x2), inverse-range (g) estimation, including 1st & 2nd stages.
#       pos:	(1x2), source position, only 2nd stage.
#     **Dim == 3:
#       theta:	(1x2), azimuth estimation, including 1st & 2nd stage.
#       phi:	(1x2), elevation estimation, including 1st & 2nd stage.
#       g:      (1x2), g estimation, include 1st & 2nd stage.
#       pos:	(1x3), source position, only 2nd stage.
#
# Reference: Y. Sun, K. C. Ho, G. Wang. J. Chen, Y. Yang, L. Chen, and Q. Wan,
# "Computationally attractive and location robust estimator for IoT device positioning,"
# IEEE Internet Things J., Nov. 2021.
#
# Yimao Sun and K. C. Ho   04-08-2022
#
#       Copyright (C) 2022
#       Computational Intelligence Signal Processing Laboratory
#       University of Missouri
#       Columbia, MO 65211, USA.
#       hod@missouri.edu
#

def TDOA_GTRS_MPR(senPos, rd, *args):
    N = len(senPos)
    M = len(senPos[0])

    Qr = args[0]
    Qs = 0
    if len(args) == 2:
        Qs = args[1]

    h1 = -rd
    l = (np.sqrt(np.sum((senPos[:, 1:] - senPos[:, 0][:, np.newaxis]) ** 2, axis=0)).T)[:, np.newaxis]
    s_bar = (senPos[:, 1:] - senPos[:, 0][:, np.newaxis]) / l.T

    # first stage
    temp1 = (senPos[:, 1:] - senPos[:, 0][:, np.newaxis]).T
    temp2 = 0.5 * (rd ** 2 - (np.sum(((senPos[:, 1:] - senPos[:, 0][:, np.newaxis]).T) ** 2, axis=1))[:, np.newaxis])
    G1 = np.concatenate((temp1, temp2), axis=1)
    B1 = np.eye(M - 1)
    W = np.eye(M - 1) @ np.linalg.inv(B1 @ Qr @ B1)

    for iter in range(2):
        A = G1[:, 0:N]
        a = G1[:, N][:, np.newaxis]
        O = W - W @ a @ a.T @ W / (a.T @ W @ a)
        AOA = A.T @ O @ A
        S = np.eye(N)

        U, singular_values, _ = np.linalg.svd(AOA)
        D = np.zeros_like(AOA)
        D[np.diag_indices(min(AOA.shape))] = singular_values
        r = np.diag(D)[:, np.newaxis]
        k = U.T @ A.T @ O @ h1

        if N == 2:
            x = np.zeros(5)
            x[0] = 1
            x[1] = 2 * r[0] + 2 * r[1]
            x[2] = r[0] ** 2 + 2 * r[0] * r[1] + r[1] ** 2 - k[0] ** 2 - k[1] ** 2
            x[3] = 2 * r[0] ** 2 * r[1] + 2 * r[0] * r[1] ** 2 - 2 * k[0] ** 2 * r[1] - 2 * k[1] ** 2 * r[0]
            x[4] = r[0] ** 2 * r[1] ** 2 - k[0] ** 2 * r[1] ** 2 - k[1] ** 2 * r[0] ** 2

        elif N == 3:
            x = np.zeros(7)
            x[0] = 1
            x[1] = 2 * r[2] + 2 * r[1] + 2 * r[0]
            x[2] = r[2] ** 2 + 4 * r[1] * r[2] + 4 * r[0] * r[2] + r[1] ** 2 + 4 * r[0] * r[1] + r[0] ** 2 - k[2] ** 2 - k[1] ** 2 - k[0] ** 2
            x[3] = 2 * r[1] * r[2] ** 2 + 2 * r[0] * r[2] ** 2 + 2 * r[1] ** 2 * r[2] + 8 * r[0] * r[1] * r[2] + 2 * r[0] ** 2 * r[2] - 2 * k[1] ** 2 * r[2] - 2 * k[0] ** 2 * r[2] + 2 * r[0] * r[1] ** 2 + 2 * r[0] ** 2 * r[1] - 2 * k[2] ** 2 * r[1] - 2 * k[0] ** 2 * r[1] - 2 * k[2] ** 2 * r[0] - 2 * k[1] ** 2 * r[0]
            x[4] = r[1] ** 2 * r[2] ** 2 + 4 * r[0] * r[1] * r[2] ** 2 + r[0] ** 2 * r[2] ** 2 - k[1] ** 2 * r[2] ** 2 - k[0] ** 2 * r[2] ** 2 + 4 * r[0] * r[1] ** 2 * r[2] + 4 * r[0] ** 2 * r[1] * r[2] - 4 * k[0] ** 2 * r[1] * r[2] - 4 * k[1] ** 2 * r[0] * r[2] + r[0] ** 2 * r[1] ** 2 - k[2] ** 2 * r[1] ** 2 - k[0] ** 2 * r[1] ** 2 - 4 * k[2] ** 2 * r[0] * r[1] - k[2] ** 2 * r[0] ** 2 - k[1] ** 2 * r[0] ** 2
            x[5] = 2 * r[0] * r[1] ** 2 * r[2] ** 2 + 2 * r[0] ** 2 * r[1] * r[2] ** 2 - 2 * k[0] ** 2 * r[1] * r[2] ** 2 - 2 * k[1] ** 2 * r[0] * r[2] ** 2 + 2 * r[0] ** 2 * r[1] ** 2 * r[2] - 2 * k[0] ** 2 * r[1] ** 2 * r[2] - 2 * k[1] ** 2 * r[0] ** 2 * r[2] - 2 * k[2] ** 2 * r[0] * r[1] ** 2 - 2 * k[2] ** 2 * r[0] ** 2 * r[1]
            x[6] = r[0] ** 2 * r[1] ** 2 * r[2] ** 2 - k[0] ** 2 * r[1] ** 2 * r[2] ** 2 - k[1] ** 2 * r[0] ** 2 * r[2] ** 2 - k[2] ** 2 * r[0] ** 2 * r[1] ** 2

        root = np.roots(x)

        # delete complex roots
        reRoot = np.real(root[root.imag == 0])
        L = len(reRoot)
        # guarantee that Y is not empty
        if L == 0:
            I = np.imag(root).argmin()
            reRoot = np.real(root[I])
            L = 1

        Y = np.zeros((N + 1, L))
        J = np.zeros((1, L))
        for i in range(L):
            Y[:N, i][:, np.newaxis] = np.linalg.inv(AOA + reRoot[i] * S) @ (A.T @ O @ h1)
            Y[N, i] = np.linalg.inv(a.T @ W @ a) @ a.T @ W @ (h1 - A @ Y[:N, i][:, np.newaxis])
            J[0, i] = (h1 - G1 @ Y[:, i][:, np.newaxis]).T @ W @ (h1 - G1 @ Y[:, i][:, np.newaxis])
        ind = np.argmin(J)
        Phi0 = Y[:, ind][:, np.newaxis]
        Phi = np.sign(np.real(Phi0)) * np.abs(Phi0)

        b = 1 + l ** 2 * Phi[-1] ** 2 - 2 * Phi[-1] * l * (s_bar.T @ Phi[0:N])
        B1 = -np.diag(np.sqrt(b).flatten())
        T = np.zeros((M - 1, N * (M - 1)))
        for i in range(M - 1):
            value = (Phi[:N] - (senPos[:, i + 1][:, np.newaxis] - senPos[:, 0][:, np.newaxis]) * Phi[-1]).T
            start_idx = N * i
            end_idx = N + N * i
            T[i, start_idx:end_idx] = value.ravel()[:N * (i + 1)]
        ones_matrix = -np.ones((M - 1, 1))
        eye_matrix = np.eye(M - 1)
        C1 = -T @ np.kron(np.hstack((ones_matrix, eye_matrix)), np.eye(N))
        if isinstance(Qs, int):
            W = np.eye(M - 1) @ np.linalg.inv(B1 @ Qr @ B1 + C1 * Qs @ C1.T)
        else:
            W = np.eye(M - 1) @ np.linalg.inv(B1 @ Qr @ B1 + C1 @ Qs @ C1.T)

    if N == 2:
        theta = np.arctan2(Phi[1], Phi[0])
        g = Psi[2]
        pos = [cos(theta), sin(theta)] / g + senPos[:, 0]

        return theta, g, pos

    elif N == 3:
        theta = np.arctan2(Phi[1], Phi[0])
        phi = np.arctan2(Phi[2], np.sqrt(Phi[0] ** 2 + Phi[1] ** 2))
        g = Phi[3]
        pos = [cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi)] / g + senPos[:, 0]

        return theta, phi, g, pos

    else:
        raise ValueError('Please check your input format of sensor positions')