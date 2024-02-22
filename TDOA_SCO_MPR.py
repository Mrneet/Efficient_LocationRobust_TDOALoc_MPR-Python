import numpy as np
import cmath

# [mprSol,pos] = TDOA_SCO_MPR(senPos, rd, varargin)
#
# Closed-form method for localization in MPR using successive constrained
# optimization, s1 not necessarily in the origin.
#
# Input:
#   senPos:     (Dim x M), postions of reciveing sensors, each column is a sensor position
#               and the first column is the reference sensor location for TDOA.
#   rd:         ((M-1)x1), TDOA measurement vector.
#   Q:          ((M-1)x(M-1)), covariance matrix of TDOAs.
#
# Output:
#   varargout: including
#     **Dim == 2:
#       theta:	(1x2), DOA estimation, including 1st & 2nd stage.
#       g:      (1x2), inverse-range (g) estimation, including 1st & 2nd stage.
#     **Dim == 3:
#       theta:	(1x2), azimuth estimation, including 1st & 2nd stage.
#       phi:	(1x2), elevation estimation, including 1st & 2nd stage.
#       g:      (1x2), g estimation, including 1st & 2nd stage.
#       pos:    (1xN), source position, only 2nd stage.
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

def TDOA_SCO_MPR(senPos, rd, *args):
    N = len(senPos)
    M = len(senPos[0])

    Qr = args[0]
    Qs = 0
    if len(args) == 2:
        Qs = args[1]

    h1 = -rd
    v1 = 0.5 * (rd ** 2 - np.sum((senPos[:, 1:] - senPos[:, 0][:, np.newaxis]).T ** 2, axis=1)[:, np.newaxis])

    # first stage
    G1 = (senPos[:, 1:] - senPos[:, 0][:, np.newaxis]).T
    W1 = np.linalg.inv(Qr)
    for itNum in range(2):
        iG1 = np.linalg.inv(G1.T @ W1 @ G1) @ (G1.T @ W1)
        alpha = iG1 @ h1
        beta = -iG1 @ v1

        x = np.zeros(3)
        x[0] = beta.T @ beta
        x[1] = 2 * (beta.T @ alpha)
        x[2] = alpha.T @ alpha - 1
        root = [-x[1] + cmath.sqrt(x[1] ** 2 - 4 * x[0] * x[2]), -x[1] - cmath.sqrt(x[1] ** 2 - 4 * x[0] * x[2])] / (2 * x[0])

        # delete complex roots
        reRoot = np.real(root[root.imag == 0])
        L = len(reRoot)

        # guarantee that Y is not empty
        if L == 0:
            I = np.imag(root).argmin()
            reRoot = np.real(root[I])
            L = 1

        # find property u_bar and g
        u_tmp = np.zeros((N, L))
        J = np.zeros((L, 1))
        reRoot = np.atleast_1d(reRoot)
        for i in range(L):
            u_tmp[:, i] = alpha[:, 0] + beta[:, 0] * reRoot[i]
            # r_rec = ((np.sqrt(np.sum(u_tmp[:, i][:, np.newaxis] - reRoot[i] * (senPos[:, :] - senPos[:, 0][:, np.newaxis]) ** 2, axis=0)).T)[:, np.newaxis]) / reRoot[i]
            r_rec1 = u_tmp[:, i][:, np.newaxis] - reRoot[i] * (senPos[:, :] - senPos[:, 0][:, np.newaxis])
            r_rec2 = np.sum((r_rec1) ** 2, axis=0)
            r_rec3 = (np.sqrt((r_rec2)).T)[:, np.newaxis] / reRoot[i]
            r_rec = r_rec3
            rd_rec = r_rec[1:] - r_rec[0]
            J[i] = ((rd - rd_rec).T @ np.linalg.inv(Qr) @ (rd - rd_rec))
        ind = np.argmin(J)
        ubar1 = u_tmp[:, ind][:, np.newaxis]
        g1 = reRoot[ind]

        h2 = -rd - G1 @ ubar1 - v1 * g1
        G3 = -G1
        v2 = -v1
        O = W1 - (W1 @ v2) / (v2.T @ W1 @ v2) * v2.T @ W1
        iG2 = np.linalg.inv(G3.T @ O @ G3)
        lambda_ = - (ubar1.T @ iG2 @ G3.T @ O @ h2) / (ubar1.T @ iG2 @ ubar1)
        delta_u = iG2 @ (G3.T @ O @ h2 + lambda_ * ubar1)
        delta_g = np.linalg.inv(v2.T @ W1 @ v2) @ (v2.T @ W1 @ (h2 - G3 @ delta_u))

        psi2 = np.vstack((ubar1, g1)) - np.vstack((delta_u, delta_g))

        # update weighting matrix
        # b = np.sqrt(np.sum((psi2[:N] + psi2[N:] * (senPos[:, 0] - senPos[:, 1:])) ** 2, axit=0)).T
        b1 = senPos[:, 0][:, np.newaxis] - senPos[:, 1:]
        b2 = psi2[-1] * b1
        b3 = psi2[:N] + b2
        b4 = np.sum((b3) ** 2, axis=0)
        b5 = (np.sqrt((b4)).T)[:, np.newaxis]
        b = b5
        B1 = -np.diag(b.flatten())

        C1 = np.zeros((M - 1, N * M))
        for i in range(M - 1):
            C = np.zeros((i + 1, N))
            C[:, :N] = -psi2[:N].T
            value = (psi2[:N] - (senPos[:, i + 1] - senPos[:, 0])[:, np.newaxis] * psi2[-1]).T
            start_idx = N * (i + 1)
            end_idx = N + N * (i + 1)
            C1[i, start_idx:end_idx] = value.ravel()[:N * (i + 1)]
        if isinstance(Qs, int):
            W1 = np.linalg.inv((B1 @ Qr @ B1 + C1 * Qs @ C1.T))
        else:
            W1 = np.linalg.inv((B1 @ Qr @ B1 + C1 @ Qs @ C1.T))

    psi2 = np.sign(np.real(psi2)) * np.abs(psi2)
    mprSol = np.vstack((np.arctan2(psi2[1], psi2[0]),
                        np.arctan2(psi2[2], np.linalg.norm(psi2[:2])),
                        psi2[N]))
    pos = psi2[0:N] / psi2[N] + senPos[:, 0][:, np.newaxis]

    return mprSol