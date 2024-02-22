import numpy as np
from math import sin, cos, pi

# ConsCRLB(senPos, srcLoc, Q )
#
# Evaluation of the CRLB.
#
# Input:
#   senPos:	    (Dim x M), postions of reciveing sensors, each column is a sensor position
#               and the first column is the reference sensor location for TDOA.
#   srcLoc  :     (Dim x 1), source location.
#   Q:          ((M-1)x(M-1)), covariance matrix of TDOAs.
#
# Output:
#   CRB:        CRB matrix
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
#      Columbia, MO 65211, USA.
#
#       hod@missouri.edu
#

def ConsCRLB(senPos, srcLoc, Q):
    N = len(senPos)
    M = len(senPos[0])
    r = (np.sqrt(np.sum((senPos - srcLoc) ** 2, axis=0)).T)[:, np.newaxis]
    u0 = (srcLoc - senPos[:, 0][:, np.newaxis]) / r[0]
    g0 = 1 / r[0]

    p = (u0 - g0 * (senPos[:, 1:] - senPos[:, :1])) / np.sqrt(np.sum((u0 - g0 * (senPos[:, 1:] - senPos[:, :1])) ** 2, axis=0))
    # p = u0 - (senPos(:,2:end)-senPos(:,1))*g0
    P = p / np.sqrt(np.sum(p ** 2, axis=0))


    DF = np.zeros_like(senPos[:, 1:])
    DF[:N, :] = -(senPos[:, 1:] - senPos[:, :1]) / np.sqrt(
        np.sum((u0 - g0 * (senPos[:, 1:] - senPos[:, :1])) ** 2, axis=0))
    # DF(1:N,:) = -(senPos(:, 2:end) - senPos(:, 1))./ r(2: end)'*r(1)
    new_row = (- P.T @ u0 + np.ones((M - 1, 1))) * r[0] ** 2
    DF = np.vstack((DF, new_row.T))
    CRB1 = np.linalg.inv(DF @ np.linalg.inv(Q) @ DF.T)

    S = np.diag(np.concatenate((np.ones(N), [0])))
    Phio = np.vstack((u0, g0))
    F = Phio.T @ S
    CRB2 = CRB1 - CRB1 @ F.T @ np.linalg.inv(F @ CRB1 @ F.T) @ F @ CRB1

    # 2D
    if N == 2:
        doa = np.arctan2(srcLoc[1], srcLoc[0])
        D1 = np.array([[-sin(doa), cos(doa), 0],
                       [0, 0, 1]])
    # 3D
    elif N == 3:
        theta = np.arctan2(srcLoc[1] - senPos[1, 0], srcLoc[0] - senPos[0, 0])
        phi = np.arctan2(srcLoc[2] - senPos[2, 0], np.linalg.norm(srcLoc[0:2] - senPos[0:2, 0][:, np.newaxis],'fro'))
        D1 = np.array([[-sin(theta) / cos(phi), cos(theta) / cos(phi), 0, 0],
                       [-cos(theta) * sin(phi), -sin(theta) * sin(phi), cos(phi), 0],
                       [0, 0, 0, 1]])
    else:
        raise ValueError('Please check your input format of sensor positions')

    CRB = D1 @ CRB2 @ D1.T
    return CRB