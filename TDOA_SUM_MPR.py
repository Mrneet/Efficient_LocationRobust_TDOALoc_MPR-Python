import numpy as np
from cmath import sqrt
from math import sin, cos, pi

def TDOA_SUM_MPR(senPos, rd, *args):
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
    inv_matrix = np.linalg.inv(np.transpose(G1) @ np.linalg.inv(Qr) @ G1)
    Phi1 = inv_matrix @ np.transpose(G1) @ np.linalg.inv(Qr) @ h1

    b = 1 + l ** 2 * Phi1[-1] ** 2 - 2 * Phi1[-1] * l * (s_bar.T @ Phi1[0:N])
    B1 = -np.diag(np.sqrt(b).flatten())

    T = np.zeros((M - 1, N * (M - 1)))
    for i in range(M - 1):
        value = (Phi1[:N] - (senPos[:, i + 1][:, np.newaxis] - senPos[:, 0][:, np.newaxis]) * Phi1[-1]).T
        start_idx = N * i
        end_idx = N + N * i
        T[i, start_idx:end_idx] = value.ravel()[:N * (i + 1)]
    ones_matrix = -np.ones((M - 1, 1))
    eye_matrix = np.eye(M - 1)
    C1 = -T @ np.kron(np.hstack((ones_matrix, eye_matrix)), np.eye(N))
    if isinstance(Qs, int):
        W1_inv = B1 @ Qr @ B1 + C1 * Qs @ C1.T
    else:
        W1_inv = B1 @ Qr @ B1 + C1 @ Qs @ C1.T
    G1_transpose = np.transpose(G1)
    term1 = np.linalg.inv(G1_transpose @ W1_inv @ G1)
    term2 = G1_transpose @ W1_inv @ h1
    Phi20 = term1 @ term2  # solution of 1st stage
    Phi2 = np.sign(np.real(Phi20)) * np.abs(Phi20)  # to keep it real

    # second stage
    diag_values = np.vstack((2 * Phi2[:N], [1]))
    B2 = np.diag(diag_values.flatten())
    h2 = np.vstack((Phi2[0:N - 1] ** 2, Phi2[N - 1] ** 2 - 1, Phi2[-1]))
    G2 = np.hstack((np.eye(N - 1), np.zeros((N - 1, 1))))
    G2 = np.vstack((G2, np.hstack((-np.ones((1, N - 1)), np.zeros((1, 1))))))
    G2 = np.vstack((G2, np.hstack((np.zeros((1, N - 1)), np.ones((1, 1))))))
    temp3 = np.dot(G1.T, np.linalg.inv(W1_inv))  # G1' * W1_inv
    temp4 = temp3 @ G1  # (G1' * W1_inv) * G1
    temp5 = np.linalg.solve(B2, temp4)
    W2 = np.dot(temp5, np.linalg.inv(B2))
    Psi0 = np.linalg.inv(G2.T @ W2 @ G2) @ G2.T @ W2 @ h2  # solution of 2nd stage
    Psi = np.sign(np.real(Psi0)) * np.abs(Psi0)  # to keep it real

    if N == 2:
        theta = np.arctan2(np.abs(np.sqrt(1-Psi[0])) * np.sign(Phi2[1]), np.sqrt(np.abs(Psi[0])) * np.sign(Phi2[0]))
        g = Psi[1]
        pos = [cos(theta), sin(theta)] / g + senPos[:, 0]

        return theta, g, pos

    elif N == 3:
        theta = np.arctan2(abs(sqrt(Psi[1])) * np.sign(Phi2[1]), abs(sqrt(Psi[0])) * np.sign(Phi2[0]))
        #theta = np.arctan2(np.abs(np.sqrt(np.maximum(Psi[1], 0))) * np.sign(Phi2[1]),np.abs(np.sqrt(np.maximum(Psi[0], 0))) * np.sign(Phi2[0]))
        # phi = math.atan2(abs(math.sqrt(1 - sum(Psi[0:2]))) * math.copysign(1, Phi2[2]), abs(math.sqrt(sum(Psi[0:2]))))
        phi = np.arctan2(abs(sqrt(1 - sum(Psi[0:2]))) * np.sign(Phi2[2]), abs(sqrt(sum(Psi[0:2]))))
        g = Psi[-1]
        pos = [cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi)] / g + senPos[:, 0]

        return theta, phi, g, pos

    else:
        raise ValueError('Please check your input format of sensor positions')
