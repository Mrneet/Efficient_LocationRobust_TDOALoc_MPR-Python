import numpy as np
from math import sin, cos, pi
from scipy.linalg import sqrtm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ConsCRLB import ConsCRLB
from TDOA_SUM_MPR import TDOA_SUM_MPR
from TDOA_SCO_MPR import TDOA_SCO_MPR
from TDOA_GTRS_MPR import TDOA_GTRS_MPR

np.random.seed(1)

senPos = np.array([[10.23, 38.38, 16.29],
                   [46.64, -87.12, 62.94],
                   [124.02, -7.98, 81.16],
                   [105.02, -51.72, 26.47],
                   [-81.56, 104.48, -80.49]]).T

# source direction
theta = 22.13 * np.pi / 180
phi = 14.41 * np.pi / 180

N = len(senPos)
M = len(senPos[0])
mon = 1000

# Monte-Carlo Simulation

nse = np.zeros((M - 1, mon))
err = np.zeros((M, N, mon))

aveNse = 0
for l in range(mon):
    nse[:, l][:, np.newaxis] = np.random.randn(M - 1, 1)
    err[:, :, l] = np.random.randn(M, N)
nse = nse - np.mean(nse, axis=1, keepdims=True)
err = err - np.mean(err, axis=2, keepdims=True)

aa = [1, 3, 7, 10, 4, 1, 9, 7, 2, 1, 3]
SS = np.kron(np.diag(aa[0:M]), np.eye(N))

print('Simulation is running ...')

nsePwr = np.array([-10])
errLvl = list(range(-40, 21, 5))
souRange = [15 * 1e2]

CRLB_a = np.zeros((len(errLvl), len(nsePwr)))
CRLB_g = np.zeros((len(errLvl), len(nsePwr)))

mse_a = np.zeros((len(errLvl), len(nsePwr), 3))
mse_g = np.zeros((len(errLvl), len(nsePwr), 3))
avBia_a = np.zeros((len(errLvl), len(nsePwr), 3))
avBia_g = np.zeros((len(errLvl), len(nsePwr), 3))

NumAlg = 3  # number of compared algorithms
tProc = [0] * NumAlg

for ir in range(len(souRange)):
    print('Range:', souRange[ir], 'm,', ir+1, '/', len(souRange), '...')

    # *************Generate Data***************
    # source location
    souLoc = souRange[ir] * np.reshape([cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi)], (3, 1)) + senPos[:, 0][:,np.newaxis]
    # true range
    r = (np.sqrt(np.sum((souLoc - senPos) ** 2, axis=0)).T)[:, np.newaxis]
    # true TDOAs
    rd = r[1:] - r[0]
    g = 1 / r[0]
    u0 = (souLoc - senPos[:, 0][:, np.newaxis]) / r[0]

    uTh = np.zeros((mon, mon, NumAlg))
    uPh = np.zeros((mon, mon, NumAlg))
    ug = np.zeros((mon, mon, NumAlg))
    ep = np.zeros((mon, mon, NumAlg))
    er = np.zeros((mon, mon, NumAlg))

    for is_ in range(len(errLvl)):
        print('10log(Error Level): ', errLvl[is_], ', ', is_ + 1, '/', len(errLvl), ' ...')

        #  Q = np.eye(M-1) * sigma(k) ** 2
        Qr = 10 ** (nsePwr / 10) * (np.ones((M - 1, M - 1)) + np.eye(M - 1)) / 2
        Qs = 10 ** (errLvl[is_] / 10) * SS
        Qsm = 10 ** (errLvl[is_] / 10) * np.diag(aa[0:M])

        b1 = senPos[:, 0][:, np.newaxis] - senPos[:, 1:]
        b2 = u0 + g * b1
        b3 = np.sum((b2) ** 2, axis=0)
        b4 = (np.sqrt((b3)).T)[:, np.newaxis]
        b = b4
        B = -np.diag(b.flatten())

        C = np.zeros((M - 1, N * M))
        for i in range(M - 1):
            C[:, :N] = -u0.T
            value = (u0 - (senPos[:, i + 1] - senPos[:, 0])[:, np.newaxis] * g).T
            start_idx = N * (i + 1)
            end_idx = N + N * (i + 1)
            C[i, start_idx:end_idx] = value.ravel()[:N * (i + 1)]
        Q1 = np.linalg.solve(B, C) @ Qs @ C.T
        Q = Qr + np.linalg.solve(B, Q1.T).T

        # Calculate CRLB
        CRB = ConsCRLB(senPos, souLoc, Q)
        CRLB_a[is_, ir] = CRB[0, 0] + CRB[1, 1]
        CRLB_g[is_, ir] = CRB[2, 2]

        # position and DOA estimation
        bia_p = np.zeros((N, mon, NumAlg))
        for i in range(mon):
            # measured TDOAs
            rd_m = rd + np.dot(sqrtm(Qr), nse[:, i][:, np.newaxis])
            senPos_m = senPos + err[:, :, i].T @ sqrtm(Qsm)

            nAg = 0
            # SCO-MPR Method
            mprSol = TDOA_SCO_MPR(senPos_m, rd_m, Qr, Qs)
            Th1 = mprSol[0]
            Ph1 = mprSol[1]
            g1 = mprSol[2]
            uTh[i, nAg, 0] = Th1[0]
            uPh[i, nAg, 0] = Ph1[0]
            ug[i, nAg, 0] = g1[0]

            # SUM-MPR Method
            nAg = nAg + 1
            Th2, Ph2, g2, _ = TDOA_SUM_MPR(senPos_m, rd_m, Qr, Qs)
            uTh[i, nAg, 0] = Th2[0]
            uPh[i, nAg, 0] = Ph2[0]
            ug[i, nAg, 0] = g2[0]

            # GTRS-MPR Method
            nAg = nAg + 1
            Th3, Ph3, g3, _ = TDOA_GTRS_MPR(senPos_m, rd_m, Qr, Qs)
            uTh[i, nAg, 0] = Th3[0]
            uPh[i, nAg, 0] = Ph3[0]
            ug[i, nAg, 0] = g3[0]

        # calculate MSE and bias
        for ia in range(nAg + 1):
            mse_a[is_, ir, ia] = np.mean((uTh[:, ia, 0] - theta) ** 2 + (uPh[:, ia, 0] - phi) ** 2)
            mse_g[is_, ir, ia] = np.mean((ug[:, ia, 0] - g) ** 2)

            avBia_a[is_, ir, ia] = np.sqrt(np.abs((np.mean(uTh[:, ia, 0]) - theta)) ** 2 + np.abs(np.mean(uPh[:, ia, 0]) - phi) ** 2)
            avBia_g[is_, ir, ia] = np.abs(np.mean(ug[:, ia, 0]) - g)

symbs = ['o', 'v', 's', '*', '^', '+', 'x']
name = ['SCO-MPR', 'SUM-MPR', 'GTRS-MPR']
xlabtext = r'$10log(\eta^2(m^2))$'
xdata = errLvl

# plot results
fig = plt.figure()
for ia in range(nAg+1):
    # 画点
    plt.plot(xdata, 10 * np.log10(mse_a[:, :, ia]), symbs[ia], linewidth=1.5, label=name[ia])
    # 显示网格
    plt.grid(True)
# 画线
plt.plot(xdata, 10 * np.log10(CRLB_a), '-', linewidth=1.5, label='CRLB')
# 图例
plt.legend(fontsize=11)
# x,y轴名称
plt.xlabel(xlabtext, fontsize=13)
plt.ylabel(r'$10log(MSE(\theta,\phi)(rad^2))$', fontsize=13)
# x,y轴坐标范围
plt.xlim([-40, 20])
plt.ylim([-50, 10])
# 显示图形
plt.show()

fig = plt.figure()
for ia in range(nAg+1):
    # 画点
    plt.plot(xdata, 10 * np.log10(mse_g[:, :, ia]), symbs[ia], linewidth=1.5, label=name[ia])
    # 显示网格
    plt.grid(True)
# 画线
plt.plot(xdata, 10 * np.log10(CRLB_g), '-', linewidth=1.5, label='CRLB')
# 图例
plt.legend(fontsize=11)
# x,y轴名称
plt.xlabel(xlabtext, fontsize=13)
plt.ylabel(r'$10log(MSE(g)(1/m^2))$', fontsize=13)
# x,y轴坐标范围
plt.xlim([-40, 20])
plt.ylim([-90, -20])
# 显示图形
plt.show()

# bias
fig = plt.figure()
for ia in range(nAg+1):
    # 画点
    plt.plot(xdata, 20 * np.log10(avBia_a[:, :, ia]), symbs[ia], linewidth=1.5, label=name[ia])
    # 显示网格
    plt.grid(True)
# 图例
plt.legend(fontsize=11)
# x,y轴名称
plt.xlabel(xlabtext, fontsize=13)
plt.ylabel(r'$20log(Bias(\theta,\phi)(rad))$', fontsize=13)
# x,y轴坐标范围
plt.xlim([-40, 20])
plt.ylim([-100, 20])
# 显示图形
plt.show()

fig = plt.figure()
for ia in range(nAg+1):
    # 画点
    plt.plot(xdata, 20 * np.log10(avBia_g[:, :, ia]), symbs[ia], linewidth=1.5, label=name[ia])
    # 显示网格
    plt.grid(True)
# 图例
plt.legend(fontsize=11)
# x,y轴名称
plt.xlabel(xlabtext, fontsize=13)
plt.ylabel(r'$20log(Bias(g)(m^{-1}))$', fontsize=13)
# x,y轴坐标范围
plt.xlim([-40, 20])
plt.ylim([-180, -40])
# 显示图形
plt.show()