import numpy as np
from math import sin, cos, pi
import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from TDOA_SCO_MPR import TDOA_SCO_MPR

np.random.seed(1)

senPos = np.array([[10.23, 38.38, 16.29],
                   [46.64, -87.12, 62.94],
                   [124.02, -7.98, 81.16],
                   [105.02, -51.72, 26.47],
                   [-81.56, 104.48, -80.49]]).T

# target direction
theta = 22.13 * np.pi / 180
phi = 14.41 * np.pi / 180

N = len(senPos)
M = len(senPos[0])
mon = 1000

tmpNsed = np.zeros((M, mon))
tmpMp = np.zeros((M, mon))
for l in range(mon):
    tmpNsed[:, l] = np.random.randn(M)
aveNse = np.mean(tmpNsed, axis=1)[:, np.newaxis] / np.sqrt(2)
PP = aveNse[1:] - aveNse[0]

for l in range(mon):
    tmpMp[:, l] = np.random.randn(M)

# Monte-Carlo Simulation
print('Simulation is running ...')

# ******* vs. noise power config *******
nsePwr = np.arange(-70, 30, 10)
souRange = [15 * 1e2]

mse_a = np.zeros((len(nsePwr), len(souRange), 3, 1))
mse_g = np.zeros((len(nsePwr), len(souRange), 3, 1))
avBia_a = np.zeros((len(nsePwr), len(souRange), 3, 1))
avBia_g = np.zeros((len(nsePwr), len(souRange), 3, 1))


NumAlg = 1  # number of compared algorithms

for ir in range(len(souRange)):
    print("Range: ", souRange[ir], ",", ir+1, "/", len(souRange), "...")
    # *************Generate Data***************
    # source location
    souLoc = souRange[ir] * np.reshape([cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi)], (3, 1)) + senPos[:, 0][:, np.newaxis]
    # true range
    r = (np.sqrt(np.sum((souLoc - senPos) ** 2, axis=0)).T)[:, np.newaxis]
    # true TDOAs
    rd = r[1:] - r[0]
    g = 1 / r[0]

    uTh = np.zeros((mon, mon, NumAlg))
    uPh = np.zeros((mon, mon, NumAlg))
    ug = np.zeros((mon, mon, NumAlg))
    ep = np.zeros((mon, mon, NumAlg))
    er = np.zeros((mon, mon, NumAlg))

    for in_ in range(len(nsePwr)):
        print('NoisePower: ', nsePwr[in_], ",", in_ + 1, "/", len(nsePwr), "...")

        # Q = np.eye(M-1)*sigma(k)^2;
        Q = 10 ** (nsePwr[in_] / 10) * (np.ones((M - 1, M - 1)) + np.eye(M - 1)) / 2

        for ig in range(3):
            gamma = 1 - 0.2 * ig
            # position and DOA estimation
            np.random.seed(1)

            bia_p = np.zeros((N, mon, NumAlg))
            for i in range(mon):
                # measured TDOAs
                mu = np.random.rand(M, 1) * 20 * np.sqrt(10 ** (nsePwr[in_] / 10))
                gg = np.ones((M, 1))
                gg[np.random.rand(M, 1) + gamma - 1 > 0] = 0
                rMp = np.sqrt(10 ** (nsePwr[in_] / 10)) * (tmpMp[:, i][:, np.newaxis] + mu) * gg
                rdMp = (rMp[1:M] - rMp[0]) / np.sqrt(2)
                rdNse = np.sqrt(10 ** (nsePwr[in_] / 10)) * ((tmpNsed[1:M, i][:, np.newaxis] - tmpNsed[0, i]) / np.sqrt(2) - PP)
                rd_m = rd + rdNse + rdMp

                nAg = 0
                # SCO-MPR Method
                mprSol = TDOA_SCO_MPR(senPos, rd_m, Q)
                Th1 = mprSol[0]
                Ph1 = mprSol[1]
                g1 = mprSol[2]
                uTh[i, nAg, 0] = Th1[0]
                uPh[i, nAg, 0] = Ph1[0]
                ug[i, nAg, 0] = g1[0]

            # calculate MSE and bias
            for ia in range(nAg + 1):
                mse_a[in_, ir, ig, ia] = np.mean((uTh[:, ia][:, np.newaxis] - theta) ** 2 + (uPh[:, ia][:, np.newaxis] - phi) ** 2)
                mse_g[in_, ir, ig, ia] = np.mean((ug[:, ia][:, np.newaxis] - g) ** 2)

                avBia_a[in_, ir, ig, ia] = np.sqrt(np.abs((np.mean(uTh[:, ia][:, np.newaxis]) - theta)) ** 2 + np.abs(np.mean(uPh[:, ia][:, np.newaxis]) - phi) ** 2)
                avBia_g[in_, ir, ig, ia] = np.abs(np.mean(ug[:, ia]) - g)

symbs = ['o', '^', '*']
xlabtext = r'$10log(\sigma^2(m^2))$'
xdata = nsePwr

# plot results
fig = plt.figure()
for ia in range(nAg+1):
    for ig in range(3):
        # 画点
        plt.plot(xdata, 10 * np.log10(mse_a[:,:,ig,ia]), symbs[ig], linewidth=1.5, label='SCO-MPR, γi = {}'.format(1 - 0.2 * (ig)))
        # 显示网格
        plt.grid(True)
# 图例
plt.legend(fontsize=11)
# x,y轴名称
plt.xlabel(xlabtext, fontsize=13)
plt.ylabel(r'$10log(MSE(\theta,\phi)(rad^2))$', fontsize=13)
# x,y轴坐标范围
plt.xlim([-70, 20])
plt.ylim([-120, 20])
# 显示图形
plt.show()

fig = plt.figure()
for ia in range(nAg+1):
    for ig in range(3):
        # 画点
        plt.plot(xdata, 10 * np.log10(mse_g[:,:,ig,ia]), symbs[ig], linewidth=1.5, label='SCO-MPR, γi = {}'.format(1 - 0.2 * (ig)))
        # 显示网格
        plt.grid(True)
# 图例
plt.legend(fontsize=11)
# x,y轴名称
plt.xlabel(xlabtext, fontsize=13)
plt.ylabel(r'$10log(MSE(g)(1/m^2))$', fontsize=13)
# x,y轴坐标范围
plt.xlim([-70, 20])
# 显示图形
plt.show()

# bias
fig = plt.figure()
for ia in range(nAg+1):
    for ig in range(3):
        # 画点
        plt.plot(xdata, 20 * np.log10(avBia_a[:,:,ig,ia]), symbs[ig], linewidth=1.5, label='SCO-MPR, γi = {}'.format(1 - 0.2 * (ig)))
        # 显示网格
        plt.grid(True)
# 图例
plt.legend(fontsize=11)
# x,y轴名称
plt.xlabel(xlabtext, fontsize=13)
plt.ylabel(r'$20log(Bias(\theta,\phi)(rad))$', fontsize=13)
# x,y轴坐标范围
plt.xlim([-70, 20])
# 显示图形
plt.show()

fig = plt.figure()
for ia in range(nAg+1):
    for ig in range(3):
        # 画点
        plt.plot(xdata, 20 * np.log10(avBia_g[:,:,ig,ia]), symbs[ig], linewidth=1.5, label='SCO-MPR, γi = {}'.format(1 - 0.2 * (ig)))
        # 显示网格
        plt.grid(True)
# 图例
plt.legend(fontsize=11)
# x,y轴名称
plt.xlabel(xlabtext, fontsize=13)
plt.ylabel(r'$20log(Bias(g)(m^{-1}))$', fontsize=13)
# x,y轴坐标范围
plt.xlim([-70, 20])
# 显示图形
plt.show()
