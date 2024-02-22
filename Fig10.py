import numpy as np
from math import sin, cos, pi
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ConsCRLB import ConsCRLB
from TDOA_SUM_MPR import TDOA_SUM_MPR
from TDOA_SCO_MPR import TDOA_SCO_MPR
from TDOA_GTRS_MPR import TDOA_GTRS_MPR

np.random.seed(1)

clor = np.array([[0, 114, 189], [217, 83, 25], [237, 177, 32], [126, 47, 142], [119, 172, 48], [77, 190, 238], [162, 20, 47]]) / 256
np.random.seed(1)

senPos = np.array([[10.23, 38.38, 16.29],
                   [46.64, -87.12, 62.94],
                   [124.02, -7.98, 81.16],
                   [105.02, -51.72, 26.47],
                   [-81.56, 104.48, -80.49]]).T

thetaN = np.arange(-180, 190, 10) * np.pi / 180
phiN = np.concatenate((np.array([-89]), np.arange(-85, 90, 5), np.array([89]))) * np.pi / 180

N = len(senPos)
M = len(senPos[0])

# setting
sigma_sqr = -10  # 10log(m^2)
range_ = 500  # m
# range_ = 10000
mon = 5000

R = len(phiN)
K = len(thetaN)

# Monte-Carlo Simulation

aveNse = 0
for l in range(mon):
    aveNse = aveNse + np.random.randn(M, 1)
aveNse = aveNse / mon / np.sqrt(2)
PP = aveNse[1:] - aveNse[0]

print('Simulation is running ...')
senPosTmp = senPos

CRLB_a = np.zeros((R, K))
CRLB_g = np.zeros((R, K))

eTh1 = np.zeros((K,mon))
ePh1 = np.zeros((K, mon))
eg1 = np.zeros((K, mon))
uTh1 = np.zeros((K, mon))
uPh1 = np.zeros((K, mon))
ug1 = np.zeros((K, mon))

eTh2 = np.zeros((K,mon))
ePh2 = np.zeros((K, mon))
eg2 = np.zeros((K, mon))
uTh2 = np.zeros((K, mon))
uPh2 = np.zeros((K, mon))
ug2 = np.zeros((K, mon))

eTh3 = np.zeros((K,mon))
ePh3 = np.zeros((K, mon))
eg3 = np.zeros((K, mon))
uTh3 = np.zeros((K, mon))
uPh3 = np.zeros((K, mon))
ug3 = np.zeros((K, mon))

mse_a1 = np.zeros((R, R))
mse_a2 = np.zeros((R, R))
mse_a3 = np.zeros((R, R))
mse_g1 = np.zeros((R, R))
mse_g2 = np.zeros((R, R))
mse_g3 = np.zeros((R, R))

for t in range(R):
    phi = phiN[t]

    for k in range(K):
        print('phi: ', round(phiN[t] * 180 / np.pi), '(deg), ', t + 1, '/', R, ';',  'theta: ', round(thetaN[k] * 180 / np.pi), '(deg), ', k + 1, '/', K, '...')
        theta = thetaN[k]
        phiTmp = phi

        souLoc = range_ * np.reshape([cos(theta) * cos(phi), sin(theta) * cos(phi), sin(phi)], (3, 1)) + senPos[:, 0][:,np.newaxis]
        souLocTmp = souLoc
        r = (np.sqrt(np.sum((souLoc - senPos) ** 2, axis=0)).T)[:, np.newaxis]
        rd = r[1:] - r[0]
        rTmp = r

        Q = 10 ** (sigma_sqr / 10) * (np.ones((M - 1, M - 1)) + np.eye(M - 1)) / 2

        # calulate CRLB
        CRB = ConsCRLB(senPos, souLoc, Q)
        CRLB_a[k, t] = CRB[0, 0] + CRB[1, 1]
        CRLB_g[k, t] = CRB[2, 2]

        # Position and DOA estimate
        nsePwr = 10 ** (sigma_sqr / 10)

        # SCO-MPR Method
        np.random.seed(1)
        for i in range(mon):
            tmp = np.random.randn(M, 1)
            rdNse = np.sqrt(nsePwr) * ((tmp[1:M] - tmp[0]) / np.sqrt(2) - PP)
            rd_m = rd + rdNse

            # SCO
            mprSol = TDOA_SCO_MPR(senPos, rd_m, Q)
            Th1 = mprSol[0]
            Ph1 = mprSol[1]
            g1 = mprSol[2]
            if np.abs(theta - Th1) > np.pi:
                eTh1[k, i] = (2 * np.pi - np.abs(theta - Th1)) ** 2
            else:
                eTh1[k, i] = (theta - Th1) ** 2
            ePh1[k, i] = (phi - Ph1) ** 2
            eg1[k, i] = (1 / r[0] - g1) ** 2
            uTh1[k, i] = Th1
            uPh1[k, i] = Ph1
            ug1[k, i] = g1

            # SUM-MPR Method
            Th2, Ph2, g2, _ = TDOA_SUM_MPR(senPos, rd_m, Q)
            if np.abs(theta - Th2) > np.pi:
                eTh2[k, i] = (2 * np.pi - np.abs(theta - Th2)) ** 2
            else:
                eTh2[k, i] = (theta - Th2) ** 2
            ePh2[k, i] = (phi - Ph2) ** 2
            eg2[k, i] = (1 / r[0] - g2) ** 2
            uTh2[k, i] = Th2
            uPh2[k, i] = Ph2
            ug2[k, i] = g2

            # GTRS-MPR Method
            Th3, Ph3, g3, _ = TDOA_GTRS_MPR(senPos, rd_m, Q)
            if np.abs(theta - Th3) > np.pi:
                eTh3[k, i] = (2 * np.pi - np.abs(theta - Th3)) ** 2
            else:
                eTh3[k, i] = (theta - Th3) ** 2
            ePh3[k, i] = (phi - Ph3) ** 2
            eg3[k, i] = (1 / r[0] - g3) ** 2
            uTh3[k, i] = Th3
            uPh3[k, i] = Ph3
            ug3[k, i] = g3

    # calcuate MSE
    # MSE of angle
    mse_a1[:, t] = np.mean(eTh1 + ePh1, axis=1)
    mse_a2[:, t] = np.mean(eTh2 + ePh2, axis=1)
    mse_a3[:, t] = np.mean(eTh3 + ePh3, axis=1)

    # MSE of g
    mse_g1[:, t] = np.mean(eg1, axis=1)
    mse_g2[:, t] = np.mean(eg2, axis=1)
    mse_g3[:, t] = np.mean(eg3, axis=1)

symbs = ['o', 'v', 's']

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(thetaN*180/np.pi, 10*np.log10(np.mean(mse_a1/CRLB_a, axis=1)), symbs[0], linewidth=1.5, label='SCO-MPR')
ax1.plot(thetaN*180/np.pi, 10*np.log10(np.mean(mse_a2/CRLB_a, axis=1)), symbs[1], linewidth=1.5, label='SUM-MPR')
ax1.plot(thetaN*180/np.pi, 10*np.log10(np.mean(mse_a3/CRLB_a, axis=1)), symbs[2], linewidth=1.5, label='GTRS-MPR')
ax1.grid(True)
plt.xlabel(r'$\theta^o (deg)$', fontsize=13)
plt.ylabel(r'$\bar{R}_a(\theta^o)$', fontsize=13)
plt.xlim([-180, 180])
plt.xticks([-180, -120, -60, 0, 60, 120, 180])
# plt.ylim([-0.5, 3])
lgd = plt.legend(fontsize=11)
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(thetaN*180/np.pi, 10*np.log10(np.mean(mse_g1/CRLB_g, axis=1)), symbs[0], linewidth=1.5, label='SCO-MPR')
ax2.plot(thetaN*180/np.pi, 10*np.log10(np.mean(mse_g2/CRLB_g, axis=1)), symbs[1], linewidth=1.5, label='SUM-MPR')
ax2.plot(thetaN*180/np.pi, 10*np.log10(np.mean(mse_g3/CRLB_g, axis=1)), symbs[2], linewidth=1.5, label='GTRS-MPR')
ax2.grid(True)
plt.xlabel(r'$\theta^o (deg)$', fontsize=13)
plt.ylabel(r'$\bar{R}_g(\theta^o)$', fontsize=13)
# plt.xlim([min(thetaN), max(thetaN)]*180/np.pi)
plt.xlim([-180, 180])
plt.xticks([-180, -120, -60, 0, 60, 120, 180])
# plt.ylim([-0.5, -3])
lgd = plt.legend(fontsize=11)
# 显示图形
plt.show()

fig = plt.figure()
# Subplot 1
ax3 = fig.add_subplot(2, 1, 1)
ax3.plot(phiN*180/np.pi, 10*np.log10(np.mean(mse_a1/CRLB_a, axis=0)), symbs[0], linewidth=1.5, label='SCO-MPR')
ax3.plot(phiN*180/np.pi, 10*np.log10(np.mean(mse_a2/CRLB_a, axis=0)), symbs[1], linewidth=1.5, label='SUM-MPR')
ax3.plot(phiN*180/np.pi, 10*np.log10(np.mean(mse_a3/CRLB_a, axis=0)), symbs[2], linewidth=1.5, label='GTRS-MPR')
ax3.grid(True)
plt.xlabel(r'$\phi^o (deg)$', fontsize=13)
plt.ylabel(r'$\bar{R}_a(\phi^o)$', fontsize=13)
# plt.xlim([min(thetaN), max(thetaN)]*180/np.pi)
plt.xlim([-90, 90])
plt.xticks([-90, -60, -30, 0, 30, 60, 90])
# plt.ylim([-0.5, 3])
lgd = plt.legend(fontsize=11)
ax4 = fig.add_subplot(2, 1, 2)
ax4.plot(phiN*180/np.pi, 10*np.log10(np.mean(mse_g1/CRLB_g, axis=0)), symbs[0], linewidth=1.5, label='SCO-MPR')
ax4.plot(phiN*180/np.pi, 10*np.log10(np.mean(mse_g2/CRLB_g, axis=0)), symbs[1], linewidth=1.5, label='SUM-MPR')
ax4.plot(phiN*180/np.pi, 10*np.log10(np.mean(mse_g3/CRLB_g, axis=0)), symbs[2], linewidth=1.5, label='GTRS-MPR')
ax4.grid(True)
plt.xlabel(r'$\phi^o (deg)$', fontsize=13)
plt.ylabel(r'$\bar{R}_g(\phi^o)$', fontsize=13)
# plt.xlim([min(thetaN), max(thetaN)]*180/np.pi)
plt.xlim([-90, 90])
plt.xticks([-90, -60, -30, 0, 30, 60, 90])
# plt.ylim([-0.5, 3])
lgd = plt.legend(fontsize=11)
# 显示图形
plt.show()
