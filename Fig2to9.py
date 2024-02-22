import numpy as np
# from math import sin, cos, pi
# from scipy.linalg import sqrtm
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

# target direction
theta = 22.13 * np.pi / 180
phi = 14.41 * np.pi / 180

N = len(senPos)
M = len(senPos[0])
mon = 1000

aveNse = 0
for l in range(mon):
    aveNse = aveNse + np.random.randn(M, 1)
aveNse = aveNse / mon / np.sqrt(2)
PP = aveNse[1:] - aveNse[0]

print('Simulation is running ...')
models = ['nse', 'rag']

nsePwr = list(range(-40, 51, 10))
souRange = [100, 300] + list(range(500, 8001, 500))
CRLB_a = np.zeros((len(nsePwr), len(souRange)))
CRLB_g = np.zeros((len(nsePwr), len(souRange)))

uTh = np.zeros((mon, 3))
uPh = np.zeros((mon, 3))
ug = np.zeros((mon, 3))
ep = np.zeros((mon, 3))
er = np.zeros((mon, 3))

mse_a = np.zeros((len(nsePwr), len(souRange), 3))
mse_g = np.zeros((len(nsePwr), len(souRange), 3))
avBia_a = np.zeros((len(nsePwr), len(souRange), 3))
avBia_g = np.zeros((len(nsePwr), len(souRange), 3))

for im in range(2):
    model = models[im]
    if model == 'nse':
        nsePwr = list(range(-40, 51, 10))
        souRange = [1500]
    elif model == 'rag':
        nsePwr = [0]
        souRange = [100, 300] + list(range(500, 8001, 500))

    for ir in range(len(souRange)):
        print("Range: ", souRange[ir], ",", ir + 1, "/", len(souRange), "...")

        # *************Generate Data***************
        # source location
        souLoc = (souRange[ir] * np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)]) + senPos[:, 0])[:, np.newaxis]
        # true range
        r = (np.sqrt(np.sum((souLoc - senPos) ** 2, axis=0)).T)[:, np.newaxis]
        # true TDOAs
        rd = r[1:] - r[0]
        g = 1 / r[0]

        for in_ in range(len(nsePwr)):
            print("10log(NoisePower):", nsePwr[in_], ",", in_ + 1, "/", len(nsePwr), "...")
            Q = 10 ** (nsePwr[in_] / 10) * (np.ones((M - 1, M - 1)) + np.eye(M - 1)) / 2

            # Calculate CRLB
            CRB = ConsCRLB(senPos, souLoc, Q)
            CRLB_a[in_, ir] = CRB[0, 0] + CRB[1, 1]
            CRLB_g[in_, ir] = CRB[2, 2]

            np.random.seed(1)
            for i in range(mon):
                # measured TDOAs
                tmp = np.random.randn(M, 1)
                rdNse = np.sqrt(10 ** (nsePwr[in_] / 10)) * ((tmp[1:M] - tmp[0]) / np.sqrt(2) - PP)
                rd_m = rd + rdNse

                nAg = 0
                # SCO-MPR Method
                mprSol = TDOA_SCO_MPR(senPos, rd_m, Q)
                Th1 = mprSol[0]
                Ph1 = mprSol[1]
                g1 = mprSol[2]
                uTh[i, nAg] = Th1
                uPh[i, nAg] = Ph1
                ug[i, nAg] = g1

                # SUM-MPR Method
                nAg = nAg + 1
                Th2, Ph2, g2, _ = TDOA_SUM_MPR(senPos, rd_m, Q)
                uTh[i, nAg] = Th2
                uPh[i, nAg] = Ph2
                ug[i, nAg] = g2

                # GTRS-MPR Method
                nAg = nAg + 1
                Th3, Ph3, g3, _ = TDOA_GTRS_MPR(senPos, rd_m, Q)
                uTh[i, nAg] = Th3
                uPh[i, nAg] = Ph3
                ug[i, nAg] = g3

            # calculate MSE and bias
            for ia in range(nAg + 1):
                mse_a[in_, ir, ia] = np.mean((uTh[:, ia] - theta) ** 2 + (uPh[:, ia] - phi) ** 2)
                mse_g[in_, ir, ia] = np.mean((ug[:, ia] - g) ** 2)

                avBia_a[in_, ir, ia] = np.sqrt(np.abs(np.mean(uTh[:, ia]) - theta) ** 2 + np.abs(np.mean(uPh[:, ia]) - phi) ** 2)
                avBia_g[in_, ir, ia] = np.abs(np.mean(ug[:, ia]) - g)

    symbs = ['o', 'v', 's', '*', '^', '+', 'x']
    name = ['SCO-MPR', 'SUM-MPR', 'GTRS-MPR']

    if model == 'nse':
        xlabtext = r'$10log(\sigma^2(m^2))$'
        xdata = nsePwr
        yl_mse = np.array([[-80, 20], [-120, 20]])
        yl_bias = np.array([[-155,3], [-250,0]])
    elif model == 'rag':
        xlabtext = r'$Range(m)$'
        xdata = souRange
        yl_mse = np.array([[-40, 10], [-80, -45]])
        yl_bias = np.array([[-80, 0], [-150, -50]])

    # plot results
    fig = plt.figure()
    for ia in range(nAg + 1):
        if model == 'nse':
            # 画点
            plt.plot(xdata, 10 * np.log10(mse_a[:, 0, ia]), symbs[ia], linewidth=1.5, label=name[ia])
            # 显示网格
            plt.grid(True)
        elif model == 'rag':
            # 画点
            plt.plot(xdata, 10 * np.log10(mse_a[0, :, ia]), symbs[ia], linewidth=1.5, label=name[ia])
            # 显示网格
            plt.grid(True)
    # 画线
    if model == 'nse':
        plt.plot(xdata, 10 * np.log10(CRLB_a[:, 0]), '-', linewidth=1.5, label='CRLB')
    elif model == 'rag':
        plt.plot(xdata, 10 * np.log10(CRLB_a[0, :]), '-', linewidth=1.5, label='CRLB')
    # 图例
    plt.legend(fontsize=11)
    # x,y轴名称
    plt.xlabel(xlabtext, fontsize=13)
    plt.ylabel(r'$10log(MSE(\theta,\phi)(rad^2))$', fontsize=13)
    # x,y轴坐标范围
    plt.ylim(yl_mse[0, :])
    # 显示图形
    plt.show()

    fig = plt.figure()
    for ia in range(nAg + 1):
        if model == 'nse':
            # 画点
            plt.plot(xdata, 10 * np.log10(mse_g[:, 0, ia]), symbs[ia], linewidth=1.5, label=name[ia])
            # 显示网格
            plt.grid(True)
        elif model == 'rag':
            # 画点
            plt.plot(xdata, 10 * np.log10(mse_g[0, :, ia]), symbs[ia], linewidth=1.5, label=name[ia])
            # 显示网格
            plt.grid(True)
    # 画线
    if model == 'nse':
        plt.plot(xdata, 10 * np.log10(CRLB_g[:, 0]), '-', linewidth=1.5, label='CRLB')
    elif model == 'rag':
        plt.plot(xdata, 10 * np.log10(CRLB_g[0, :]), '-', linewidth=1.5, label='CRLB')
    # 图例
    plt.legend(fontsize=11)
    # x,y轴名称
    plt.xlabel(xlabtext, fontsize=13)
    plt.ylabel(r'$10log(MSE(g)(1/m^2))$', fontsize=13)
    # x,y轴坐标范围
    plt.ylim(yl_mse[1, :])
    # 显示图形
    plt.show()

    # Bias
    fig = plt.figure()
    for ia in range(nAg + 1):
        if model == 'nse':
            # 画点
            plt.plot(xdata, 20 * np.log10(avBia_a[:, 0, ia]), symbs[ia], linewidth=1.5, label=name[ia])
            # 显示网格
            plt.grid(True)
        elif model == 'rag':
            # 画点
            plt.plot(xdata, 20 * np.log10(avBia_a[0, :, ia]), symbs[ia], linewidth=1.5, label=name[ia])
            # 显示网格
            plt.grid(True)
    # 图例
    plt.legend(fontsize=11)
    # x,y轴名称
    plt.xlabel(xlabtext, fontsize=13)
    plt.ylabel(r'$20log(Bias(\theta,\phi)(rad))$', fontsize=13)
    # x,y轴坐标范围
    plt.ylim(yl_bias[0, :])
    # 显示图形
    plt.show()

    fig = plt.figure()
    for ia in range(nAg + 1):
        if model == 'nse':
            # 画点
            plt.plot(xdata, 20 * np.log10(avBia_g[:, 0, ia]), symbs[ia], linewidth=1.5, label=name[ia])
            # 显示网格
            plt.grid(True)
        elif model == 'rag':
            # 画点
            plt.plot(xdata, 20 * np.log10(avBia_g[0, :, ia]), symbs[ia], linewidth=1.5, label=name[ia])
            # 显示网格
            plt.grid(True)
    # 图例
    plt.legend(fontsize=11)
    # x,y轴名称
    plt.xlabel(xlabtext, fontsize=13)
    plt.ylabel(r'$20log(Bias(g)(m^{-1}))$', fontsize=13)
    # x,y轴坐标范围
    plt.ylim(yl_bias[1, :])
    # 显示图形
    plt.show()

