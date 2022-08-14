# This is python coding for Assignment of Fuzzy Logic
# Assignment topic is using Fuzzy Logic to predict share price movement with 4 input parameters and an output parameter
# Please refer to the 9 rules and membership function in the submitted assignment file

import numpy as np
import main
import matplotlib.pyplot as plt

# Defining the Memberships
PE = np.linspace(0, 50, 200)
low_PE = np.zeros_like(PE)
med_PE = np.zeros_like(PE)
high_PE = np.zeros_like(PE)
for i in range(len(PE)):
    low_PE[i] = main.decreasing(PE[i], 0, 10)
    med_PE[i] = main.trape(PE[i], 8, 20, 20, 30)
    high_PE[i] = main.increasing(PE[i], 20, 30)

ROE = np.linspace(0, 30, 200)
poor_ROE = np.zeros_like(ROE)
ave_ROE = np.zeros_like(ROE)
high_ROE = np.zeros_like(ROE)
for i in range(len(ROE)):
    poor_ROE[i] = main.decreasing(ROE[i], 5, 7)
    ave_ROE[i] = main.trape(ROE[i], 5, 9, 9, 13)
    high_ROE[i] = main.increasing(ROE[i], 10, 15)

GPP = np.linspace(-1, 1, 100)
Neg_GPP = np.zeros_like(GPP)
Neu_GPP = np.zeros_like(GPP)
Pos_GPP = np.zeros_like(GPP)
for i in range(len(GPP)):
    Neg_GPP[i] = main.gauss1(GPP[i], -1, 0.25)
    Neu_GPP[i] = main.gauss(GPP[i], 0, 0.16)
    Pos_GPP[i] = main.gauss2(GPP[i], 1, 0.25)

MAJEC = np.linspace(-3, 3, 200)
Dec_MAJEC = np.zeros_like(MAJEC)
Const_MAJEC = np.zeros_like(MAJEC)
Inc_MAJEC = np.zeros_like(MAJEC)
for i in range(len(MAJEC)):
    Dec_MAJEC[i] = main.decreasing(MAJEC[i], -1, 0)
    Const_MAJEC[i] = main.trape(MAJEC[i], -1, 0, 0, 1)
    Inc_MAJEC[i] = main.increasing(MAJEC[i], 0, 1)

SPM = np.linspace(-8, 8, 100)
Red_SPM = np.zeros_like(SPM)
Const_SPM = np.zeros_like(SPM)
Inc_SPM = np.zeros_like(SPM)
for i in range(len(SPM)):
    Red_SPM[i] = main.gauss1(SPM[i], -5, 0.6)
    Const_SPM[i] = main.gauss(SPM[i], 0, 1.9)
    Inc_SPM[i] = main.gauss2(SPM[i], 5, 0.6)

# Plots
plt.figure(0)
plt.subplot(5, 1, 1)
plt.plot(PE, low_PE, label="LOW_PE")
plt.plot(PE, med_PE, label="MEDIUM_PE")
plt.plot(PE, high_PE, label="HIGH_PE")
plt.legend()

plt.subplot(5, 1, 2)
plt.plot(ROE, poor_ROE, label="POOR_ROE")
plt.plot(ROE, ave_ROE, label="AVERAGE_ROE")
plt.plot(ROE, high_ROE, label="HIGH_ROE")
plt.legend()

plt.subplot(5, 1, 3)
plt.plot(GPP, Neg_GPP, label="NEGATIVE_GPP")
plt.plot(GPP, Neu_GPP, label="NEUTRAL_GPP")
plt.plot(GPP, Pos_GPP, label="POSITIVE_GPP")
plt.legend()

plt.subplot(5, 1, 4)
plt.plot(MAJEC, Dec_MAJEC, label="DECREASING_GDP")
plt.plot(MAJEC, Const_MAJEC, label="CONSTANT_GDP")
plt.plot(MAJEC, Inc_MAJEC, label="INCREASING_GDP")
plt.legend()

plt.subplot(5, 1, 5)
plt.plot(SPM, Red_SPM, label="SHARE PRICE REDUCED")
plt.plot(SPM, Const_SPM, label="SHARE PRICE CONSTANT")
plt.plot(SPM, Inc_SPM, label="SHARE PRICE INCREASED")

plt.legend()
plt.show()
