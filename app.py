# This is python coding for Assignment of Fuzzy Logic
# Assignment topic is using Fuzzy Logic to predict share price movement with 4 input parameters and an ouput parameter
# Please refer to the 9 rules and membership function in the submitted assignment file
# Define membership functions to reflect the verbal definitions in the following rules

import numpy as np
import main
import matplotlib.pyplot as plt

PE = np.linspace(0,50,200)
low_PE = np.zeros_like(PE)
med_PE = np.zeros_like(PE)
high_PE = np.zeros_like(PE)
for i in range(len(PE)):
    low_PE[i]=main.decreasing(PE[i],0,10)
    med_PE[i]=main.trape(PE[i],8,20,20,30)
    high_PE[i]=main.increasing(PE[i],20,30)

ROE = np.linspace(0,30,200)
poor_ROE = np.zeros_like(ROE)
ave_ROE = np.zeros_like(ROE)
high_ROE = np.zeros_like(ROE)
for i in range(len(ROE)):
    poor_ROE[i]=main.decreasing(ROE[i],5,7)
    ave_ROE[i]=main.trape(ROE[i],5,9,9,13)
    high_ROE[i]=main.increasing(ROE[i],10,15)

GPP = np.linspace(-1,1,100)
Neg_GPP = np.zeros_like(GPP)
Neu_GPP = np.zeros_like(GPP)
Pos_GPP = np.zeros_like(GPP)
for i in range(len(GPP)):
    Neg_GPP[i]=main.gauss1(GPP[i],-1,0.25)
    Neu_GPP[i]=main.gauss(GPP[i],0,0.16)
    Pos_GPP[i]=main.gauss2(GPP[i],1,0.25)

MAJEC = np.linspace(-3,3,200)
Dec_MAJEC = np.zeros_like(MAJEC)
Const_MAJEC = np.zeros_like(MAJEC)
Inc_MAJEC = np.zeros_like(MAJEC)
for i in range(len(MAJEC)):
    Dec_MAJEC[i]=main.decreasing(MAJEC[i],-1,0)
    Const_MAJEC[i]=main.trape(MAJEC[i],-1,0,0,1)
    Inc_MAJEC[i]=main.increasing(MAJEC[i],0,1)

SPM = np.linspace(-8,8,100)
Red_SPM = np.zeros_like(SPM)
Const_SPM = np.zeros_like(SPM)
Inc_SPM = np.zeros_like(SPM)
for i in range(len(SPM)):
    Red_SPM[i]=main.gauss1(SPM[i],-5,0.6)
    Const_SPM[i]=main.gauss(SPM[i],0,1.9)
    Inc_SPM[i]=main.gauss2(SPM[i],5,0.6)

# Writing arbitrary 4 inputs for testing
PE_input = 40
ROE_input = 9
GPP_input = 0
MAJEC_input =0

# Fuzzification of the 4 inputs using membership function

low_PE_input=main.decreasing(PE_input,0,10)
med_PE_input=main.trape(PE_input,8,20,20,30)
high_PE_input=main.increasing(PE_input,20,30)

poor_ROE_input=main.decreasing(ROE_input,5,7)
ave_ROE_input=main.trape(ROE_input,5,9,9,13)
high_ROE_input=main.increasing(ROE_input,10,15)

Neg_GPP_input=main.gauss1(GPP_input,-1,0.25)
Neu_GPP_input=main.gauss(GPP_input,0,0.16)
Pos_GPP_input=main.gauss2(GPP_input,1,0.25)

Dec_MAJEC_input=main.decreasing(MAJEC_input,-1,0)
Const_MAJEC_input=main.trape(MAJEC_input,-1,0,0,1)
Inc_MAJEC_input=main.increasing(MAJEC_input,0,1)

# Evaluate the rules
# R1 : If low PE, and Good ROE, and Positive General public perception, and Growing GDP, then Share Price will increase
R1 = np.fmin(np.min((low_PE_input,high_ROE_input,Pos_GPP_input,Inc_MAJEC_input)),Inc_SPM)

#R2 : If low PE, and Average ROE, and Neutral General public perception, and Constant GDP, then Share Price will be constant
R2 = np.fmin(np.min((low_PE_input,ave_ROE_input,Neu_GPP_input,Const_MAJEC_input)),Const_SPM)

#R3 : If low PE, and Poor ROE, and Negative General public perception, and Decreasing GDP, then Share Price will reduce
R3 = np.fmin(np.min((low_PE_input,poor_ROE_input,Neg_GPP_input,Dec_MAJEC_input)),Red_SPM)

#R4 : If Medium PE, and Good ROE, and Positive General public perception, and Growing GDP, then Share Price will increase
R4 = np.fmin(np.min((med_PE_input,high_ROE_input,Pos_GPP_input,Inc_MAJEC_input)),Inc_SPM)

#R5 : If Medium PE, and Average ROE, and Neutral General public perception, and Constant GDP, then Share Price will be constant
R5 = np.fmin(np.min((med_PE_input,ave_ROE_input,Neu_GPP_input,Const_MAJEC_input)),Const_SPM)

#R6 : If Medium PE, and Poor ROE, and Negative General public perception, and Decreasing GDP, then Share Price will reduce
R6 = np.fmin(np.min((med_PE_input,poor_ROE_input,Neg_GPP_input,Dec_MAJEC_input)),Red_SPM)

#R7 : If High PE, and Good ROE, and Positive General public perception, and Growing GDP, then Share Price will increase
R7 = np.fmin(np.min((high_PE_input,high_ROE_input,Pos_GPP_input,Inc_MAJEC_input)),Inc_SPM)

#R8 : If High PE, and Average ROE, and Neutral General public perception, and Constant GDP, then Share Price will reduce
R8 = np.fmin(np.min((high_PE_input,ave_ROE_input,Neu_GPP_input,Const_MAJEC_input)),Red_SPM)

#R9 : If High PE, and Poor ROE, and Negative General public perception, and Decreasing GDP, then Share Price will reduce
R9 = np.fmin(np.min((high_PE_input,poor_ROE_input,Neg_GPP_input,Dec_MAJEC_input)),Red_SPM)

# Summarize the rules
R = np.maximum.reduce([R1,R2,R3,R4,R5,R6,R7,R8,R9])

# Defuzzification (Centroid method)
SPM_output= np.trapz(R * SPM, SPM) / np.trapz(R, SPM)
# print("R1",R1)
# print("R2",R2)
# print("R3",R3)
# print("R4",R4)
# print("R5",R5)
# print("R6",R6)
# print("R7",R7)
# print("R8",R8)
# print("R9",R9)
print("Centroid of R is ", SPM_output)

# Plots for visualization
plt.figure(figsize=(8,6))
plt.subplot(5,1,1)
plt.plot(PE,low_PE,label="LOW_PE")
plt.plot(PE,med_PE,label="MEDIUM_PE")
plt.plot(PE,high_PE,label="HIGH_PE")
plt.scatter([PE_input,PE_input],[low_PE_input,low_PE_input])
plt.scatter([PE_input,PE_input],[med_PE_input,med_PE_input])
plt.scatter([PE_input,PE_input],[high_PE_input,high_PE_input])
plt.legend(loc='center left',fontsize=8, bbox_to_anchor=(1,0.5))
plt.subplot(5,1,2)
plt.plot(ROE,poor_ROE,label="POOR_ROE")
plt.plot(ROE,ave_ROE,label="AVERAGE_ROE")
plt.plot(ROE,high_ROE,label="HIGH_ROE")
plt.scatter([ROE_input,ROE_input],[poor_ROE_input,poor_ROE_input])
plt.scatter([ROE_input,ROE_input],[ave_ROE_input,ave_ROE_input])
plt.scatter([ROE_input,ROE_input],[high_ROE_input,high_ROE_input])
plt.legend(loc='center left',fontsize=8, bbox_to_anchor=(1,0.5))
plt.subplot(5,1,3)
plt.plot(GPP,Neg_GPP,label="NEGATIVE_GPP")
plt.plot(GPP,Neu_GPP,label="NEUTRAL_GPP")
plt.plot(GPP,Pos_GPP,label="POSITIVE_GPP")
plt.scatter([GPP_input,GPP_input],[Neg_GPP_input,Neg_GPP_input])
plt.scatter([GPP_input,GPP_input],[Neu_GPP_input,Neu_GPP_input])
plt.scatter([GPP_input,GPP_input],[Pos_GPP_input,Pos_GPP_input])
plt.legend(loc='center left',fontsize=8, bbox_to_anchor=(1,0.5))
plt.subplot(5,1,4)
plt.plot(MAJEC,Dec_MAJEC,label="DECREASING_GDP")
plt.plot(MAJEC,Const_MAJEC,label="CONSTANT_GDP")
plt.plot(MAJEC,Inc_MAJEC,label="INCREASING_GDP")
plt.scatter([MAJEC_input,MAJEC_input],[Dec_MAJEC_input,Dec_MAJEC_input])
plt.scatter([MAJEC_input,MAJEC_input],[Const_MAJEC_input,Const_MAJEC_input])
plt.scatter([MAJEC_input,MAJEC_input],[Inc_MAJEC_input,Inc_MAJEC_input])
plt.legend(loc='center left',fontsize=8, bbox_to_anchor=(1,0.5))
plt.subplot(5,1,5)
plt.plot(SPM,Red_SPM,label="SHARE PRICE REDUCED")
plt.plot(SPM,Const_SPM,label="SHARE PRICE CONSTANT")
plt.plot(SPM,Inc_SPM,label="SHARE PRICE INCREASED")
plt.fill_between(SPM,R1)
plt.fill_between(SPM,R2)
plt.fill_between(SPM,R3)
plt.fill_between(SPM,R4)
plt.fill_between(SPM,R5)
plt.fill_between(SPM,R6)
plt.fill_between(SPM,R7)
plt.fill_between(SPM,R8)
plt.fill_between(SPM,R9)
plt.scatter([SPM_output, SPM_output],[0,0])
plt.legend(loc='center left',fontsize=8, bbox_to_anchor=(1,0.5))
plt.subplots_adjust(left = 0.05, hspace=0.8)
plt.show()