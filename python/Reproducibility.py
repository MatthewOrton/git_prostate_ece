import pandas as pd
from pyirr import intraclass_correlation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import numpy as np


df1 = pd.read_csv('/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/ECE_Prostate_Semantic/ECE_Semantic_Data/Reader01.csv')
df2 = pd.read_csv('/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/ECE_Prostate_Semantic/ECE_Semantic_Data/Reader02.csv')
dfC = pd.read_excel('/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/ECE_Prostate_Semantic/ECE_Semantic_Data/GG_MG.xlsx', sheet_name='GG_MG')
dfC = dfC.dropna(axis=0)

discreteTargets = ['SmoothCapsularBulgin',
 'CapsularDisruption',
 'UnsharpMargins',
 'irregularContour',
 'BlackEstrition',
 'measurableECE_',
 'retroprostaticAngleObl_']

contTargets = ['LesionSize', 'CapsularContactLength']

print("Cohen's kappa for discrete targets")
for target in discreteTargets:
    print(target.ljust(25) + ' = ' + str(np.round(cohen_kappa_score(dfC[target+'MG'],dfC[target+'GG']),3)))
print(' ')

print('ICC for continuous targets')
for target in contTargets:
    print(target.ljust(25) + ' = ' + str(np.round(intraclass_correlation(np.stack((dfC[target + 'MG'], dfC[target + 'GG']), axis=1), "twoway", "agreement").value,3)))
print(' ')


print('ICC for discrete targets')
for target in discreteTargets:
    print(target.ljust(25) + ' = ' +  str(np.round(intraclass_correlation(np.stack((dfC[target + 'MG'], dfC[target + 'GG']), axis=1), "twoway", "agreement").value,3)))
print(' ')


