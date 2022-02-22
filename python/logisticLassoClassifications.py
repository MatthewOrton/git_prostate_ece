import sys
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics/machineLearning')
from loadAndPreProcessRadiomicsFile import loadAndPreProcessRadiomicsFile
from classificationNestedCVpermutationTest import classificationNestedCVpermutationTest

import numpy as np
import pandas as pd
from itertools import compress

from featureSelect_correlation import featureSelect_correlation

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold, RepeatedStratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# use all the processors unless we are in debug mode
n_jobs = -1
if getattr(sys, 'gettrace', None)():
    n_jobs = 1

n_permutations = 200
n_repeats = n_permutations

targetStrs = []
targetStrs.append("EvoST_Branched_vs_Punctuated")
# targetStrs.append("Loss_9p21_3")
# targetStrs.append("ITH_Index_Binarised")
# targetStrs.append("EvoST_Branched")
# targetStrs.append("EvoST_Punctuated")
# targetStrs.append("EvoST_Linear")
# targetStrs.append("EvoST_Unclassifiable")
# targetStrs.append("WGII_Median_Binarised")
# targetStrs.append("WGII_Max_Binarised")
# targetStrs.append("Loss_14q31_1")
# targetStrs.append("Loss_14q31_1_isClonal")
# targetStrs.append("Loss_9p21_3_isClonal")
# targetStrs.append("BAP1")
# targetStrs.append("BAP1_isClonal")
# targetStrs.append("PBRM1")
# targetStrs.append("PBRM1_isClonal")

for targetStr in targetStrs:

    print(' ')
    radiomicsFile1 = '/Users/morton/Dicom Files/TracerX/XNAT_Collaborations_Local/radiomics_2p5Dmerged_Original/radiomicFeatures/radiomics_2p5Dmerged_Original.csv'
    radiomicsFile2 = '/Users/morton/Dicom Files/TracerX/XNAT_Collaborations_Local/radiomics_2p5Dmerged_hiLoEnhancement/radiomicFeatures/radiomics_2p5Dmerged_hiLoEnhancement.csv'
    radiomicsFile1rep = '/Users/morton/Dicom Files/TracerX/XNAT_Collaborations_Local/radiomics_2p5Dmerged_Original_Reproducibility/radiomicFeatures/radiomics_2p5Dmerged_Original_Reproducibility.csv'
    radiomicsFile2rep = '/Users/morton/Dicom Files/TracerX/XNAT_Collaborations_Local/radiomics_2p5Dmerged_hiLoEnhancement_Reproducibility/radiomicFeatures/radiomics_2p5Dmerged_hiLoEnhancement.csv'
    radiomicsFiles = [[radiomicsFile1, radiomicsFile2], [radiomicsFile1rep, radiomicsFile2rep]]
    dfR, dfFu, iccValues = loadAndPreProcessRadiomicsFile(radiomicsFiles, index_col=1, correlation_threshold=0.95, iccThreshold=0.5, logTransform=True)

    # read radiomics data files
    radiomicsFile1 = '/Users/morton/Dicom Files/TracerX/XNAT_Collaborations_Local/radiomics_2p5Dmerged_Original/radiomicFeatures/radiomics_2p5Dmerged_Original.csv'
    dfR1 = pd.read_csv(radiomicsFile1)
    dfR2 = pd.read_csv(radiomicsFile2)

    # repro data files
    radiomicsFile1rep = '/Users/morton/Dicom Files/TracerX/XNAT_Collaborations_Local/radiomics_2p5Dmerged_Original_Reproducibility/radiomicFeatures/radiomics_2p5Dmerged_Original_Reproducibility.csv'
    dfR1rep = pd.read_csv(radiomicsFile1rep)
    radiomicsFile2rep = '/Users/morton/Dicom Files/TracerX/XNAT_Collaborations_Local/radiomics_2p5Dmerged_hiLoEnhancement_Reproducibility/radiomicFeatures/radiomics_2p5Dmerged_hiLoEnhancement.csv'
    dfR2rep = pd.read_csv(radiomicsFile2rep)

    # join data frames from pairs of files
    dfR = pd.merge(dfR1, dfR2, on="StudyPatientName")
    dfRrep = pd.merge(dfR1rep, dfR2rep, on="StudyPatientName")
    del dfR1, dfR2, dfR1rep, dfR2rep

    # remove unwanted columns
    dfR = dfR.loc[:,~dfR.columns.str.startswith('source')]
    dfR = dfR.loc[:,~dfR.columns.str.startswith('diagnostics')]
    dfRrep = dfRrep.loc[:,~dfRrep.columns.str.startswith('source')]
    dfRrep = dfRrep.loc[:,~dfRrep.columns.str.startswith('diagnostics')]

    dfR = dfR.loc[:,~dfR.columns.str.contains('rawAll_original_firstorder_90Percentile')]
    dfRrep = dfRrep.loc[:, ~dfRrep.columns.str.contains('rawAll_original_firstorder_90Percentile')]


    # remove shape features that we know are derived from others
    shapeFeaturesRemove = ['SurfaceArea', 'VoxelVolume', 'SurfaceVolumeRatio', 'Compactness1', 'Compactness2', 'SphericalDisproportion', 'MinorAxisLength', 'LeastAxisLength']
    for feat in shapeFeaturesRemove:
        dfR = dfR.loc[:, ~dfR.columns.str.contains(feat)]
        dfRrep = dfRrep.loc[:, ~dfRrep.columns.str.contains(feat)]

    # these features will be linearly correlated with MeshVolume^(1/3), so scale them by MeshVolume so the residual is directly available for modelling
    # shapeFeaturesNormalise = [] #['Maximum3DDiameter', 'MajorAxisLength', 'MinorAxisLength', 'LeastAxisLength']
    # shapeNorm = np.array(dfR["rawAll_original_shape_MeshVolume"]) ** (1/3)
    # shapeNormRep = np.array(dfRrep["rawAll_original_shape_MeshVolume"]) ** (1 / 3)
    # for feat in shapeFeaturesNormalise:
    #     for thisFeat in dfR.columns[dfR.columns.str.contains(feat)]:
    #         dfR[thisFeat] = np.array(dfR[thisFeat])/shapeNorm
    #         dfR.columns = [x+'_volNorm' if x == thisFeat else x for x in dfR.columns]
    #         dfRrep[thisFeat] = np.array(dfRrep[thisFeat])/shapeNormRep
    #         dfRrep.columns = [x + '_volNorm' if x == thisFeat else x for x in dfRrep.columns]

    # filter columns
    # colStr =  ['trimmedCore_original']
    # colStr = ['rawAll_original']
    # colStr = ['rawAll_original_shape']
    # colStr = ['trimmedWhole_original']
    # colStr = ['lowEnhancing_original']
    colStr = ['rawAll_original_shape', 'rawAll_original_firstorder', 'lowEnhancing_original', 'highEnhancing_original']
    #colStr = ['lowEnhancing_original', 'highEnhancing_original']

    colNames = dfR.columns
    colStr.append('StudyPatientName')
    colFilter = colNames.str.startswith(tuple(colStr))
    colNames = [colNames[n] for n in np.nonzero(colFilter)[0]]
    colNames.remove("StudyPatientName")
    colStr.remove("StudyPatientName")

    dfR = dfR.loc[:,colFilter]
    dfRrep = dfRrep.loc[:,colFilter]

    # remove all high enhancing shape features
    dfR = dfR.loc[:,~dfR.columns.str.contains('highEnhancing_original_shape')]
    dfRrep = dfRrep.loc[:,~dfRrep.columns.str.contains('highEnhancing_original_shape')]

    # normalise lowEnhancing MeshVolume by the rawAll MeshVolume
    if (dfR.columns=="rawAll_original_shape_MeshVolume").any():
        lowEnhanceVol = np.array(dfR.loc[:, dfR.columns.str.contains('lowEnhancing_original_shape_MeshVolume')])
        tumourVol =  np.array(dfR.loc[:, dfR.columns.str.contains('rawAll_original_shape_MeshVolume')])
        dfR.loc[:, dfR.columns.str.contains('lowEnhancing_original_shape_MeshVolume')] = lowEnhanceVol/tumourVol
        dfR.columns = [x + '_fraction' if x == 'lowEnhancing_original_shape_MeshVolume' else x for x in dfR.columns]
        #
        lowEnhanceVol = np.array(dfRrep.loc[:, dfRrep.columns.str.contains('lowEnhancing_original_shape_MeshVolume')])
        tumourVol =  np.array(dfRrep.loc[:, dfRrep.columns.str.contains('rawAll_original_shape_MeshVolume')])
        dfRrep.loc[:, dfRrep.columns.str.contains('lowEnhancing_original_shape_MeshVolume')] = lowEnhanceVol/tumourVol
        dfRrep.columns = [x + '_fraction' if x == 'lowEnhancing_original_shape_MeshVolume' else x for x in dfRrep.columns]

    # log the MeshVolume features
    for feat in dfR.columns[dfR.columns.str.endswith('MeshVolume')]:
        dfR[feat] = np.log(dfR[feat] + 0.0000001)
        dfR.columns = [x + '_log' if x == feat else x for x in dfR.columns]
        dfRrep[feat] = np.log(dfRrep[feat])
        dfRrep.columns = [x + '_log' if x == feat else x for x in dfRrep.columns]


    colNames = list(dfR.columns)
    colNames.remove("StudyPatientName")

    # read clinical data file
    clinicalFile = '/Users/morton/Dicom Files/TracerX/Analysis/SS_200221_Cohort_summary_MOcurated.xlsx'
    dfC = pd.read_excel(clinicalFile, sheet_name='S1B_update', usecols='A:AB', header=2)

    # add some columns that will be classification targets
    dfC["ITH_Index_Binarised"] = dfC["ITH_Index"] > np.nanmedian(dfC["ITH_Index"])
    dfC["WGII_Median_Binarised"] = dfC["WGII_Median"] > np.nanmedian(dfC["WGII_Median"])
    dfC["EvoST_Punctuated"] = dfC["EvoST"] == "Punctuated"
    dfC["EvoST_Linear"] = dfC["EvoST"] == "Linear"
    dfC["EvoST_Branched"] = dfC["EvoST"] == "Branched"
    dfC["EvoST_Unclassifiable"] = dfC["EvoST"] == "Unclassifiable"
    EvoST_Branched_vs_Punctuated = dfC["EvoST"] == "Branched"
    EvoST_Branched_vs_Punctuated[np.logical_and(dfC["EvoST"] != "Branched", dfC["EvoST"] != "Punctuated")] = np.nan
    dfC["EvoST_Branched_vs_Punctuated"] = EvoST_Branched_vs_Punctuated

    # get rows of main data corresponding to repro data
    dfRrep0 = pd.merge(dfR, dfRrep["StudyPatientName"], on="StudyPatientName")

    dfR = pd.merge(dfC[["Subject", targetStr]], dfR, left_on="Subject", right_on="StudyPatientName")
    dfR.drop("Subject", 1, inplace=True)
    dfR.drop("StudyPatientName", 1, inplace=True)

    # remove rows with non-valid targets
    dfR = dfR.loc[~dfR[targetStr].isnull().values,:]


    # get ICCs
    # get rows of dfR corresponding to dfRrep
    iccValue = {}
    for col in colNames:
        data = np.stack((dfRrep[col], dfRrep0[col]), axis=1)
        if np.all(np.isfinite(np.squeeze(data))):
            iccValue[col] = intraclass_correlation(data, "twoway", "agreement").value
        else:
            iccValue[col] = 0

    # remove columns that fail the ICC threshold
    iccThreshold = 0.6
    iccMask = [value>iccThreshold for key, value in iccValue.items()]

    sns.scatterplot(data=dfR, x="rawAll_original_shape_Sphericity", y="rawAll_original_firstorder_InterquartileRange", hue=targetStr)
    plt.show()
    sns.scatterplot(data=dfR, x="rawAll_original_shape_Sphericity", y="rawAll_original_shape_MeshVolume_log", hue=targetStr)
    plt.show()
    sns.scatterplot(data=dfR, x="rawAll_original_shape_MeshVolume_log", y="rawAll_original_firstorder_InterquartileRange", hue=targetStr)
    plt.show()
    sns.scatterplot(data=dfR, x="highEnhancing_original_glcm_MaximumProbability", y="highEnhancing_original_firstorder_Energy", hue=targetStr)
    plt.show()
    # sns.scatterplot(data=dfR, x="rawAll_original_firstorder_90Percentile", y="highEnhancing_original_glcm_MaximumProbability", hue=targetStr)
    # plt.show()




    y = np.array(dfR[targetStr])
    dfR.drop(targetStr, 1, inplace=True)
    X = np.array(dfR)

    X = X[:, iccMask]
    colNames = list(compress(colNames, iccMask))


    # remove any rows with nans in X
    rowOK = np.sum(np.isnan(X),axis=1)==0
    X = X[rowOK,:]
    y = y[rowOK]

    # log transform any features that are all positive or all negative
    # for n, column in enumerate(X.T):
    #     if np.all(np.sign(column)==np.sign(column[0])):
    #         X[:,n] = np.log(np.abs(column))
    #     X[:, n] = X[:,n] - np.mean(X[:,n])
    #     X[:, n] = X[:, n]/np.std(X[:, n])

    # pre-process data

    # Remove any features that are correlated with MeshVolume or Sphericity (columns 0 and 1)
    correlationThreshold = 0.9
    xCorr = np.abs(spearmanr(X).correlation)
    ind = np.logical_and(xCorr[:,0]<correlationThreshold, xCorr[:,1]<correlationThreshold)
    ind[0] = True
    ind[1] = True
    X = X[:,ind]
    colNames = list(compress(colNames, ind))
    #
    # X = featureSelect_correlation(threshold=0.9).fit_transform(X)
    fsc = featureSelect_correlation(threshold=correlationThreshold, exact=True)
    fsc.fit(X)
    selectionMask = fsc._get_support_mask()
    X = X[:, selectionMask]
    colNames = list(compress(colNames, selectionMask))
    X = StandardScaler().fit_transform(X)


    estimator = {"model": LogisticRegression(solver="liblinear", max_iter=10000, penalty='l1'),
                 "name": "Logistic",
                 "p_grid": {"C": np.logspace(-4,0,20)},
                 "scoring": "neg_log_loss", "result": {}}



    print(' ')
    # print(estimator["model"])
    print('target feature = \033[1m' + targetStr + '\033[0m')
    print('N              = ' + str(len(y)) + ' = ' + str(np.count_nonzero(y==0)) + ' + ' + str(np.count_nonzero(y==1)))
    print('Features       = ' + str(colStr).replace('[','').replace(']','').replace("'",'').replace(',',' &'))
    # print('ICC threshold   = ' + str(iccThreshold))
    # print('n_repeats      = ' + str(n_repeats))
    # print('n_permutations = ' + str(n_permutations))

    np.random.seed(0)

    # fit to all data using CV for lasso parameter optimisation
    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
    clfAll = GridSearchCV(estimator=estimator["model"], param_grid=estimator["p_grid"], cv=inner_cv, refit=True, verbose=0, scoring=estimator["scoring"])
    clfAll.fit(X,y)
    resubAUROC = roc_auc_score(y, clfAll.predict_proba(X)[:, 1])

    print('AUCROC (resub) = ' + str(np.round(resubAUROC,3)))

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True)
    clf = GridSearchCV(estimator=estimator["model"], param_grid=estimator["p_grid"], cv=inner_cv, refit=True, verbose=0, scoring=estimator["scoring"])

    # Permutation Tests for Studying Classifier Performance Markus Ojala

    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats)
    cv_result = cross_validate(clf, X=X, y=y, cv=outer_cv, scoring="roc_auc", return_estimator=True, verbose=0, n_jobs=n_jobs)
    # get scores for each repeat, averaged over the CV-folds
    scores = np.mean(np.reshape(cv_result['test_score'], (n_repeats, -1)), axis=1)

    print('AUCROC (CV)    = \033[1m' + str(np.mean(scores).round(3)) + '\033[0m') # + ' \u00B1 ' + str(np.std(scores).round(3)))

    # permutation test needs to use the same type of splitter as for outer_cv, but only needs to use one repeat
    outer_cv.n_repeats = 1
    _, perm_scores, _ = permutation_test_score(clf, X, y, scoring="roc_auc", cv=outer_cv, n_permutations=n_permutations, verbose=0, n_jobs=n_jobs)

    p_values = []
    for score in scores:
        p_values.append((np.count_nonzero(perm_scores >= score) + 1) / (n_permutations + 1))

    #print('AUCROC (perm)  = ' + str(np.mean(perm_scores).round(3))) # + ' \u00B1 ' + str(np.std(perm_scores).round(3)))
    print('p-value        = \033[1m' + str(np.mean(p_values).round(4)) + '\033[0m') # + ' (' + str(np.quantile(p_values, 0.025).round(4)) + ', ' + str(np.quantile(p_values, 0.975).round(4)) + ')')
    print(' ')
    print('coeff  ICC   AUROC featureName')
    coef = np.squeeze(clfAll.best_estimator_.coef_)
    I = np.argsort(np.abs(coef))[::-1]
    coef = coef[I]
    colNames = [colNames[n] for n in I]
    X = X[:,I]
    for n in np.nonzero(coef)[0]:
        rocauc = roc_auc_score(y, X[:,n])
        if rocauc<0.5:
            rocauc = 1 - rocauc
        print(str('{:.3f}'.format(round(coef[n], 3))).rjust(6) + ' ' +
              str('{:.3f}'.format(round(iccValue[colNames[n]],3))).ljust(5) + ' ' +
              str('{:.3f}'.format(round(rocauc, 3))).ljust(5) + ' ' +
              colNames[n])

    print(' ')
