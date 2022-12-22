import sys, glob, traceback, os, shutil, inspect
sys.path.append('/Users/morton/Documents/git/git_icrpythonradiomics')
from copy import deepcopy
import numpy as np
import pydicom
from time import strftime, localtime

# modules/functions I've written
from getSeriesUIDFolderDict import getSeriesUIDFolderDict
from radiomicAnalyser2 import radiomicAnalyser2, saveThumbnail

xnatFolder = '/Users/morton/Dicom Files/Prostate_ECE/XNAT_ProCancerI'

paramFileName = '/Users/morton/Dicom Files/Prostate_ECE/Params.yaml'

outputFolder = os.path.join(xnatFolder, 'extractions', 'extractions__' + strftime("%Y%m%d_%H%M", localtime()))

# copy all code including this file into output folder
os.makedirs(os.path.join(outputFolder, 'code'))
modulesToCopy = [getSeriesUIDFolderDict, radiomicAnalyser2]
[shutil.copyfile(inspect.getfile(x), os.path.join(outputFolder, 'code', os.path.split(inspect.getfile(x))[1])) for x in modulesToCopy]
shutil.copyfile(__file__, os.path.join(outputFolder, 'code', os.path.split(__file__)[1]))
shutil.copyfile(paramFileName, os.path.join(outputFolder, 'code', os.path.split(paramFileName)[1]))

# find all assessors and sort them
assessors = glob.glob(os.path.join(xnatFolder, 'assessors', '*.dcm'))
assessors.sort()

# dictionary to locate the images referenced by the assessors
seriesFolderDict = getSeriesUIDFolderDict(os.path.join(xnatFolder, 'referencedScans'))

assessorsWithWarnings = []
resultsFileList = []
thumbnailFileList = []

for assessor in assessors:

    print('Processing ' + assessor)

    try:

        # load all images in the linked series
        radAn = radiomicAnalyser2(assessor, seriesFolderDict, paramFileName) #, maxNonCompatibleInstances=maxNonCompatibleInstances)

        # get patient ID from filename (should be the same as PatientName in dicom file(s), but occasionally may not be
        patID = os.path.split(assessor)[1].split('__II__')[0]

        # special handling for one patient
        if patID=='0348':
            # this patient has a slice missing, but it isn't inside the lesion so it doesn't matter
            contiguousInstanceNumberCheck = False
            sliceSpacingUniformityThreshold = 1
        else:
            contiguousInstanceNumberCheck = True
            sliceSpacingUniformityThreshold = 0.005

        # loop over all ROIs in assessor
        roiList = []
        titleStr = ''
        for roiName in radAn.seriesData['ROINames']:

            radAn.computePyradiomicsFeatures(roiName,
                                             sliceSpacingUniformityThreshold=sliceSpacingUniformityThreshold,
                                             contiguousInstanceNumberCheck=contiguousInstanceNumberCheck,
                                             computePercentiles=True)

            # save results file
            resultsFile = os.path.join(outputFolder, 'radiomicFeatures', 'subjects', 'radiomicFeatures__' + os.path.split(assessor)[1].replace('.dcm','.csv'))
            resultsFileSaved = radAn.saveRadiomicsFeatures(resultsFile,
                                                           ProjectName='ProCancerI_ECE',
                                                           StudyPatientName=patID,
                                                           fileSubscript='__' + roiName)
            resultsFileList.append(resultsFileSaved)

            # keep copy of this ROI so we can make thumbnail image that contains all ROIs for this assessor (should only be one per patient in this study)
            roiList.append(deepcopy(radAn.selectedROI))

            # title string for thumbnail file to indicate if any ROIs are not contiguous
            if titleStr=='' and not radAn.selectedROI['maskContiguous']:
                titleStr = '\n' + r'$\bf{WARNING}$: contains volumes with missing slices'
                assessorsWithWarnings.append(assessor)

        # save thumbnail file
        thumbnailFile = os.path.join(outputFolder, 'thumbnails', 'thumbnail__' + os.path.split(assessor)[1].replace('.dcm', '.pdf'))
        saveThumbnail(roiList,
                      thumbnailFile,
                      volumePad=[2, 50, 50],
                      titleStr=patID+titleStr)
        thumbnailFileList.append(thumbnailFile)

        print('\n')

    except:
        print('\033[1;31;48m'+'_'*50)
        traceback.print_exc(file=sys.stdout)
        print('_'*50 + '\033[0;30;48m')

print('Assessors with warnings:')
print(assessorsWithWarnings)

# combine separate .csv files into one
if len(resultsFileList)>0:
    resultsFileList.sort()
    csvList = []
    # get column headings from first file
    with open(resultsFileList[0]) as fo:
        s = fo.readlines()
        csvList.append(s[0])
    # data from each file
    for csvFile in resultsFileList:
        with open(csvFile) as fo:
            s = fo.readlines()
            csvList.append(s[1])
    # write combined .csv
    with open(os.path.join(outputFolder, 'radiomicFeatures', 'radiomicFeatures.csv'), 'w') as file_handler:
        for item in csvList:
            file_handler.write("{}".format(item))
    print('')
    print('Combined results written to file '+file_handler.name)