import sys
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics')

import os, glob
from getSopInstDict import getSopInstDict
from radiomicAnalyser import radiomicAnalyser
from subprocess import call
import sys, traceback, shutil, inspect
import pygit2
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
from PyPDF2 import PdfFileMerger
import pydicom
import re
from time import strftime, localtime
import pydicom
import pandas as pd

projectFolder = '/Users/morton/Dicom Files/Prostate_ECE'

project = {}
project["projectStr"] = 'Prostate_ECE'
project["inputPath"] = os.path.join(projectFolder, 'XNAT', 'experiments')
project["assessorStyle"] = {"type": "seg", "format": "dcm"}
project["roiObjectLabelFilter"] = ''
project["paramFileName"] =  os.path.join(projectFolder, 'Params.yaml')
outputFolder = 'radiomicFeatures__' + strftime("%Y%m%d%H%M", localtime())
project["outputPath"] = os.path.join(projectFolder, 'XNAT', 'extractions', outputFolder)

# copy all code including this file - N.B. add any more local modules to list if needed
os.makedirs(os.path.join(project["outputPath"],'code'))
modulesToCopy = [getSopInstDict, radiomicAnalyser]
[shutil.copyfile(inspect.getfile(x), os.path.join(project['outputPath'], 'code', os.path.split(inspect.getfile(x))[1])) for x in modulesToCopy]
shutil.copyfile(__file__, os.path.join(project["outputPath"], 'code', os.path.split(__file__)[1]))
shutil.copyfile(project["paramFileName"], os.path.join(project["outputPath"], 'code', os.path.split(project["paramFileName"])[1]))


patientFolders = glob.glob(os.path.join(project["inputPath"], 'IAP_*'))
patientFolders.sort()

seriesUIDinfo = pd.read_csv(os.path.join(projectFolder, 'SeriesUIDinfoV2.csv'))
seriesUIDinfo = seriesUIDinfo.replace(np.nan, '')

resultsFiles = []
thumbnailFiles = []
thumbnailPathStr = 'thumbnails'

seriesUIDdict = {}
for patientFolder in patientFolders:

    if not os.path.isdir(patientFolder):
        continue

    # make dictionary to map seriesInstanceUID onto the folder containing this series
    for scanFolder in os.listdir(os.path.join(patientFolder, 'scans')):
        dcmFiles = glob.glob(os.path.join(patientFolder, 'scans', scanFolder, '**/*.dcm'), recursive=True)
        if len(dcmFiles) > 0:
            dcm = pydicom.read_file(dcmFiles[0])
            seriesUIDdict[dcm.SeriesInstanceUID] = os.path.split(dcmFiles[0])[0]



for patientFolder in patientFolders:
    if not os.path.isdir(patientFolder):
        continue

    patientID = os.path.split(patientFolder)[1].split('__II__')[0]
    patientNumber = int(patientID.replace('IAP_',''))

    # only process development data for now
    #if patientNumber > 237:
    #    continue

    # get row with SeriesInstanceUIDs for this patient
    seriesUIDinfo_rowIdx = np.where(seriesUIDinfo['patID']==patientID)[0]


    try:

        # almost all masks are stored as dicom images, but a few cases are as dicom seg files.  Need to do different things in each case

        segFiles = glob.glob(os.path.join(projectFolder, 'XNAT', 'assessors', patientID+'*.dcm'))

        if len(segFiles)>0:
            # THIS CODE FOR MASKS STORED AS DICOM-SEG

            for segFile in segFiles:
                radAn = radiomicAnalyser(project, segFile)
                radAn.sopInstDict, _, instanceNumDict, sopInst2instanceNumberDict, _ = getSopInstDict(patientFolder)
                radAn.extraDictionaries = {'instanceNumDict': instanceNumDict, 'sopInst2instanceNumberDict': sopInst2instanceNumberDict}
                radAn.loadImageData(includeExtraTopAndBottomSlices=True, includeContiguousEmptySlices=False)
                radAn.createMask()

                # when the masks are stored as images this messes up the way the patient name is used when writing the results files
                # Am misusing  fileSubscript and assessorFileName to make it behave sensibly
                radAn.assessorFileName = ''

                # trim off empty slices, but leave two either side
                slices = np.where(np.sum(np.sum(radAn.mask, axis=2), axis=1))[0]
                minSliceIdx = np.max([0, slices[0] - 2])
                maxSliceIdx = np.min([radAn.mask.shape[0] - 1, slices[-1] + 3])
                radAn.imageData['imageVolume'] = radAn.imageData['imageVolume'][minSliceIdx:maxSliceIdx, :, :]
                radAn.mask = radAn.mask[minSliceIdx:maxSliceIdx, :, :]

                if radAn.roiObjectLabelFound=='Repro':
                    radAn.StudyPatientName += '_repro'

                # no normalization
                radAn.computeRadiomicFeatures(normalize=False, featureKeyPrefixStr='noNormalize_')

                # with normalization of whole image
                radAn.computeRadiomicFeatures(normalize=True, featureKeyPrefixStr='normalized_')

                # normalization only to mask
                maskMean = np.mean(radAn.imageData['imageVolume'][radAn.mask==1])
                maskStd = np.std(radAn.imageData['imageVolume'][radAn.mask == 1])
                radAn.imageData['imageVolume'] = (radAn.imageData['imageVolume'] - maskMean)/maskStd
                radAn.computeRadiomicFeatures(normalize=False, featureKeyPrefixStr='maskNormalize_')

                resultsFiles.append(radAn.saveResult(fileSubscript=patientID))

                if False:
                    showMaskBoundary = True
                    showContours = False
                    showMaskHolesWithNewColour = True
                    thumbnail = radAn.saveThumbnail(fileStr='_' + patientID, titleStrExtra=patientID, pathStr=thumbnailPathStr,
                                                    showHistogram=False, showMaskHolesWithNewColour=True)
                    thumbnailFiles.append(thumbnail["fileName"])

        if len(seriesUIDinfo_rowIdx)>0:
            # THIS CODE FOR MASKS STORED AS IMAGES


            imageSeriesUIDinfo = seriesUIDinfo['imageSeriesUID'][seriesUIDinfo_rowIdx[0]]
            if imageSeriesUIDinfo=='':
                print('\033[1;31;48mNo image data found!\033[0;30;48m')
                print('Info from csv = ' + seriesUIDinfo['information'][seriesUIDinfo_rowIdx[0]])
                continue
            imageFolder = seriesUIDdict[imageSeriesUIDinfo]

            # skip if folder does not match patientFolder - this is because the folder structure for the test data is different and stupid!
            if patientFolder not in imageFolder:
                continue

            radAn = radiomicAnalyser(project, assessorFileName=imageFolder) #, sopInstDict=sopInstDict)

            # lesion mask
            lesionSeriesUID = seriesUIDinfo['lesionMaskSeriesUID'][seriesUIDinfo_rowIdx[0]]
            if lesionSeriesUID == '':
                print('\033[1;31;48mNo lesion mask found!\033[0;30;48m')
                print('Info from csv = ' + seriesUIDinfo['information'][seriesUIDinfo_rowIdx[0]] + ' ' + lesionSeriesUID)
                continue
            lesionFolder = seriesUIDdict[lesionSeriesUID]
            radAn.loadImageData(loadAllImagesFromFolder=lesionFolder, includeContiguousEmptySlices=False)
            mask = radAn.imageData['imageVolume']
            # IAP_247 has mask values around 15 for some very strange reason
            mask[mask>1] = 1
            radAn.mask = mask

            # images
            radAn.loadImageData(loadAllImagesFromFolder=imageFolder, includeContiguousEmptySlices=False)
            radAn.roiObjectLabelFound = 'Lesion'

            # when the masks are stored as images this messes up the way the patient name is used when writing the results files
            # Am misusing  fileSubscript and assessorFileName to make it behave sensibly
            radAn.assessorFileName = ''

            # trim off empty slices, but leave two either side
            slices = np.where(np.sum(np.sum(radAn.mask,axis=2), axis=1))[0]
            minSliceIdx = np.max([0, slices[0] - 2])
            maxSliceIdx = np.min([radAn.mask.shape[0]-1, slices[-1] + 3])
            radAn.imageData['imageVolume'] = radAn.imageData['imageVolume'][minSliceIdx:maxSliceIdx, :, :]
            radAn.mask = radAn.mask[minSliceIdx:maxSliceIdx, :, :]

            # no normalization
            radAn.computeRadiomicFeatures(normalize=False, featureKeyPrefixStr='noNormalize_')

            # with normalization of whole image
            radAn.computeRadiomicFeatures(normalize=True, featureKeyPrefixStr='normalized_')

            # normalization only to mask
            maskMean = np.mean(radAn.imageData['imageVolume'][radAn.mask == 1])
            maskStd = np.std(radAn.imageData['imageVolume'][radAn.mask == 1])
            radAn.imageData['imageVolume'] = (radAn.imageData['imageVolume'] - maskMean) / maskStd
            radAn.computeRadiomicFeatures(normalize=False, featureKeyPrefixStr='maskNormalize_')

            resultsFiles.append(radAn.saveResult(fileSubscript=patientID))

            if False:
                showMaskBoundary = True
                showContours = False
                showMaskHolesWithNewColour = True
                thumbnail = radAn.saveThumbnail(fileStr='_'+patientID, titleStrExtra=patientID, pathStr=thumbnailPathStr, showHistogram=False, showMaskHolesWithNewColour=True)
                thumbnailFiles.append(thumbnail["fileName"])


    except:
        print('\033[1;31;48m'+'_'*50)
        traceback.print_exc(file=sys.stdout)
        print('_'*50 + '\033[0;30;48m')


# combine separate .csv files into one
if len(resultsFiles)>0:
    resultsFiles.sort()
    csvList = []
    # get column headings from first file
    with open(resultsFiles[0]) as fo:
        s = fo.readlines()
        csvList.append(s[0])
    # data from each file
    for csvFile in resultsFiles:
        with open(csvFile) as fo:
            s = fo.readlines()
            csvList.append(s[1])
    # write combined .csv
    with open(os.path.join(project["outputPath"],'radiomicFeatures',outputFolder+'.csv'), 'w') as file_handler:
        for item in csvList:
            file_handler.write("{}".format(item))
    print('')
    print('Combined results written to file '+file_handler.name)

# combine thumbnail pdfs into one doc
if len(thumbnailFiles)>0:
    thumbnailFiles.sort()
    merger = PdfFileMerger()
    for pdf in thumbnailFiles:
        merger.append(pdf)
    merger.write(os.path.join(project["outputPath"], thumbnailPathStr, 'roiThumbnails.pdf'))
    merger.close()
