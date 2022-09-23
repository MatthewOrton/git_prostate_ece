import sys
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics')

import os, glob
from getSopInstDict import getSopInstDict
from radiomicAnalyser import radiomicAnalyser
from subprocess import call
import sys, traceback
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
project["outputPath"] = os.path.join(projectFolder, 'XNAT', 'roiThumbnails')

thumbnailPathStr = 'roiThumbnails_'+strftime("%Y.%m.%d_%H.%M.%S", localtime())

patientFolders = glob.glob(os.path.join(project["inputPath"], 'IAP_*'))
patientFolders.sort()
thumbnailFiles = []

seriesUIDinfo = pd.read_csv(os.path.join(projectFolder, 'SeriesUIDinfo.csv'))
seriesUIDinfo = seriesUIDinfo.replace(np.nan, '')


for patientFolder in patientFolders:

    if not os.path.isdir(patientFolder):
        continue

    patientID = os.path.split(patientFolder)[1].split('__II__')[0]
    patientNumber = int(patientID.replace('IAP_',''))

    # only process development data for now
    if patientNumber > 237:
        continue

    # get row with SeriesInstanceUIDs for this patient
    seriesUIDinfo_rowIdx = np.where(seriesUIDinfo['patID']==patientID)[0][0]


    try:

        # almost all masks are stored as dicom images, but a few cases are as dicom seg files.  Need to do different things in each case

        segFile = glob.glob(os.path.join(projectFolder, 'XNAT', 'assessors', patientID+'*.dcm'))

        if len(segFile)>1:
            print('\033[1;31;48mMore than one matching seg file found!\033[0;30;48m')
            continue

        if len(segFile)==1:
            # THIS CODE FOR MASKS STORED AS DICOM-SEG
            radAn = radiomicAnalyser(project, segFile[0])
            radAn.sopInstDict, _, instanceNumDict, sopInst2instanceNumberDict = getSopInstDict(patientFolder)
            radAn.extraDictionaries = {'instanceNumDict': instanceNumDict, 'sopInst2instanceNumberDict': sopInst2instanceNumberDict}
            radAn.loadImageData(includeExtraTopAndBottomSlices=True, includeContiguousEmptySlices=False)
            radAn.createMask()
            # just copy mask to mask delete as we don't have prostate masks in this case
            radAn.maskDelete = radAn.mask

        else:
            # THIS CODE FOR MASKS STORED AS IMAGES

            # make dictionary to map seriesInstanceUID onto the folder containing this series
            seriesUIDdict = {}
            for scanFolder in os.listdir(os.path.join(patientFolder, 'scans')):
                dcmFiles = glob.glob(os.path.join(patientFolder, 'scans', scanFolder, '**/*.dcm'), recursive=True)
                if len(dcmFiles)>0:
                    dcm = pydicom.read_file(dcmFiles[0])
                    seriesUIDdict[dcm.SeriesInstanceUID] = os.path.split(dcmFiles[0])[0]

            imageSeriesUIDinfo = seriesUIDinfo['imageSeriesUID'][seriesUIDinfo_rowIdx]
            if imageSeriesUIDinfo=='':
                print('\033[1;31;48mNo image data found!\033[0;30;48m')
                print('Info from csv = ' + seriesUIDinfo['information'][seriesUIDinfo_rowIdx])
                continue
            imageFolder = seriesUIDdict[imageSeriesUIDinfo]
            radAn = radiomicAnalyser(project, assessorFileName=imageFolder) #, sopInstDict=sopInstDict)

            lesionSeriesUID = seriesUIDinfo['lesionMaskSeriesUID'][seriesUIDinfo_rowIdx]
            if lesionSeriesUID == '':
                print('\033[1;31;48mNo lesion mask found!\033[0;30;48m')
                print('Info from csv = ' + seriesUIDinfo['information'][seriesUIDinfo_rowIdx] + ' ' + lesionSeriesUID)
                continue
            lesionFolder = seriesUIDdict[lesionSeriesUID]
            radAn.loadImageData(loadAllImagesFromFolder=lesionFolder, includeContiguousEmptySlices=False)
            radAn.maskDelete = radAn.imageData['imageVolume']

            # prostate mask
            prostateSeriesUID = seriesUIDinfo['prostateMaskSeriesUID'][seriesUIDinfo_rowIdx]
            if prostateSeriesUID == '':
                # if no prostate mask then set to the lesion mask
                radAn.mask = radAn.maskDelete
            else:
                prostateFolder = seriesUIDdict[prostateSeriesUID]
                radAn.loadImageData(loadAllImagesFromFolder=prostateFolder, includeContiguousEmptySlices=False)
                radAn.mask = radAn.imageData['imageVolume']

            # images last of all
            radAn.loadImageData(loadAllImagesFromFolder=imageFolder, includeContiguousEmptySlices=False)
            radAn.roiObjectLabelFound = 'Lesion'

        # trim off empty slices, but leave two either side
        slices = np.where(np.sum(np.sum(radAn.mask,axis=2), axis=1))[0]
        minSliceIdx = np.max([0, slices[0] - 2])
        maxSliceIdx = np.min([radAn.mask.shape[0]-1, slices[-1] + 3])
        radAn.imageData['imageVolume'] = radAn.imageData['imageVolume'][minSliceIdx:maxSliceIdx, :, :]
        radAn.mask = radAn.mask[minSliceIdx:maxSliceIdx, :, :]
        radAn.maskDelete = radAn.maskDelete[minSliceIdx:maxSliceIdx, :, :]

        showMaskBoundary = True
        showContours = False
        showMaskHolesWithNewColour = True
        thumbnail = radAn.saveThumbnail(fileStr='_'+patientID, titleStrExtra=patientID, pathStr=thumbnailPathStr, showHistogram=False, showMaskHolesWithNewColour=True)
        thumbnailFiles.append(thumbnail["fileName"])

    except:
        print('\033[1;31;48m'+'_'*50)
        traceback.print_exc(file=sys.stdout)
        print('_'*50 + '\033[0;30;48m')


# combine thumbnail pdfs into one doc
if len(thumbnailFiles)>0:
    thumbnailFiles.sort()
    merger = PdfFileMerger()
    for pdf in thumbnailFiles:
        merger.append(pdf)
    merger.write(os.path.join(project["outputPath"], thumbnailPathStr, 'roiThumbnails.pdf'))
    merger.close()


