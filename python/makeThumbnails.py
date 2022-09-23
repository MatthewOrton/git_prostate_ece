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

project = {}
project["projectStr"] = 'RADSARC-R'
project["inputPath"] = '/Users/morton/Dicom Files/Prostate_ECE'
project["assessorStyle"] = {"type": "SEG", "format": "DCM"}
project["roiObjectLabelFilter"] = ''
project["paramFileName"] = '' #/Users/morton/Dicom Files/RADSARC_R/Params.yaml'
project["outputPath"] = os.path.join(project["inputPath"], 'roiThumbnails')

thumbnailPathStr = 'roiThumbnails_'+strftime("%Y.%m.%d_%H.%M.%S", localtime())

sopInstDict, _, _, _ = getSopInstDict(project["inputPath"])

assessors = glob.glob(os.path.join(project["inputPath"],'Segs', 'lesion', 'IAP_*'))
assessors = [x for x in assessors if '(no index lesion)' not in x]
assessors.sort()
thumbnailFiles = []
assessors = [assessors[86]]

for n, assessor in enumerate(assessors):
    try:

        lesionFolder = os.path.join(assessor, 'SEG_T2')
        prostateFolder = lesionFolder.replace('lesion','prostate').replace('SEG_T2','SEG_PROST')
        imageFolder = os.path.join(assessor.replace('Segs/lesion','Images'), 'REG_T2')

        radAn = radiomicAnalyser(project, assessorFileName=lesionFolder, sopInstDict=sopInstDict)

        # In this study the segmentations are actually stored as dicom images not dicomSeg.  DUUUURRRR!!  
        # Use loadImageData to get the masks and transfer manually into the mask variable
        # Set maskDelete to be for the lesion, and mask to be for the whole prostate
        # lesion mask
        radAn.loadImageData(loadAllImagesFromFolder=lesionFolder, includeContiguousEmptySlices=False)
        radAn.maskDelete = radAn.imageData['imageVolume']
        zPosLesion = [x[2] for x in radAn.imageData['imagePositionPatient']]
        # prostate mask
        radAn.loadImageData(loadAllImagesFromFolder=prostateFolder, includeContiguousEmptySlices=False)
        radAn.mask = radAn.imageData['imageVolume']
        zPosProstate = [x[2] for x in radAn.imageData['imagePositionPatient']]
        # images
        radAn.loadImageData(loadAllImagesFromFolder=imageFolder, includeContiguousEmptySlices=False)
        radAn.roiObjectLabelFound = 'Lesion'
        zPosImage = [x[2] for x in radAn.imageData['imagePositionPatient']]

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
        patientID = 'IAP_'+radAn.StudyPatientName.replace('IAP','').replace('_','') # some of them have 'IAP_' in the string already, and some do not.  Make sure we only have one copy of 'IAP_'
        vmin = 0
        vmax = 1023
        # vmin = vmin, vmax = vmax,
        thumbnail = radAn.saveThumbnail(fileStr='_'+patientID, titleStrExtra=patientID, pathStr=thumbnailPathStr, showHistogram=False, showMaskHolesWithNewColour=True) # ) #showMaskBoundary=showMaskBoundary, showContours = showContours, linewidth=0.04, showMaskHolesWithNewColour=showMaskHolesWithNewColour,
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


