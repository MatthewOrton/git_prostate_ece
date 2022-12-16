import sys, glob, traceback, os
sys.path.append('/Users/morton/Documents/git/git_icrpythonradiomics')

import numpy as np
import pydicom
from getSeriesUIDFolderDict import getSeriesUIDFolderDict
from saveThumbnail import saveThumbnail
from dataLoader import dataLoader

xnatFolder = '/Users/morton/Dicom Files/Prostate_ECE/XNAT_ProCancerI'

# Some patients have multiple lesions stored in separate assessor files so get unique patient IDs
assessors = glob.glob(os.path.join(xnatFolder, 'assessors', '*.dcm'))
assessors.sort()

# dictionary to locate the images referenced by the rts files
seriesFolderDict = getSeriesUIDFolderDict(os.path.join(xnatFolder, 'referencedScans'))

# some DICOM series have an extra coronal reformat image as part of the series that we will discard up to this max limit
maxNonCompatibleInstances = 1

assessorsWithWarnings = []

for assessor in assessors:

    print('Processing ' + assessor)

    try:

        data = dataLoader(assessor, seriesFolderDict, maxNonCompatibleInstances=maxNonCompatibleInstances)

        # if not any(data.seriesData):
        #     continue

        if len(data.seriesData['ROINames'])==1:
            roiName = data.seriesData['ROINames'][0]
            thisLesion = data.getNamedROI(roiName, sliceSpacingUniformityThreshold=0.005)

            titleStr = str(rts.PatientName)
            if not thisLesion['maskContiguous']:
                titleStr += '\n' + r'$\bf{WARNING}$: contains volumes with missing slices'
                assessorsWithWarnings.append(assessor)
            else:
                titleStr += '\n '


        imageFilesUsed = saveThumbnail([thisLesion], thumbnailFile, volumePad=[5, 20, 20], titleStr=titleStr)

        # list all files in the same image folder and delete any that are not used
        allImageFiles = glob.glob(os.path.join(os.path.split(imageFilesUsed[0])[0], '*.dcm'))
        imageFilesNotUsed = list(set(allImageFiles) - set(imageFilesUsed))
        for file in imageFilesNotUsed:
            os.remove(file)

    except:
        print('\033[1;31;48m'+'_'*50)
        traceback.print_exc(file=sys.stdout)
        print('_'*50 + '\033[0;30;48m')

print('Assessors with warnings:')
print(assessorsWithWarnings)
