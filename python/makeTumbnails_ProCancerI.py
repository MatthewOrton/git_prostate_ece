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

        # some are named 'Index_lesion' and some 'Index_lesion '
        if len(data.seriesData['ROINames'])==1 and 'Index_lesion' in data.seriesData['ROINames'][0]:
            roiName = data.seriesData['ROINames'][0]
        else:
            print('No ROI called "Index_lesion" found!')
            continue

        thisLesion = data.getNamedROI(roiName, sliceSpacingUniformityThreshold=0.005)

        titleStr = os.path.split(assessor)[1].split('__II__')[0]
        if not thisLesion['maskContiguous']:
            titleStr += '\n' + r'$\bf{WARNING}$: contains volumes with missing slices'
            assessorsWithWarnings.append(assessor)
        else:
            titleStr += '\n '

        thumbnailFile = assessor.replace('assessors', 'Thumbnails').replace('.dcm','.pdf')

        saveThumbnail([thisLesion], thumbnailFile, volumePad=[2, 50, 50], imageGrayLevelLimits=[0, 1000], titleStr=titleStr)

        print('\n')

    except:
        print('\033[1;31;48m'+'_'*50)
        traceback.print_exc(file=sys.stdout)
        print('_'*50 + '\033[0;30;48m')

print('Assessors with warnings:')
print(assessorsWithWarnings)
