import sys, os, shutil
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics')
from xnatDownloader import xnatDownloader
import glob
import pygit2
from time import strftime, localtime

serverURL = 'https://xnatcollaborations.icr.ac.uk/'

projectStr = 'Prostate_ECE'
downloadPath = '/Users/morton/Dicom Files/Prostate_ECE/XNAT'
assessorStyle = {"type": "SEG", "format": "DCM"}
roiCollectionLabelFilter = ''

xu = xnatDownloader(serverURL = serverURL,
                    projectStr=projectStr,
                    downloadPath=downloadPath,
                    removeSecondaryAndSnapshots=True,
                    assessorStyle=assessorStyle,
                    roiCollectionLabelFilter=roiCollectionLabelFilter)

# xu.getProjectDigest()
# xu.downloadAssessors_Project(destinFolder='assessors')
#xu.downloadExperiments_Project()


for nPat in range(238,299):
   xu.subjectList_downloadExperiments(['IAP_' + str(nPat)])