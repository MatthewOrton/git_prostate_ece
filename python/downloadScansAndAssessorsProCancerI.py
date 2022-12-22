import sys, os, shutil
sys.path.append('/Users/morton/Documents/git/git_icrpythonradiomics')
from xnatDownloader import xnatDownloader
import glob
import pygit2
from time import strftime, localtime

serverURL = 'https://xnatanon.icr.ac.uk/'

projectStr = 'PROCANCERI'
downloadPath = '/Users/morton/Dicom Files/Prostate_ECE/XNAT_ProCancerI'
assessorStyle = {"type": "SEG", "format": "DCM"}
roiCollectionLabelFilter = 'Index_lesion'

xu = xnatDownloader(serverURL = serverURL,
                    projectStr=projectStr,
                    downloadPath=downloadPath,
                    removeSecondaryAndSnapshots=True,
                    assessorStyle=assessorStyle,
                    roiCollectionLabelFilter=roiCollectionLabelFilter)

xu.downloadAssessors_Project(destinFolder='assessors')
#xu.downloadImagesReferencedByAssessors(keepEntireScan=True)