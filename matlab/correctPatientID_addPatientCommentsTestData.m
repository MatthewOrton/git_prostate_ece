clear all

import org.dcm4che2.data.*
import icr.etherj.dicom.*

% initialise java objects
jToolkit = DicomToolkit.getToolkit();
jDcmRx = DicomReceiver();

patFolders = dir('/Volumes/BigBackup/PROSTATE controlo/anon/studyData/IAP*');

projectStr = 'Prostate_ECE';

for ip = 1:length(patFolders)
    
    sourceFolder = fullfile(patFolders(ip).folder, patFolders(ip).name);
    
    jScanner = jToolkit.createPathScan();
    jScanner.addContext(jDcmRx);
    
    % scan directory and sub-directories
    searchSubFolders = true;
    jScanner.scan(sourceFolder, searchSubFolders);
    jRoot = jDcmRx.getPatientRoot();

    
    seriesCount = 0;
    % get patient and check exactly one patient
    jPatList = jRoot.getPatientList();
    
    % get first studyUID and re-use so all data for each patient will end
    % up in the same study/experiment
    % studyUID = jPatList.get(0).getStudyList().get(0).getUid;
    % studyID = jPatList.get(0).getStudyList().get(0).getId;
    for iPat = 0:jPatList.size-1
        jPatient = jPatList.get(iPat);
        %patientID = ['IAP_' strrep(strrep(char(jPatient.getId),'IAP',''),'_','')];
        patientID = upper(strrep(patFolders(ip).name, ' ', '_'));
        jStudyList = jPatient.getStudyList();
        for iStudy = 0:jStudyList.size-1
            jStudy = jStudyList.get(iStudy);
            
            jSeriesList = jStudy.getSeriesList();
            for iSer = 0:jSeriesList.size-1
                
                jSeries = jSeriesList.get(iSer);
                
                % add index to start of seriesNumber so they will be unique
                % in each newly created study, but preserve the original
                % series number
                seriesCount = seriesCount + 1;
                seriesNumber = 10000*seriesCount + jSeries.getNumber;
                
                jInstList = jSeries.getSopInstanceList();
                for iInst = 0:jInstList.size-1
                    jInst = jInstList.get(iInst);
                    jDcm = jInst.getDicomObject();
                    newDcm = BasicDicomObject();
                    jDcm.copyTo(newDcm);
                    % newDcm.putString(Tag.SeriesNumber, VR.IS, num2str(seriesNumber));
                    newDcm.putString(Tag.PatientID, VR.LO, patientID);
                    newDcm.putString(Tag.PatientName, VR.LO, patientID);
                    % newDcm.putString(Tag.StudyInstanceUID, VR.UI, studyUID);
                    % newDcm.putString(Tag.StudyID, VR.SH, char(studyID));
                    sessionInfo = [char(jInst.getStudyDate) '_' num2str(round(str2double(jInst.getStudyTime))) '_' num2str(iPat+1) '_' num2str(iStudy+1)];
                    patientCommentsStr = ['Project: ' projectStr '; Subject: ' patientID '; Session: ' sessionInfo '; AA:True'];
                    newDcm.putString(Tag.PatientComments, VR.LT, patientCommentsStr);
                    newDcm.putString(Tag.InstitutionName, VR.LO, '');
                    newDcm.putString(Tag.InstitutionAddress, VR.ST, '');
                    newDcm.putString(Tag.PerformingPhysicianName, VR.PN, '');
                    newDcm.putString(Tag.OperatorsName, VR.PN, '');
                    newDcm.putString(Tag.ReferringPhysicianName, VR.PN, '');
                    newDcm.putString(Tag.StudyDescription, VR.LO, 'ProstateExam');
                    [sourceFolder, sourceFile] = fileparts(char(jInst.getFile));
                    destinFolder = strrep(sourceFolder,'studyData','studyDataCorrected');
                    if ~exist(destinFolder,'dir')
                        mkdir(destinFolder)
                    end
                    destinFile = fullfile(destinFolder, [sourceFile '.dcm']);
                    DicomUtils.writeDicomFile(newDcm, java.io.File(destinFile));
                end
            end
        end
    end
    disp(patFolders(ip).name)
end
