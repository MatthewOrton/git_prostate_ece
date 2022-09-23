clear all
close all

patFolders = dir('/Users/morton/Dicom Files/Prostate_ECE/Images/IAP*');

getPatID = @(x) ['IAP_' strrep(strrep(strrep(x,'IAP',''),'_',''),'-','')];

for n = 1:length(patFolders)
    patID{n} = patFolders(n).name(1:7);
    disp(patID{n})

    if length(patFolders(n).name)>7
        information{n} = patFolders(n).name(8:end);
    end
    
    % image series
    files = dir(fullfile(patFolders(n).folder, patFolders(n).name, 'REG_T2','*.dcm'));
    if length(files)>3
        info = dicominfo(fullfile(files(5).folder, files(5).name));
        if strcmp(getPatID(info.PatientID), patID{n})
            imageSeriesUID{n} = info.SeriesInstanceUID;
        else
            information{n} = ['xxx ' getPatID(info.PatientID) ' xxx'];
        end
    end
    
    % lesion mask series
    thisFolder = fullfile(patFolders(n).folder, patFolders(n).name, 'SEG_T2','*.dcm'); 
    files = dir(strrep(thisFolder, 'Images', 'Segs/lesion'));
    if length(files)>3
        info = dicominfo(fullfile(files(5).folder, files(5).name));
        if strcmp(getPatID(info.PatientID), patID{n})
            lesionMaskSeriesUID{n} = info.SeriesInstanceUID;
        else
            information{n} = ['xxx ' getPatID(info.PatientID) ' xxx'];
        end
    end
    
    % prostate mask series
    thisFolder = fullfile(patFolders(n).folder, patFolders(n).name, 'SEG_PROST','*.dcm'); 
    files = dir(strrep(thisFolder, 'Images', 'Segs/prostate'));
    if length(files)>3
        info = dicominfo(fullfile(files(5).folder, files(5).name));
        if strcmp(getPatID(info.PatientID), patID{n})
            prostateMaskSeriesUID{n} = info.SeriesInstanceUID;
        else
            information{n} = ['xxx ' getPatID(info.PatientID) ' xxx'];
        end
    end
    
end
information{n} = '';


% IAP_020 is a special case where the original lesion and prostate masks have been
% swapped, so swap back
idx = find(cell2mat(cellfunQ(@(x) strcmp(x, 'IAP_020'), patID)));
prUid = lesionMaskSeriesUID{idx};
leUid = prostateMaskSeriesUID{idx};
lesionMaskSeriesUID{idx} = leUid;
prostateMaskSeriesUID{idx} = prUid;

tbl = table(patID', information', imageSeriesUID', lesionMaskSeriesUID', prostateMaskSeriesUID');
tbl.Properties.VariableNames = {'patID', 'information', 'imageSeriesUID', 'lesionMaskSeriesUID', 'prostateMaskSeriesUID'};

writetable(tbl, '/Users/morton/Dicom Files/Prostate_ECE/SeriesUIDinfo3.csv');