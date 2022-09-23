clear all
close all

[~,~,selectionData] = xlsread('/Users/morton/Dicom Files/Prostate_ECE/XNAT/ProjectScanInfo_Prostate_ECE_2022.05.09_12.16.57_annotated.xlsx','A1028:H1633');

warning off
ind = cell2mat(cellfunQ(@(x) strcmp(strrep(x,' ',''), 'x'), selectionData(:,8)));
warning on

selectionData = selectionData(ind,:);

seriesUIDinfo = containers.Map;
for n = 1:size(selectionData,1)
   scanFolders = dir(fullfile('/Users/morton/Dicom Files/Prostate_ECE/XNAT/experiments', [selectionData{n,2} '__II__' selectionData{n,3}], 'scans'));
   thisFolderIdx = find(cell2mat(arrayfunQ(@(x) startsWith(x.name, num2str(selectionData{n,5})), scanFolders)));
   thisFolderSearch = fullfile(scanFolders(thisFolderIdx).folder,scanFolders(thisFolderIdx).name, 'resources','DICOM','files','*.dcm');
   dcmFiles = dir(thisFolderSearch);
   disp(num2str(length(dcmFiles)))
   warning off
   info = dicominfo(fullfile(dcmFiles(1).folder, dcmFiles(1).name));
   warning on
   disp(info.SeriesInstanceUID)
   patID = selectionData{n,2};
   if seriesUIDinfo.isKey(patID)
       thisStruct = seriesUIDinfo(patID);
   else
       thisStruct = struct('lesionUID','','imageUID','');
   end

   if strcmp(selectionData{n,6}(end-2:end),'seg')
       thisStruct.lesionUID = info.SeriesInstanceUID;
   elseif strcmp(selectionData{n,6}(end-2:end),'reg')
       thisStruct.imageUID = info.SeriesInstanceUID;
   end
   seriesUIDinfo(patID) = thisStruct;
end

%%
clear patID

keys = seriesUIDinfo.keys;
for n = 1:length(keys)
    patID{n} = keys{n};
    information{n} = '';
    imageSeriesUID{n} = seriesUIDinfo(keys{n}).imageUID;
    lesionMaskSeriesUID{n} = seriesUIDinfo(keys{n}).lesionUID;
    prostateMaskSeriesUID{n} = '';
end

tbl = table(patID', information', imageSeriesUID', lesionMaskSeriesUID', prostateMaskSeriesUID');
tbl.Properties.VariableNames = {'patID', 'information', 'imageSeriesUID', 'lesionMaskSeriesUID', 'prostateMaskSeriesUID'};

writetable(tbl, '/Users/morton/Dicom Files/Prostate_ECE/SeriesUIDinfoTest.csv');