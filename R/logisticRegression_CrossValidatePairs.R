# clear environment and close figures
rm(list=ls())
if (dev.cur()!=1) {dev.off()}

setwd("/Users/morton/Documents/GitHub/prostate_ece_semantic/R")

library(dplyr)
library(caret)
library(Hmisc)
library(rms)
library(pROC)
library(formula.tools)

set.seed(123456)

# read data
dataFileName <- '/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/ECE_Prostate_Semantic/ECE_Semantic_Data/discovery.csv'
df <- read.csv(dataFileName)

# discard some features
df <- select(df, -c('PID', 'Gleason.biopsy', 'MeasurableECE'))
# MeasurableECE : is radiologists view on the target variable, so remove to avoid biasing

# one missing value, replace with median
df$PSA[is.na(df$PSA)] = median(df$PSA[!is.na(df$PSA)])

# df$log.PSA <- log(df$PSA)
# df <- select(df, -c('PSA'))
# 
# df$log.MajorLengthIndex <- log(df$MajorLengthIndex)
# df <- select(df, -c('MajorLengthIndex'))
# 
# df$log.ProstateVolume <- log(df$ProstateVolume)
# df <- select(df, -c('ProstateVolume'))

targetName <- 'ECE_Pathology'

featureNames <- colnames(select(df, -c(targetName)))
continuousFeatures <- c('PSA', 'MajorLengthIndex', 'ProstateVolume', 'CapsularContactLength')
factorFeatures <- names(df)[!featureNames %in% continuousFeatures]

formulaList <- c(NULL)
for (i in seq(length(featureNames)-1)){
  for (j in seq(length(featureNames)-i)){
    feat1 <- featureNames[i]
    feat2 <- featureNames[i+j]
    if ((feat1 %in% continuousFeatures) & !(feat2 %in% continuousFeatures)){
      # one continuous, rcs order = 4
      formulaList <- c(formulaList, as.formula(paste(targetName, ' ~ rcs(', feat1 ,',4) + ', feat2, sep='')))
      #formulaList <- c(formulaList, as.formula(paste(targetName, ' ~ ', feat1 ,' * ', feat2, sep='')))
    } 
    else if (!(feat1 %in% continuousFeatures) & (feat2 %in% continuousFeatures)){
      # one continuous, rcs order = 4
      formulaList <- c(formulaList, as.formula(paste(targetName, ' ~ rcs(', feat2 ,',4) + ', feat1, sep='')))
      #formulaList <- c(formulaList, as.formula(paste(targetName, ' ~ ', feat2 , ' * ', feat1, sep='')))
    }
    # else if ((feat1 %in% continuousFeatures) & (feat2 %in% continuousFeatures)){
    #   # both continuous, rcs order = 3
    #   formulaList <- c(formulaList, as.formula(paste(targetName, ' ~ rcs(', feat1 ,',3) + rcs(', feat2, ',3)', sep='')))
    # } 
    # else if (!(feat1 %in% continuousFeatures) & !(feat2 %in% continuousFeatures)){
    #   # both discrete, no rcs
    #   formulaList <- c(formulaList, as.formula(paste(targetName, ' ~ ', feat2 ,' + ', feat1, sep='')))
    # }
  }
}


ddist <- datadist(df)
options(datadist='ddist')

fitTwoParameters <- function(df, targetName, formulaList){

  nFolds <- 5
  nRepeats <- 5

  CindexCV <- replicate(length(formulaList), 0)
  for (j in seq(nRepeats)){
    flds <- createFolds(df[,targetName], k = nFolds, list = TRUE, returnTrain = FALSE)
    for (i in seq(nFolds)){
      dfTrain <- df[-flds[[i]],]
      dfTest  <- df[ flds[[i]],]
    
      fits <- lapply(formulaList, lrm, data=dfTrain)
      CindexCV <- CindexCV + sapply(fits, function(x){auc(roc(dfTest$ECE_Pathology, predict(x, dfTest), levels=c(0,1), direction='<'))})
    }
  }

  idx <- which.max(CindexCV)
  model <- lrm(formulaList[[idx]], df)
  return(list('model'=model, 'modelIndex'=idx))
}

resultResub <- fitTwoParameters(df, targetName, formulaList)
AUCresub <- auc(roc(df$ECE_Pathology, predict(resultResub[['model']], df), levels=c(0,1), direction='<'))
cat('\n')
cat('___________________________________\n')
cat('AUC resub     =', AUCresub, '\n')
thisModel <- Reduce(paste, deparse(resultResub[['model']]$terms[[3]]))
cat(gsub(', 3[)]', '', gsub(', 4[)]', '', gsub('rcs[(]', '', c(as.character(thisModel))))),'\n')
cat('\n')

nBoot <- 200
AUCoptimism <- replicate(nBoot, 0)

modelCount <- replicate(length(formulaList), 0)
i <- 1
fitErrorCount <- 0
while (i<nBoot){
  tryCatch({
    idxBoot <- createResample(df[,targetName], 1)
    dfBoot <- df[idxBoot[[1]],]
    resultBoot <- fitTwoParameters(dfBoot, targetName, formulaList)
    modelCount[resultBoot[['modelIndex']]] <- modelCount[resultBoot[['modelIndex']]] + 1
    AUCfit <- auc(roc(dfBoot$ECE_Pathology, predict(resultBoot[['model']], dfBoot), levels=c(0,1), direction='<'))
    AUCall <- auc(roc(df$ECE_Pathology, predict(resultBoot[['model']], df), levels=c(0,1), direction='<'))
    AUCoptimism[i] <- AUCfit - AUCall
    thisModel <- Reduce(paste, deparse(resultBoot[['model']]$terms[[3]]))
    cat(i, ' : ', gsub('ECE_Pathology ~ ', '', gsub(', 3[)]', '', gsub(', 4[)]', '', gsub('rcs[(]', '', c(as.character(thisModel)))))),'\n')
    i <- i + 1
  },
    error = function(err){
    fitErrorCount <- fitErrorCount + 1
      cat('Fit error\n')
  })
}
  

cat('\n')
cat('AUC resub     =', AUCresub, '\n')
cat('AUC optimism  =', mean(AUCoptimism), '\n')
cat('AUC bootstrap =', AUCresub - mean(AUCoptimism), '\n')
cat('\n')

modelIdx <- which(modelCount>0)
modelIdx <- modelIdx[order(-modelCount[modelIdx])]

for (k in modelIdx){
  cat(modelCount[k], ' : ', gsub('ECE_Pathology ~ ', '', gsub(', 3[)]', '', gsub(', 4[)]', '', gsub('rcs[(]', '', c(as.character(as.character(formulaList[k]))))))), '\n', sep='')
}
 














# # event frequency summary for all variables
# pdf("summaryAll.pdf", width = 8, height = 15) 
# plot(summary(ECE_Pathology ~ ., data=df))
# dev.off()
# 
# 
# # TumourGradeMRI = 'I' has similar event rate to "no", so combine
# df$TumorGradeMRI[df$TumorGradeMRI=='I'] <- 'no'
# 
# # get lists of continuous and factor features
# continuousFeatures <- c('ProstateVolume', 'MajorLengthIndex', 'CapsularContactLength', 'PSA')
# factorFeatures <- names(df)[!names(df) %in% c('ECE_Pathology', continuousFeatures)]
# 
# pdf("continuousVariableHistograms.pdf", width = 8, height = 8) 
# par(mfrow=c(2,2))
# for (cf in continuousFeatures){
#   hist(df[,cf], xlab=cf, ylab='Frequency', main='', breaks=15)
#   # df$single <- df[,cf]
#   # ddist <- datadist(df$ECE_Pathology, df$single)
#   # options(datadist='ddist')
#   # f <- lrm(ECE_Pathology ~ single, df, x=TRUE, y=TRUE)
#   # lines(df$single, f$linear.predictors)
#   # df <- select(df, -c('single'))
# }
# dev.off()
# 
# ####################################################
# # histograms suggest log-transforming these features
# ####################################################
# df$log.PSA <- log(df$PSA)
# df <- select(df,-c('PSA'))
# df$log.ProstateVolume <- log(df$ProstateVolume)
# df <- select(df,-c('ProstateVolume'))
# df$log.MajorLengthIndex <- log(df$MajorLengthIndex)
# df <- select(df,-c('MajorLengthIndex'))
# 
# continuousFeatures = c('log.ProstateVolume', 'log.MajorLengthIndex', 'CapsularContactLength', 'log.PSA')
# 
# 
# pdf("continuousVariableHistograms_transformed.pdf", width = 8, height = 8) 
# par(mfrow=c(2,2))
# for (cf in continuousFeatures){
#   hist(df[,cf], xlab=cf, ylab='Frequency', main='', breaks=15)
#   # df$single <- df[,cf]
#   # ddist <- datadist(df$ECE_Pathology, df$single)
#   # options(datadist='ddist')
#   # f <- lrm(ECE_Pathology ~ single, df, x=TRUE, y=TRUE)
#   # lines(df$single, f$linear.predictors)
#   # df <- select(df, -c('single'))
# }
# dev.off()
# 
# ####################################################
# # outcome probability curves for continuous variables to determine if transformations are needed
# ####################################################
# pdf("continuousVariableEffects.pdf", width = 8, height = 8) 
# par(mfrow=c(2,2))
# for (cf in continuousFeatures){
#   plsmo(df[,cf], df[,'ECE_Pathology'], method='lowess', xlab=cf, ylab='logit Pr(ECE=1)', datadensity=TRUE, fun=qlogis, ylim=c(-4,4))
#   # df$single <- df[,cf]
#   # ddist <- datadist(df$ECE_Pathology, df$single)
#   # options(datadist='ddist')
#   # f <- lrm(ECE_Pathology ~ single, df, x=TRUE, y=TRUE)
#   # lines(df$single, f$linear.predictors)
#   # df <- select(df, -c('single'))
# }
# dev.off()
# 
# 
# 
# ####################################################
# # curves suggest linear effects should be sufficient
# ####################################################
# 
# ####################################################
# # outcome prob curves for continuous/discrete interactions
# ####################################################
# 
# for (cf in continuousFeatures){
#   pdf(paste('Interactions_', cf, '.pdf', sep=''), width = 10, height = 8)
#   par(mfrow=c(3,5))
#   for (ff in factorFeatures){
#     # note: may need to use special version of logit transform on y-axis as there are some groups where event rate = 0
#     plsmo(df[,cf], df[,'ECE_Pathology'], method='lowess', xlab=cf, ylab='logit Pr(ECE=1)', group=df[,ff], datadensity=TRUE, ylim=c(-6,6), fun=function(x){log((0.01+x)/(1-x))})
#     title(ff)
#   }
#   dev.off()
# }
# 
# 
# ####################################################
# # These plots indicate interactions are present between continuous and discrete variables.  Given data size we will *not* use non-linear terms for any interactions
# ####################################################
# 
# 
# ddist <- datadist(df)
# options(datadist = 'ddist')
# attach(df)
# 
# ####################################################
# # test if interactions between continous features are needed
# ####################################################
# print(anova(lrm(ECE_Pathology ~ (log.ProstateVolume + log.MajorLengthIndex + CapsularContactLength + log.PSA)^2)))
# ####################################################
# # No!
# ####################################################
# 
# 
# ####################################################
# # test if the AnatDev group of features are needed (assuming they do not interact with other discrete features)
# ####################################################
# print(anova(lrm(ECE_Pathology ~ (log.ProstateVolume + log.MajorLengthIndex + CapsularContactLength + log.PSA)*(AnatDev01 + AnatDev02 + AnatDev03 + AnatDev04))))
# ####################################################
# # anova suggests removing all AnatDev features
# ####################################################
# 
# 
# ####################################################
# # Test all discrete interactions - we have to avoid IrregularContour*BlackEstritionPeripFat because this has frequency=0 for 1/4 combinations
# ####################################################
# print(anova(lrm(ECE_Pathology ~ (GleasonBinary + IndLesPIRADS_V2 + TumorGradeMRI + SmoothCapsularBulging + CapsularDisruption + UnsharpMargin + IrregularContour + BlackEstritionPeripFat + highsignalT1FS)* 
#                                 (GleasonBinary + IndLesPIRADS_V2 + TumorGradeMRI + SmoothCapsularBulging + CapsularDisruption + UnsharpMargin + highsignalT1FS))))
# ####################################################
# # anova suggests no interactions between discrete features are needed
# ####################################################
# 
# # doesn't like PSA!!
# f.final <- lrm(ECE_Pathology ~ (log.MajorLengthIndex + CapsularContactLength)*(GleasonBinary + IndLesPIRADS_V2 + TumorGradeMRI + SmoothCapsularBulging + CapsularDisruption + UnsharpMargin + IrregularContour + BlackEstritionPeripFat + highsignalT1FS))
# print(f.final)
# poop
# 
# f.final <- update(f.final, x=T, y=T)
# print(validate(f.final, B=100))
# poop
# 
# ddist <- datadist(ECE_Pathology, CapsularContactLength, IrregularContour)
# options(datadist='ddist')
# 
# f <- lrm(ECE_Pathology ~ CapsularContactLength + IrregularContour)
# pdf('Model.pdf', width = 8, height = 6)
# print(plot(Predict(f, CapsularContactLength), ylim=c(-6,4)))
# dev.off()
# pdf('Data.pdf', width = 8, height = 6)
# plsmo(CapsularContactLength, ECE_Pathology, method='lowess', xlab='CapsularContactLength', ylab='logit Pr(ECE=1)', group=IrregularContour, datadensity=TRUE, fun=qlogis, ylim=c(-6,4)) 
# dev.off()
# print(f)
# 
# f <- update(f, x=T, y=T)
# set.seed(131)
# val <- validate(f, B=100)
# print(val)
# 
# pdf('calibrationPlot.pdf', width = 8, height = 8)
# cal <- calibrate(f, B=100)
# plot(cal, xlim=c(0,1), ylim=c(0,1))
# title(paste(as.character(f$sformula)[2], ' ~ ', as.character(f$sformula)[3], sep=''))
# dev.off()
