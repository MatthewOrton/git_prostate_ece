# clear environment and close figures
rm(list=ls())
if (dev.cur()!=1) {dev.off()}

setwd("/Users/morton/Documents/GitHub/prostate_ece_semantic/R")

# library(stringr)
# library(survival)
# library(survminer)
# library(glmnet)
# library(ggplot2)
# library(stringr)
 library(dplyr)
# library(caret)
# library(mltools)
# library(data.table)
library(Hmisc)
library(rms)
# library(randomForestSRC)
# library(tidyverse)


# read data
dataFileName <- '/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/ECE_Prostate_Semantic/ECE_Semantic_Data/discovery.csv'
df <- read.csv(dataFileName)

# discard some features
df <- select(df, -c('PID', 'Gleason.biopsy', 'MeasurableECE'))
# MeasurableECE : is radiologists view on the target variable, so remove to avoid biasing

#########
# !!!!!!!
#########
# remove RetroprostaticAngleOblit=1 cases as these are all in the positive class
df <- df[df$RetroprostaticAngleOblit=="NO",]
df <- select(df, -c('RetroprostaticAngleOblit'))

# one missing value, replace with median
df$PSA[is.na(df$PSA)] = median(df$PSA[!is.na(df$PSA)])

# NO/YES to 0, 1
for (feat in c('SmoothCapsularBulging', 'CapsularDisruption', 'UnsharpMargin', 'IrregularContour', 'BlackEstritionPeripFat', 'highsignalT1FS')){ #'RetroprostaticAngleOblit',
  df[feat] <- as.numeric(df[feat] == "YES") 
}
df$GleasonBinary <- df$GleasonBinary
# make target numeric
df$ECE_Pathology <- as.numeric(df$ECE_Pathology)

# event frequency summary for all variables
pdf("summaryAll.pdf", width = 8, height = 15) 
plot(summary(ECE_Pathology ~ ., data=df))
dev.off()


df$TumorGradeMRI.I <- as.numeric(df$TumorGradeMRI=='I')
df$TumorGradeMRI.no <- as.numeric(df$TumorGradeMRI=='no')
df <- select(df, -c('TumorGradeMRI'))

# get lists of continuous and factor features
continuousFeatures <- c('ProstateVolume', 'MajorLengthIndex', 'CapsularContactLength', 'PSA')
factorFeatures <- names(df)[!names(df) %in% c('ECE_Pathology', continuousFeatures)]

pdf("continuousVariableHistograms.pdf", width = 8, height = 8) 
par(mfrow=c(2,2))
for (cf in continuousFeatures){
  hist(df[,cf], xlab=cf, ylab='Frequency', main='', breaks=15)
  # df$single <- df[,cf]
  # ddist <- datadist(df$ECE_Pathology, df$single)
  # options(datadist='ddist')
  # f <- lrm(ECE_Pathology ~ single, df, x=TRUE, y=TRUE)
  # lines(df$single, f$linear.predictors)
  # df <- select(df, -c('single'))
}
dev.off()

####################################################
# histograms suggest log-transforming these features
####################################################
df$log.PSA <- log(df$PSA)
df <- select(df,-c('PSA'))
df$log.ProstateVolume <- log(df$ProstateVolume)
df <- select(df,-c('ProstateVolume'))
df$log.MajorLengthIndex <- log(df$MajorLengthIndex)
df <- select(df,-c('MajorLengthIndex'))

continuousFeatures = c('log.ProstateVolume', 'log.MajorLengthIndex', 'CapsularContactLength', 'log.PSA')


pdf("continuousVariableHistograms_transformed.pdf", width = 8, height = 8) 
par(mfrow=c(2,2))
for (cf in continuousFeatures){
  hist(df[,cf], xlab=cf, ylab='Frequency', main='', breaks=15)
  # df$single <- df[,cf]
  # ddist <- datadist(df$ECE_Pathology, df$single)
  # options(datadist='ddist')
  # f <- lrm(ECE_Pathology ~ single, df, x=TRUE, y=TRUE)
  # lines(df$single, f$linear.predictors)
  # df <- select(df, -c('single'))
}
dev.off()

####################################################
# outcome probability curves for continuous variables to determine if transformations are needed
####################################################
pdf("continuousVariableEffects.pdf", width = 8, height = 8) 
par(mfrow=c(2,2))
for (cf in continuousFeatures){
  plsmo(df[,cf], df[,'ECE_Pathology'], method='lowess', xlab=cf, ylab='logit Pr(ECE=1)', datadensity=TRUE, fun=qlogis, ylim=c(-4,4))
  # df$single <- df[,cf]
  # ddist <- datadist(df$ECE_Pathology, df$single)
  # options(datadist='ddist')
  # f <- lrm(ECE_Pathology ~ single, df, x=TRUE, y=TRUE)
  # lines(df$single, f$linear.predictors)
  # df <- select(df, -c('single'))
}
dev.off()



####################################################
# curves suggest linear effects should be sufficient
####################################################

####################################################
# outcome prob curves for continuous/discrete interactions
####################################################

for (cf in continuousFeatures){
  pdf(paste('Interactions_', cf, '.pdf', sep=''), width = 10, height = 8)
  par(mfrow=c(3,5))
  for (ff in factorFeatures){
    # note: may need to use special version of logit transform on y-axis as there are some groups where event rate = 0
    plsmo(df[,cf], df[,'ECE_Pathology'], method='lowess', xlab=cf, ylab='logit Pr(ECE=1)', group=df[,ff], datadensity=TRUE, ylim=c(-6,6), fun=function(x){log((0.01+x)/(1-x))})
    title(ff)
  }
  dev.off()
}


####################################################
# These plots indicate interactions are present between continuous and discrete variables.  Given data size we will *not* use non-linear terms for any interactions
####################################################


ddist <- datadist(df)
options(datadist = 'ddist')
attach(df)

####################################################
# test if interactions between continous features are needed
####################################################
print(anova(lrm(ECE_Pathology ~ (log.ProstateVolume + log.MajorLengthIndex + CapsularContactLength + log.PSA)^2)))
####################################################
# No!
####################################################


####################################################
# test if the AnatDev group of features are needed (assuming they do not interact with other discrete features)
####################################################
print(anova(lrm(ECE_Pathology ~ (log.ProstateVolume + log.MajorLengthIndex + CapsularContactLength + log.PSA)*(AnatDev01 + AnatDev02 + AnatDev03 + AnatDev04))))
####################################################
# anova suggests removing all AnatDev features
####################################################


####################################################
# Test all discrete interactions - we have to avoid IrregularContour*BlackEstritionPeripFat because this has frequency=0 for 1/4 combinations
####################################################
# print(anova(lrm(ECE_Pathology ~ (GleasonBinary + IndLesPIRADS_V2 + TumorGradeMRI.I + TumorGradeMRI.no + SmoothCapsularBulging + CapsularDisruption + UnsharpMargin + IrregularContour + BlackEstritionPeripFat + highsignalT1FS)* 
#                                (GleasonBinary + IndLesPIRADS_V2 + TumorGradeMRI.I + TumorGradeMRI.no + SmoothCapsularBulging + CapsularDisruption + UnsharpMargin + highsignalT1FS))))
####################################################
# anova suggests no interactions between discrete features are needed
####################################################

# doesn't like PSA!!
f.final <- lrm(ECE_Pathology ~ (log.MajorLengthIndex + CapsularContactLength)*(GleasonBinary + IndLesPIRADS_V2 + TumorGradeMRI.I + TumorGradeMRI.no + SmoothCapsularBulging + CapsularDisruption + UnsharpMargin + IrregularContour + BlackEstritionPeripFat + highsignalT1FS))
print(f.final)


f.final <- update(f.final, x=T, y=T)
print(validate(f.final, B=100))


ddist <- datadist(ECE_Pathology, CapsularContactLength, CapsularDisruption, GleasonBinary, RetroprostaticAngleOblit, IrregularContour)
options(datadist='ddist')

f <- lrm(ECE_Pathology ~ CapsularContactLength + IrregularContour)
pdf('Model.pdf', width = 8, height = 6)
print(plot(Predict(f, CapsularContactLength), ylim=c(-6,4)))
dev.off()
pdf('Data.pdf', width = 8, height = 6)
plsmo(CapsularContactLength, ECE_Pathology, method='lowess', xlab='CapsularContactLength', ylab='logit Pr(ECE=1)', group=IrregularContour, datadensity=TRUE, fun=qlogis, ylim=c(-6,4)) 
dev.off()
print(f)

f <- update(f, x=T, y=T)
set.seed(131)
val <- validate(f, B=100)
print(val)

pdf('calibrationPlot.pdf', width = 8, height = 8)
cal <- calibrate(f, B=100)
plot(cal, xlim=c(0,1), ylim=c(0,1))
title(paste(as.character(f$sformula)[2], ' ~ ', as.character(f$sformula)[3], sep=''))
dev.off()
