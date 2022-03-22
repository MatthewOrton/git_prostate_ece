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
library(glmnet)

set.seed(123456)

# read data
dataFileName <- '/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/ECE_Prostate_Semantic/ECE_Semantic_Data/discovery.csv'
df <- read.csv(dataFileName)

print(table(df[, c('MeasurableECE', 'ECE_Pathology')]))

# discard some features
df <- select(df, -c('PID', 'Gleason.biopsy', 'MeasurableECE'))
# MeasurableECE : is radiologists view on the target variable, so remove to avoid biasing

# one missing value, replace with median
df$PSA[is.na(df$PSA)] = median(df$PSA[!is.na(df$PSA)])

# binarise TumourGradeMRI
df$TumorGradeMRI.I <- as.numeric(df$TumorGradeMRI=='I')
df$TumorGradeMRI.no <- as.numeric(df$TumorGradeMRI=='no')
df <- select(df, -c('TumorGradeMRI'))

# NO/YES to 0, 1
for (feat in c('SmoothCapsularBulging', 'CapsularDisruption', 'UnsharpMargin', 'IrregularContour', 'BlackEstritionPeripFat', 'highsignalT1FS', 'RetroprostaticAngleOblit')){
  df[feat] <- as.numeric(df[feat] == "YES") 
}
df$GleasonBinary <- df$GleasonBinary
# make target numeric
df$ECE_Pathology <- as.numeric(df$ECE_Pathology)

targetName <- 'ECE_Pathology'

fitNParameters <- function(df, targetName, nFeatures){

  x <- scale(as.matrix(select(df, -c('ECE_Pathology'))))
  y <- df$ECE_Pathology
  
  lambda <- exp(seq(from=-5, to=-0, length.out=1000))
  f.all <- glmnet(x, y, family='binomial', lambda=lambda)
  # Find lambda index where number of non-zero features = nFeatures
  # Note that the first occurence is chosen as this corresponds to the weakest lambda that gives the desired number of non-zero features
  # If required feature count not found then look for one less feature
  if (sum(f.all$df == nFeatures) == 0){
    nFeatures <- nFeatures - 1
  }
  if (sum(f.all$df == nFeatures) == 0){
    nFeatures <- nFeatures - 1
    browser()
  }
  lambda.best <- min(f.all$lambda[which(f.all$df == nFeatures)])
  f.best <- glmnet(x, y, family='binomial', lambda=lambda.best)
  return(list("model"=f.best, "scaledData"=x))
}


nFeatures <- 6
cat('\nN features =', nFeatures ,'\n')

f.all <- fitNParameters(df, targetName, nFeatures)
model.all <- f.all[['model']]
scaledData <- f.all[['scaledData']]
score.all <- as.vector(predict(model.all, newx=scaledData))
AUC.all <- auc(roc(df$ECE_Pathology, score.all, levels=c(0,1), direction='<'))
cat('\nAUC (resub) = ', AUC.all, '\n', sep='')
cat(paste(rownames(model.all$beta)[which(model.all$beta != 0)], collapse = ' + '), '\n\n')

# plot(roc(df$ECE_Pathology, score.all, levels=c(0,1), direction='<'), xlim=c(1,0), ylim=c(0,1))
# points(123/(123+2), 20/(20+24), pch=4, cex=4)

nBoot <- 1000
AUCoptimism <- replicate(nBoot, 0)
featureSelected <- matrix(0, ncol(df)-1, nBoot)

for (iBoot in seq(nBoot)){
  idxBoot <- createResample(df[,targetName], 1)$Resample1
  
  dfBoot <- df[idxBoot, ]
  f <- fitNParameters(dfBoot, targetName, nFeatures)
  model <- f[['model']]
  scaler <- f[['scaledData']]
  
  matBoot <- scale(as.matrix(select(dfBoot, -c(targetName))), attr(scaler, "scaled:center"), attr(scaler, "scaled:scale"))
  scoreBoot <- as.vector(predict(model, newx=matBoot))
  
  matTest <- scale(as.matrix(select(df, -c(targetName))), attr(scaler, "scaled:center"), attr(scaler, "scaled:scale"))
  scoreTest <- as.vector(predict(model, newx=matTest))
  
  AUCboot <- auc(roc(dfBoot$ECE_Pathology, scoreBoot, levels=c(0,1), direction='<'))
  AUCtest <- auc(roc(df$ECE_Pathology,     scoreTest, levels=c(0,1), direction='<'))
  AUCoptimism[iBoot] <- AUCboot - AUCtest
  
  featureSelected[,iBoot] <- as.numeric(model$beta != 0)
}


cat('AUC optimism  =', mean(AUCoptimism), '\n')
cat('AUC bootstrap =', AUC.all - mean(AUCoptimism), '\n\n')


rownames(featureSelected) <- rownames(model$beta)
featureFreq <- t(rowSums(featureSelected))
#print(featureFreq[,featureFreq > 0])

featureCombSig <- as.vector(t(as.vector(2^(seq(nrow(featureSelected))-1))) %*% featureSelected)
featureCombFreq <- as.data.frame(table(featureCombSig))

for (i in order(-featureCombFreq$Freq)[1:10]){
  idx <- which(featureCombSig==featureCombFreq$featureCombSig[i])[1] 
  cat(featureCombFreq$Freq[i], ' : ', paste(rownames(featureSelected)[featureSelected[,idx]==1], collapse=' + '), '\n\n', sep='')
}

cf <- 'CapsularContactLength'
par(mfrow=c(1,3))
for (bf in c('IrregularContour', 'GleasonBinary', 'CapsularDisruption')){
  plsmo(df[,cf], df[,targetName], method='lowess', xlab=cf, ylab='logit Pr(ECE=1)', group=df[,bf], datadensity=TRUE, ylim=c(-6,6), fun=function(x){log((0.01+x)/(1.01-x))})
  title(bf)
}


ddist <- datadist(df)
options(datadist='ddist')

f <- lrm(ECE_Pathology ~ rcs(CapsularContactLength,3)*(GleasonBinary + CapsularDisruption + IrregularContour))
f <- update(f, x=T, y=T)
val <- validate(f, B=100)
cat((val['Dxy',]['index.corrected']+1)/2)
#pdf('calibrationPlot.pdf', width = 8, height = 8)
cal <- calibrate(f, B=100)
par(mfrow=c(1,1))
plot(cal, xlim=c(0,1), ylim=c(0,1))
#title(paste(as.character(f$sformula)[2], ' ~ ', as.character(f$sformula)[3], sep=''))
#dev.off()
