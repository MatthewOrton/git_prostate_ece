# clear environment and close figures
rm(list=ls())
if (dev.cur()!=1) {dev.off()}


if (startsWith(getwd(),'/Users/morton')){
  setwd("/Users/morton/Documents/GitHub/prostate_ece_semantic/R")
  currentEnv <- 'laptop'
} else {
  currentEnv <- 'xnatdev'
}

library(stringr)
library(survival)
library(survminer)
library(glmnet)
library(gglasso)
library(ggplot2)
library(stringr)
library(dplyr)
library(caret)
library(mltools)
library(data.table)
library(Hmisc)
library(pROC)
# library(randomForestSRC)
# library(tidyverse)

# read data
dataFileName <- '/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/ECE_Prostate_Semantic/ECE_Semantic_Data/discovery.csv'
df <- read.csv(dataFileName)

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
#df$GleasonBinary <- factor(df$GleasonBinary)
# make target numeric
df$ECE_Pathology <- as.numeric(df$ECE_Pathology)

targetName <- 'ECE_Pathology'

set.seed(123456)

###################
# training function
#
# Fit LR+LASSO to training data with features.
# For the features with non-zero coeff, append all feature interactions to make new data frame.
# Fit LR+LASSO to this data set that includes interactions.
# Test on test data 
fitOnce <- function(Xtrain, yTrain, Xtest, yTest, lambdaStyle='lambda.min', makeInteractions=FALSE, maxFeatures=NULL){


  if (makeInteractions){
    selectedFeatures <- colnames(Xtrain)
    
    
    # scale data and apply scaling to test data
    #Xtrain <- scale(Xtrain)
    #Xtest <- scale(Xtest, attr(Xtrain, "scaled:center"), attr(Xtrain, "scaled:scale"))
    
    # select features and make interacting features for the selected features
    Xtrain <- select(as.data.frame(Xtrain), selectedFeatures)
    Xtest <- select(as.data.frame(Xtest), selectedFeatures)
    for (aa in seq(length(selectedFeatures)-1)){
      for (bb in seq(length(selectedFeatures)-aa)){
        newFeatureName  <- paste(selectedFeatures[aa], 'x', selectedFeatures[aa+bb])
        Xtrain[newFeatureName] <- Xtrain[selectedFeatures[aa]]*Xtrain[selectedFeatures[aa+bb]]
        Xtest[newFeatureName] <- Xtest[selectedFeatures[aa]]*Xtest[selectedFeatures[aa+bb]]
      }
    }
  }


  # scale data and apply scaling to test data
  Xtrain <- scale(Xtrain)
  Xtest <- scale(Xtest, attr(Xtrain, "scaled:center"), attr(Xtrain, "scaled:scale"))
  

  # fit with interacting features to training data
  # fitInt <- cv.glmnet(Xtrain, yTrain, family = "binomial", alpha=1, type.measure="deviance", keep=TRUE, parallel=TRUE)
  group <- c(1, 2, 2, 2, 3, 3, 3, 3, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
  fitInt <- cv.gglasso(x=Xtrain, y=2*yTrain-1, group=group, loss='logit') #, alpha=1, type.measure="deviance", keep=TRUE, parallel=TRUE)
  # get selected features
  coefs <- as.matrix(coef(fitInt, s=lambdaStyle))

  # re-do fit with lambda constrained so we get a model with no more than maxFeatures
  if (!is.null(maxFeatures)){
    if (sum(coefs!=0) > maxFeatures){
      lambda <- fitInt$lambda[fitInt$nzero <= maxFeatures]
      #fitInt <- cv.glmnet(Xtrain, yTrain, family = "binomial", alpha=1, type.measure="deviance", keep=TRUE, parallel=TRUE, lambda=lambda)
      fitInt <- cv.gglasso(x=Xtrain, y=2*yTrain-1, group=group, loss='logit', lambda=TRUE) #, alpha=1, type.measure="deviance", keep=TRUE, parallel=TRUE)
      # get selected features
      coefs <- as.matrix(coef(fitInt, s=lambdaStyle))
    }
  }
  
  
  scoreTrain <- as.vector(predict(fitInt, newx = Xtrain, s=lambdaStyle, type='link'))
  scoreTest <- as.vector(predict(fitInt, newx = Xtest, s=lambdaStyle, type='link'))

  # this bit allows us to compute a net AUC accounting for 9 cases with RetroprostaticAngleOblit=1 that might be excluded
  # scoreTrain <- rbind(as.matrix(scoreTrain), matrix(max(scoreTrain), 9))
  # yTrain <- rbind(as.matrix(yTrain), matrix(1, 9))
  # scoreTest <- rbind(as.matrix(scoreTest), matrix(max(scoreTest), 9))
  # yTest <- rbind(as.matrix(yTest), matrix(1, 9))
  
  aucTrain <- auc(roc(as.numeric(yTrain), as.vector(scoreTrain), levels=c(0,1), direction='<'))
  aucTest <- auc(roc(as.numeric(yTest), as.vector(scoreTest), levels=c(0,1), direction='<'))
  
  return(list('aucTrain'=aucTrain, 'aucTest'=aucTest, 'beta'=coefs))
}

completeFit <- function(df, makeInteractions=FALSE, lambdaStyle='lambda.min', nBoot=100, maxFeatures=NULL, removeRetroprostaticAngleOblit=FALSE, seed=NULL){

  set.seed(seed)
    
  cat('\n')
  cat('___________________________\n')
  
  ########################
  # All RetroprostaticAngleOblit=1 cases have ECE_Pathology=1 as well
  # This patient selection assumes that this is always true and can be used as a first-step predictor
  # We then need a multivariate predictor for RetroprostaticAngleOblit=0 cases
  #######################
  
  if (removeRetroprostaticAngleOblit){
    df <- df[df$RetroprostaticAngleOblit==0,]
    df <- select(df, -c('RetroprostaticAngleOblit'))
    cat('Remove RetroprostaticAngleOblit =', removeRetroprostaticAngleOblit, '\n')
  }
  
  X <- as.matrix(select(df, -'ECE_Pathology'))
  y <- as.matrix(select(df, 'ECE_Pathology'))

  cat('makeInteractions =', makeInteractions, '\n')
  cat('lambdaStyle =', lambdaStyle, '\n')
  cat('Max feature count =', maxFeatures, '\n')
  cat('Bootstrap samples =', nBoot, '\n')
  
  result <- fitOnce(X, y, X, y, makeInteractions=makeInteractions, lambdaStyle=lambdaStyle, maxFeatures=maxFeatures)
  aucResub <- result$aucTrain[1]
  cat('\nAUROC (resub) =', aucResub, '\n')
  cat('Feature count =', sum(result$beta !=0) - 1, '\n\n')
  for (i in which(result$beta !=0)){
    if (rownames(result$beta)[i] != '(Intercept)') {
      cat(rownames(result$beta)[i], round(result$beta[i],3), '\n', sep=' ')
    }
  }
  
  
  # Bootstrap the performance
  aucOptimism <- rep(0, nBoot)
  # dummy run to get the size info for featureSelected
  result <- fitOnce(X, y, X, y, makeInteractions=makeInteractions, lambdaStyle=lambdaStyle, maxFeatures=maxFeatures)
  featureSelected <- matrix(0, length(result$beta)-1 , nBoot)

  for (i in seq(nBoot)){
    
    iBoot <- createResample(y,  1)$Resample1

    Xtrain <- X[iBoot,]
    yTrain <- y[iBoot,]

    result <- fitOnce(Xtrain, yTrain, X, y, makeInteractions=makeInteractions, lambdaStyle=lambdaStyle, maxFeatures=maxFeatures)
    aucOptimism[i] <- result$aucTrain[1] - result$aucTest[1]
    featureSelected[,i] <- as.numeric(result$beta[2:length((result$beta))] != 0)
  }
  cat('\nAUROC optimism = ', mean(aucOptimism), '\n')
  cat('AUROC boot = ', aucResub - mean(aucOptimism), '\n')
  print(table(colSums(featureSelected)))
  cat('\n')
  
  rownames(featureSelected) <- rownames(result$beta)[2:length((result$beta))]
  featureFreq <- t(rowSums(featureSelected))
  #print(featureFreq[,featureFreq > 0])
  
  
  featureCombSig <- as.vector(t(as.vector(2^(seq(nrow(featureSelected))-1))) %*% featureSelected)
  featureCombFreq <- as.data.frame(table(featureCombSig))
  
  for (i in order(-featureCombFreq$Freq)[1:5]){
    idx <- which(featureCombSig==featureCombFreq$featureCombSig[i])[1] 
    cat(featureCombFreq$Freq[i], ' : ', paste(rownames(featureSelected)[featureSelected[,idx]==1], collapse=' + '), '\n\n', sep='')
  }
  
  
}


nBoot <- 10
seed <- 123456
completeFit(df, makeInteractions=FALSE, lambdaStyle='lambda.1se', nBoot=nBoot, seed=seed)

