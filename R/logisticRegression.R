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
df$GleasonBinary <- df$GleasonBinary
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
fitOnce <- function(Xtrain, yTrain, Xtest, yTest, lambdaStyle='lambda.min', makeInteractions=FALSE, makeAllInteractions=FALSE){

  if (makeAllInteractions){
    makeInteractions = TRUE
  }
  
  # scale data and apply scaling to test data
  Xtrain <- scale(Xtrain)
  Xtest <- scale(Xtest, attr(Xtrain, "scaled:center"), attr(Xtrain, "scaled:scale"))

  if (!makeAllInteractions){  
    # fit non-interacting features to training data
    fitRaw <- cv.glmnet(Xtrain, yTrain, family = "binomial", alpha=1, type.measure="deviance", keep=TRUE, parallel=TRUE)
    # get selected features
    coefs <- as.matrix(coef(fitRaw, s=lambdaStyle))
    scoreTest <- predict(fitRaw, newx = Xtest, s=lambdaStyle)
  }
  
  if (makeInteractions){
    if (makeAllInteractions){
      selectedFeatures <- colnames(Xtrain)
    } else {
      selectedFeatures <- rownames(coefs)[coefs != 0 & rownames(coefs)!='(Intercept)']
    }
    
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
    # scale data and apply scaling to test data
    Xtrain <- scale(Xtrain)
    Xtest <- scale(Xtest, attr(Xtrain, "scaled:center"), attr(Xtrain, "scaled:scale"))
    
    # fit with interacting features to training data
    fitInt <- cv.glmnet(Xtrain, yTrain, family = "binomial", alpha=1, type.measure="deviance", keep=TRUE, parallel=TRUE)
    # get selected features
    coefs <- as.matrix(coef(fitInt, s=lambdaStyle))

    scoreTest <- predict(fitInt, newx = Xtest, s=lambdaStyle)
  }
  
  return(list('auc'=auc(roc(as.numeric(yTest), as.vector(scoreTest), levels=c(0,1), direction='<')),
              'featureCount'=sum(coefs!=0)))
}

completeFit <- function(df, makeInteractions=FALSE, makeAllInteractions=FALSE, lambdaStyle='lambda.min', nRepeats=10){
  
  X <- as.matrix(select(df, -'ECE_Pathology'))
  y <- as.matrix(select(df, 'ECE_Pathology'))
  
  cat('\n')
  cat('___________________________\n')
  cat('makeAllInteractions =', makeAllInteractions, '\n')
  cat('makeInteractions =', makeInteractions, '\n')
  cat('lambdaStyle =', lambdaStyle, '\n')
  
  result <- fitOnce(X, y, X, y, makeInteractions=makeInteractions, lambdaStyle=lambdaStyle, makeAllInteractions=makeAllInteractions)
  
  cat('AUROC (resub) =', result$auc[1], '\n')
  cat('Feature count =', result$featureCount, '\n')
  
  
  # Cross-validate the performance
  nFolds <- 5
  aucArr <- rep(0, nFolds*nRepeats)
  featureCount <- rep(0, nFolds*nRepeats)
  counter <- 0
  for (j in seq(nRepeats)){
    
    flds <- createFolds(y, k = nFolds, list = TRUE, returnTrain = FALSE)
    nonZeroCoeff <- list()
    
    for (i in seq(nFolds)){
      
      Xtrain <- X[-flds[[i]],]
      yTrain <- y[-flds[[i]],]
      Xtest  <- X[ flds[[i]],]
      yTest  <- y[ flds[[i]],]
      
      counter <- counter + 1
      result <- fitOnce(Xtrain, yTrain, Xtest, yTest, makeInteractions=makeInteractions, lambdaStyle=lambdaStyle, makeAllInteractions=makeAllInteractions)
      aucArr[counter] <- result$auc[1]
      featureCount[counter] <- result$featureCount
    }
  }
  cat('AUROC (CV) = ', mean(aucArr), ' +/- ' , sqrt(var(aucArr)), '\n')
  print(table(featureCount))
  cat('\n')
}


completeFit(df, makeInteractions=FALSE, makeAllInteractions=FALSE, lambdaStyle='lambda.1se')
# completeFit(df, makeInteractions=TRUE, makeAllInteractions=FALSE, lambdaStyle='lambda.1se')
# completeFit(df, makeInteractions=TRUE, makeAllInteractions=TRUE, lambdaStyle='lambda.1se')

#completeFit(df, makeInteractions=FALSE, makeAllInteractions=FALSE, lambdaStyle='lambda.min')
#completeFit(df, makeInteractions=TRUE, makeAllInteractions=FALSE, lambdaStyle='lambda.min', nRepeats=1)
#completeFit(df, makeInteractions=TRUE, makeAllInteractions=TRUE, lambdaStyle='lambda.min')