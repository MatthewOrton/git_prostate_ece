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
  df[feat] <- factor(as.numeric(df[feat] == "YES"))
}
df$GleasonBinary <- factor(df$GleasonBinary)
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
  fitInt <- cv.glmnet(Xtrain, yTrain, family = "binomial", alpha=1, type.measure="deviance", keep=TRUE, parallel=TRUE)
  # get selected features
  coefs <- as.matrix(coef(fitInt, s=lambdaStyle))

  # re-do fit with lambda constrained so we get a model with no more than maxFeatures
  if (!is.null(maxFeatures)){
    if (sum(coefs!=0) > maxFeatures){
      lambda <- fitInt$lambda[fitInt$nzero <= maxFeatures]
      fitInt <- cv.glmnet(Xtrain, yTrain, family = "binomial", alpha=1, type.measure="deviance", keep=TRUE, parallel=TRUE, lambda=lambda)
      # get selected features
      coefs <- as.matrix(coef(fitInt, s=lambdaStyle))
    }
  }
  
  
  scoreTrain <- as.vector(predict(fitInt, newx = Xtrain, s=lambdaStyle))
  scoreTest <- as.vector(predict(fitInt, newx = Xtest, s=lambdaStyle))

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


# get smoothed estimate of risk curves 
pp <- plsmo(jitter(df$CapsularContactLength), df$ECE_Pathology, method='lowess', xlab='CapsularContactLength', ylab='logit Pr(ECE=1)', group=df$GleasonBinary, datadensity=TRUE, ylim=c(-6,6), xlim=c(0,40), fun=function(x){log((0.01+x)/(1.01-x))})

# extract values of curves into a data frame for plotting with ggplot
dfPlsmo <- data.frame(x=c(pp[['0']]$x, pp[['1']]$x))
dfPlsmo$y <- c(pp[['0']]$y, pp[['1']]$y)
dfPlsmo$y <- exp(dfPlsmo$y)/(1+exp(dfPlsmo$y))
dfPlsmo$GleasonBinary <- factor(c(rep(0,length(pp[['0']]$y)), rep(1,length(pp[['1']]$y))))

# settings used to determine data limits when running lrm()
ddist <- datadist(df, q.effect=c(0,1), q.display=c(0,1))
options(datadist='ddist')

# fit logistic regression models and overlay the plsmo empirical curves
onePlot <- function(df, formulaStr, var1str, var2str, count){

  if (is.null(var2str)){
    pp <- plsmo(jitter(df[,var1str]), df$ECE_Pathology, method='lowess', xlab='CapsularContactLength', ylab='logit Pr(ECE=1)', datadensity=TRUE, ylim=c(-6,6), xlim=c(0,40), fun=function(x){log((0.01+x)/(1.01-x))})
    
    # extract values of curves into a data frame for plotting with ggplot
    dfPlsmo <- data.frame(x=c(pp[['0']]$x, pp[['1']]$x))
    dfPlsmo$y <- c(pp[['0']]$y, pp[['1']]$y)
    dfPlsmo$y <- exp(dfPlsmo$y)/(1+exp(dfPlsmo$y))

    
    f1 <- lrm(as.formula(formulaStr), data=df)
    eval(parse(text=paste0('p1 <- Predict(f1, ', var1str, ', fun=plogis)')))
    p1 <- as.data.frame(p1)
    gg <- ggplot()
    gg <- gg + geom_line(data=p1, aes_string(x=var1str, y='yhat'))
    gg <- gg + geom_ribbon(data=p1, aes_string(x=var1str, ymin='lower', ymax='upper'), alpha=0.15)
    gg <- gg + geom_line(data=dfPlsmo, aes_string(x='x', y='y'), linetype='dashed')
    gg <- gg + scale_y_continuous(breaks=c(0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999))
    gg <- gg + coord_trans(y = 'logit', ylim=c(0.001,0.999))
    gg <- gg + labs(title=formulaStr, y = 'Pr(ECE_Pathology = 1)')
    
    fileName <- paste0(var1str, '_', count, '.pdf')
  } else {
    pp <- plsmo(jitter(df[,var1str]), df$ECE_Pathology, method='lowess', xlab='CapsularContactLength', ylab='logit Pr(ECE=1)', group=df[,var2str], datadensity=TRUE, ylim=c(-6,6), xlim=c(0,40), fun=function(x){log((0.01+x)/(1.01-x))})
  
    # extract values of curves into a data frame for plotting with ggplot
    dfPlsmo <- data.frame(x=c(pp[['0']]$x, pp[['1']]$x))
    dfPlsmo$y <- c(pp[['0']]$y, pp[['1']]$y)
    dfPlsmo$y <- exp(dfPlsmo$y)/(1+exp(dfPlsmo$y))
    dfPlsmo[,var2str] <- factor(c(rep(0,length(pp[['0']]$y)), rep(1,length(pp[['1']]$y))))
    
    
    f1 <- lrm(as.formula(formulaStr), data=df)
    eval(parse(text=paste0('p1 <- Predict(f1, ', var1str, ', ', var2str, ', fun=plogis)')))
    p1 <- as.data.frame(p1)
    gg <- ggplot()
    gg <- gg + geom_line(data=p1, aes_string(x=var1str, y='yhat', group=var2str, color=var2str))
    gg <- gg + geom_ribbon(data=p1, aes_string(x=var1str, ymin='lower', ymax='upper', group=var2str), alpha=0.15)
    gg <- gg + geom_line(data=dfPlsmo, aes_string(x='x', y='y', group=var2str, color=var2str), linetype='dashed')
    gg <- gg + scale_y_continuous(breaks=c(0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999))
    gg <- gg + coord_trans(y = 'logit', ylim=c(0.001,0.999))
    gg <- gg + labs(title=formulaStr, y = 'Pr(ECE_Pathology = 1)')
    
    fileName <- paste0(var1str, '_', var2str, '_', count, '.pdf')
  }
  ggsave(fileName, width = 15, height = 12, units = "cm")
}

onePlot(df, 'ECE_Pathology ~ CapsularContactLength', 'CapsularContactLength', NULL, 1)

onePlot(df, 'ECE_Pathology ~ CapsularContactLength + GleasonBinary', 'CapsularContactLength', 'GleasonBinary', 1)
onePlot(df, 'ECE_Pathology ~ CapsularContactLength*GleasonBinary', 'CapsularContactLength', 'GleasonBinary', 2)
onePlot(df, 'ECE_Pathology ~ rcs(CapsularContactLength,3) + GleasonBinary', 'CapsularContactLength', 'GleasonBinary', 3)
onePlot(df, 'ECE_Pathology ~ rcs(CapsularContactLength,3)*GleasonBinary', 'CapsularContactLength', 'GleasonBinary', 4)

onePlot(df, 'ECE_Pathology ~ CapsularContactLength + IrregularContour', 'CapsularContactLength', 'IrregularContour', 1)
onePlot(df, 'ECE_Pathology ~ CapsularContactLength + CapsularDisruption', 'CapsularContactLength', 'CapsularDisruption', 1)

poop
# main bootstrap runs for different fitting methods

nBoot <- 10
seed <- 123456
completeFit(df, makeInteractions=FALSE, lambdaStyle='lambda.min', nBoot=nBoot, seed=seed)
completeFit(df, makeInteractions=FALSE, lambdaStyle='lambda.1se', nBoot=nBoot, seed=seed)
completeFit(df, makeInteractions=FALSE, lambdaStyle='lambda.1se', nBoot=nBoot, maxFeatures=5, seed=seed)


poop



# interactions
#completeFit(df, makeInteractions=TRUE, lambdaStyle='lambda.1se', maxFeatures=6)
#completeFit(df, makeInteractions=FALSE, lambdaStyle='lambda.1se', maxFeatures=6)


cf <- 'CapsularContactLength'
par(mfrow=c(2,3))
plotAlone <- plsmo(df[,cf], df[,targetName], method='lowess', xlab=cf, ylab='logit Pr(ECE=1)', datadensity=TRUE, ylim=c(-6,6), fun=function(x){log((0.01+x)/(1.01-x))}, col=2)[['1']]
title('No interactions')
df$single <- df[,cf]
ddist <- datadist(df$ECE_Pathology, df$single)
options(datadist='ddist')
f <- lrm(ECE_Pathology ~ single, df, x=TRUE, y=TRUE)
lines(df$single, f$linear.predictors, col=4)



for (bf in c('IrregularContour', 'GleasonBinary', 'CapsularDisruption')){
  plsmo(df[,cf], df[,targetName], method='lowess', xlab=cf, ylab='logit Pr(ECE=1)', group=df[,bf], datadensity=TRUE, ylim=c(-6,6), fun=function(x){log((0.01+x)/(1.01-x))}, colors=c('r','g'))
  #lines(plotAlone[['x']], plotAlone[['y']], col=2)
  title(bf)
  df$cat <- df[,bf]  
  ddist <- datadist(df$ECE_Pathology, df$single, df$cat)
  options(datadist='ddist')
  f <- lrm(ECE_Pathology ~ single + cat, df, x=TRUE, y=TRUE)
  xx <- c(0, 35)
  yy <- c(predict(f, data.frame('single'=xx[1], 'cat'=0)),
          predict(f, data.frame('single'=xx[2], 'cat'=0)))
  lines(xx, yy, col=4, lty='dashed')
  
  xx <- c(0, 35)
  yy <- c(predict(f, data.frame('single'=xx[1], 'cat'=1)),
          predict(f, data.frame('single'=xx[2], 'cat'=1)))
  lines(xx, yy, col=3, lty='dashed')
}

rowIdx <- df$RetroprostaticAngleOblit==0
plsmo(df[rowIdx,cf], df[rowIdx,targetName], method='lowess', xlab=cf, ylab='logit Pr(ECE=1)', datadensity=TRUE, ylim=c(-6,6), fun=function(x){log((0.01+x)/(1.01-x))})
#lines(plotAlone[['x']], plotAlone[['y']], col=2)
title('RetroprostaticAngleOblit==0')


poop


ddist <- datadist(df)
options(datadist='ddist')

par(mfrow=c(1,2))


f <- lrm(ECE_Pathology ~ CapsularContactLength + GleasonBinary + CapsularDisruption + IrregularContour + RetroprostaticAngleOblit, data=df)
f <- update(f, x=T, y=T)
val <- validate(f, B=100)
cat((val['Dxy',]['index.corrected']+1)/2)
#pdf('calibrationPlot.pdf', width = 8, height = 8)
cal <- calibrate(f, B=100)
plot(cal, xlim=c(0,1), ylim=c(0,1))
#title(paste(as.character(f$sformula)[2], ' ~ ', as.character(f$sformula)[3], sep=''))
#dev.off()

f <- lrm(ECE_Pathology ~ rcs(CapsularContactLength,3)*(GleasonBinary + CapsularDisruption + IrregularContour), data=df)
f <- update(f, x=T, y=T)
val <- validate(f, B=100)
cat((val['Dxy',]['index.corrected']+1)/2)
#pdf('calibrationPlot.pdf', width = 8, height = 8)
cal <- calibrate(f, B=100)
plot(cal, xlim=c(0,1), ylim=c(0,1))
#title(paste(as.character(f$sformula)[2], ' ~ ', as.character(f$sformula)[3], sep=''))
#dev.off()
