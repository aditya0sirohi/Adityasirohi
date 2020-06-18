## R-script  running / training each model and using the best one to predict "classe" with the test dataset
## For the rpubs report go here: https://rpubs.com/bzhang93/coursera-machine-learning-project

# libraries
library(lattice)
library(ggplot2)
library(caret)
library(kernlab)
library(rattle)
library(corrplot)
set.seed(1234)

# load the data
traincsv <- read.csv("./data/pml-training.csv")
testcsv <- read.csv("./data/pml-testing.csv")

# cleaning the data
traincsv <- traincsv[,colMeans(is.na(traincsv)) < .9] #removing mostly na columns
traincsv <- traincsv[,-c(1:7)] #removing metadata which is irrelevant to the outcome
# removing zero variance variables
nvz <- nearZeroVar(traincsv)
traincsv <- traincsv[,-nvz]
dim(traincsv)

# create validation and training set from training dataset
inTrain <- createDataPartition(y=traincsv$classe, p=0.7, list=F)
train <- traincsv[inTrain,]
valid <- traincsv[-inTrain,]

# Run Models
print("Training models, please wait...")
control <- trainControl(method="cv", number=3, verboseIter=F) #set up fixed training parameters

## decision tree
mod_trees <- train(classe~., data=train, method="rpart", trControl = control, tuneLength = 5)
pred_trees <- predict(mod_trees, valid)
cmtrees <- confusionMatrix(pred_trees, factor(valid$classe))
fancyRpartPlot(mod_trees$finalModel)

## random forest
mod_rf <- train(classe~., data=train, method="rf", trControl = control, tuneLength = 5)
pred_rf <- predict(mod_rf, valid)
cmrf <- confusionMatrix(pred_rf, factor(valid$classe))

## GBM
mod_gbm <- train(classe~., data=train, method="gbm", trControl = control, tuneLength = 5, verbose = F)
pred_gbm <- predict(mod_gbm, valid)
cmgbm <- confusionMatrix(pred_gbm, factor(valid$classe))

# SVM
mod_svm <- train(classe~., data=train, method="svmLinear", trControl = control, tuneLength = 5, verbose = F)
pred_svm <- predict(mod_svm, valid)
cmsvm <- confusionMatrix(pred_svm, factor(valid$classe))

# informing user of each model's evaluation
print("Accuracy and out of sample error rate for each model...")
models <- c("Tree", "RF", "GBM", "SVM")
accuracy <- round(c( cmtrees$overall[1], cmrf$overall[1], cmgbm$overall[1], cmsvm$overall[1]),3) #accuracy
oos_error <- 1 - accuracy #out of sample error
data.frame(accuracy = accuracy, oos_error = oos_error, row.names = models)

# using best model
print("Using best model to predict class outcome for each test set observation...")
pred <- predict(mod_rf, testcsv)
print(pred)

print("Done.")

