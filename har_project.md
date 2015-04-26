# Practical Machine Learning Course Project
Rashan Jibowu  
April 25, 2015  

### Overview

This report details my approach toward predicting the manner in which a person performed an exercise. Using data from the [Human Activity Recognition project](http://groupware.les.inf.puc-rio.br/har), below, I describe the process of building a predictive model that determines whether a person is doing an exercise correctly. 

### Set up


```r
library(ggplot2)
library(caret)
```

```
## Loading required package: lattice
```

```r
library(foreach)
library(data.table)
library(parallel)
library(doParallel)
```

```
## Loading required package: iterators
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
# urls
data.test.url <- c("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
data.train.url <- c("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
```

#### Get and load data


```r
# set to true if you always want to download the data regardless of whether it was already downloaded
manual.download <- FALSE

# download data if not already downloaded
if (!file.exists("./data/training.csv") || !file.exists("./data/testing.csv") || manual.download) {
    if (!file.exists("./data")) {
        dir.create("./data")
    }
    download.file(data.train.url, destfile = "./data/training.csv", method = "curl")
    download.file(data.test.url, destfile = "./data/testing.csv", method = "curl")
}

# load the data
data.test <- read.csv("./data/testing.csv", na.strings = c("NA", "#DIV/0!", ""), colClasses = c("character"))

data.train <- read.csv("./data/training.csv", na.strings = c("NA", "#DIV/0!", ""), colClasses = c("character"))
```

#### Clean up data  

Since the data file came in a bit messy, let's do some clean up. I loaded the data in and treated all variables as strings. When coerced to the correct data types, NAs will surface appropriately


```r
# coerce data to correct data types
classes <- c("int", "character", "int", "int", "character", "character", "int", rep("num", times = 152), "character")

convertToClasses <- function(df, classes) {
    for (i in 1:ncol(df)) {

        if (classes[i] == "num") {
            df[,i] <- as.numeric(df[,i])
        }

        if (classes[i] == "int") {
            df[,i] <- as.integer(df[,i])
        }

        if (classes[i] == "character") {
            df[,i] <- as.character(df[,i])
        }
    }
    
    # return the cleaned data frame
    df
}

data.train <- convertToClasses(data.train, classes)
data.test <- convertToClasses(data.test, classes)
```

#### Remove irrelevant factors

Some columns represent data that ought not to be considered for machine learning. They include timestamps (`raw_timestamp_part_1`, `raw_timestamp_part_2`, and `cvtd_timestamp`), `user_name`, the row number (`X`), and window data (`num_window` and `new_window`). If this metadata were included in a final model, it would significantly reduce the overall robustness of the model to deal with new cases.


```r
# remove irrelevant factors
irrelevant <- grep("*timestamp*|^user_name|^X$|*window*", colnames(data.train))
data.train <- data.train[,-irrelevant]
```

### Selecting Features

Many variables have (lots of) missing values. The absence of information in these variables make them unhelpful. As a result, let's remove those. 

Of the remaining variables, let's retain those variables that are related to belt, arm, forearm and dumbbell measurements.


```r
unhelpful <- sapply(data.train, function (x) any(is.na(x) | x == ""))
relevant <- grepl("belt|(^fore)*arm|dumbbell", colnames(data.train))
predictors <- colnames(data.train[,(!unhelpful & relevant)])

data.train <- data.train[,c(predictors, "classe")]
```

Before training the model, let's ensure that our target variable is a categorical variable.


```r
data.train$classe <- as.factor(data.train$classe)
```

### Cross validation

Below, I cross-validate the data by using 70% of the training data to train the model and the remainder to validate the quality of the model. If performant, I'll use this model to make predictions on the testing data.


```r
inTrain <- createDataPartition(data.train$classe, p = 0.7, list = FALSE)
data.train.subtrain <- data.train[inTrain,]
data.train.test <- data.train[-inTrain,]
```

### Preprocess the data

Since there are so many potential predictor variables (n = 52) and since the vast majority of them are numeric, the dataset would benefit from preprocessing the data so that the data are centered and scaled, enabling better predictions.


```r
# center and scale data - exclude the target (col 53)
preProc <- preProcess(data.train.subtrain[,c(1:52)], center = TRUE, scale = TRUE)
preProc
```

```
## 
## Call:
## preProcess.default(x = data.train.subtrain[, c(1:52)], center =
##  TRUE, scale = TRUE)
## 
## Created from 13737 samples and 52 variables
## Pre-processing: centered, scaled
```

### Attempt to parallelize

Since the data set is large, the training process may take awhile to complete. To potentially speed up the process, let's attempt to use multiple computing cores to process data in parallel.


```r
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
```

### Train model

Let's train the model on the subtrain data.


```r
set.seed(12345)
trControl <- trainControl(classProbs = TRUE, 
                          savePredictions = TRUE, 
                          allowParallel=TRUE)

trainingModel <- train(classe ~ ., data = data.train.subtrain, method = "rf")
```

### Measuring Quality of Model

#### Evaluating on subtrain data


```r
data.train.subtrain.predictions <- predict(trainingModel, data.train.subtrain)
confusionMatrix(data.train.subtrain.predictions, data.train.subtrain$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3906    0    0    0    0
##          B    0 2658    0    0    0
##          C    0    0 2396    0    0
##          D    0    0    0 2252    0
##          E    0    0    0    0 2525
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

Based on the analysis above, the "in-sample" error rate is 0%. Despite the high accuracy, there will likely be some error for the cross-validating testing dataset and the external testing dataset to be a little greater. Let's set expectations to be between 1% and 2%.

### Evaluation on cross-validating testing data


```r
data.train.test.predictions <- predict(trainingModel, data.train.test)
confusionMatrix(data.train.test.predictions, data.train.test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1667    5    0    0    0
##          B    5 1129    6    0    0
##          C    1    5 1016    5    2
##          D    0    0    4  959    3
##          E    1    0    0    0 1077
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9937          
##                  95% CI : (0.9913, 0.9956)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.992           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9958   0.9912   0.9903   0.9948   0.9954
## Specificity            0.9988   0.9977   0.9973   0.9986   0.9998
## Pos Pred Value         0.9970   0.9904   0.9874   0.9928   0.9991
## Neg Pred Value         0.9983   0.9979   0.9979   0.9990   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2833   0.1918   0.1726   0.1630   0.1830
## Detection Prevalence   0.2841   0.1937   0.1749   0.1641   0.1832
## Balanced Accuracy      0.9973   0.9945   0.9938   0.9967   0.9976
```

#### Estimating out-of-sample error

The above analysis shows an _intermediate_ "out-of-sample" error of just under 1% for the cross-validating test dataset. Based on this, I would expect the true "out-of-sample" error on the external testing data set to be around 1.5%-2.0%.

### Final Model


```r
trainingModel$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.71%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3901    4    1    0    0 0.001280082
## B   14 2635    8    1    0 0.008653123
## C    0   11 2374   11    0 0.009181970
## D    0    2   29 2219    2 0.014653641
## E    0    1    4    9 2511 0.005544554
```

Below are the variables that were most important to this model.


```r
varImp(trainingModel)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                      Overall
## roll_belt             100.00
## pitch_forearm          57.53
## yaw_belt               52.38
## magnet_dumbbell_z      44.37
## pitch_belt             42.16
## roll_forearm           41.61
## magnet_dumbbell_y      40.59
## accel_dumbbell_y       20.79
## accel_forearm_x        17.73
## magnet_dumbbell_x      17.28
## roll_dumbbell          17.13
## magnet_forearm_z       14.92
## magnet_belt_z          14.84
## accel_dumbbell_z       13.86
## accel_belt_z           13.59
## total_accel_dumbbell   12.73
## magnet_belt_y          12.35
## gyros_belt_z           11.38
## yaw_arm                11.11
## magnet_belt_x          10.15
```

Save the final model for evaluation against the test data.


```r
save(trainingModel, file="trainingModel.RData")
```
