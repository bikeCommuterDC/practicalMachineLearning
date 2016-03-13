# Machine Learning Project
March 14, 2016  

#Setup
The below steps setup the R environment but code is not shown.  
- Load R packages and set the working directory  
- Turn on parallel processing. This enables R to use all 4 CPU cores on my laptop when running the decision trees below, reducing processing time.  


#Data Import and Prep
Import both the training and test data.

```r
#import training and testing datasets
training_raw <- read.csv("pml-training.csv", stringsAsFactors = FALSE, row.names = 1)
testing_raw <- read.csv("pml-testing.csv", stringsAsFactors = FALSE, row.names = 1)
```


#Pre-Processing
Pre-processing includes the following steps:  
- separate the training data into two data sets: (1) contains 80 percent of the training data and will be used to train the model and (2) contains 20 percent of the training data and will be used to cross validate the model prior to running the model on the testing data.  
- Remove variables with near-zero variance.  
- Remove variables that contain at least 95 percent "NA".

Variables with low variance or mostly Na's are removed for two reasons:  (1) given that they are mostly the same value implies they will not have any predictive value and (2) when the data are partitioned during random forests it is likely the variables will contain the same value in certain subgroups (or be entirely NAs). If they are the same value they do not add predictive value. 

```r
#Separate training data into training and cross-validation datasets
inTrain <- createDataPartition(y = training_raw$classe, p = 0.8, list = FALSE)

training.training <- training_raw[inTrain,]
training.xval <- training_raw[-inTrain,]

#remove variables with near-zero variance from the training data
nsv <- nearZeroVar(training.training)
training.clean.1 <- training.training[,-nsv]

#remove columns with 95% or more NAs
training.clean.2 <- training.clean.1[, colSums(is.na(training.clean.1)) < nrow(training.clean.1)*.95]

#remove the remaining first five columns which provide data such as names and timestamps which are not relevant for model creation
training.final <- training.clean.2[,-c(1:5)]

#remove transitional datasets to clear memory space
remove(training.clean.2)
remove(training.clean.1)
```

#Modeling
Since there are so many variables in the data set that could be included in the model, we skip plotting covariates as a means for variable selection. Instead we simply run the model with all variables and allow the algorithm to determine the important predictors.

The model was built using random forests. Random forest was chose for the following reasons.  
1. RF does not require transformation of data prior to modeling.  
2. We have a large collection of variables (even after removing low variance and NAs) and random forest will effectively determine the important variables.  
3. The data is relatively small so run time will not be too lengthy for random forests.  
4. Random forests are known to produce highly accurate models in many applications.  

```r
#Due to long processing times, the training code is commented out when creating knitr file. The model was saved as an RDS file and called in the below steps. 
#modelFit <- train(classe ~ . , data= training.final, method = "rf")
#saveRDS(modelFit, "FinalModel.RDS")

#call previously saved random forest model
modelFit = readRDS("FinalModel.RDS")

#predict classification for cross validation data
xval.predictions <- predict(modelFit, newdata = training.xval)

#produce confusion matrix
confusionMatrix(xval.predictions, training.xval$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  757    0    0    0
##          C    0    2  683    1    0
##          D    0    0    1  642    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.999           
##                  95% CI : (0.9974, 0.9997)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9987          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9974   0.9985   0.9984   1.0000
## Specificity            1.0000   1.0000   0.9991   0.9997   1.0000
## Pos Pred Value         1.0000   1.0000   0.9956   0.9984   1.0000
## Neg Pred Value         1.0000   0.9994   0.9997   0.9997   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1930   0.1741   0.1637   0.1838
## Detection Prevalence   0.2845   0.1930   0.1749   0.1639   0.1838
## Balanced Accuracy      1.0000   0.9987   0.9988   0.9991   1.0000
```
Results on the training data show that the model is highly accurate, 99.4%. The in-sample error rate is 0.6%. Next we use the cross-validation data to estimate the out-of-sample error. 

#Cross Validation
The following code predicts the classification variable on the cross validation data set and produces a confusion matrix. 


```r
#predict classification for cross validation data
xval.predictions <- predict(modelFit, newdata = training.xval)

#produce confusion matrix
confusionMatrix(xval.predictions, training.xval$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    0  757    0    0    0
##          C    0    2  683    1    0
##          D    0    0    1  642    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.999           
##                  95% CI : (0.9974, 0.9997)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9987          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9974   0.9985   0.9984   1.0000
## Specificity            1.0000   1.0000   0.9991   0.9997   1.0000
## Pos Pred Value         1.0000   1.0000   0.9956   0.9984   1.0000
## Neg Pred Value         1.0000   0.9994   0.9997   0.9997   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1930   0.1741   0.1637   0.1838
## Detection Prevalence   0.2845   0.1930   0.1749   0.1639   0.1838
## Balanced Accuracy      1.0000   0.9987   0.9988   0.9991   1.0000
```

##Expected Out-of-Sample Error Rate
As shown in the above output the accuracy is 99.39% for the cross validation data, indicating that the out-of-sample error rate is 0.61%


#Predictions for Project Prediction Quiz
The follow predicts classifications for the 20 observations in the testing data set. 

```r
#prediction classification for 20 sample testing data
testing.predictions <- predict(modelFit, newdata = testing_raw)

#output predictions
testing.predictions
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

#References
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.


```



