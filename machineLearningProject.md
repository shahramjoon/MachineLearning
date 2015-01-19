---
title: "A predictive Model to Recognize Quality of Weight Lifting Exercises"
output: html_document
---


##Synopsis

In this report, we try to create a predictive model to recognize the quality of Weight LIfting Exercises has done by participants in the study. We are using random forest modeling with 3 folden cross validation for our modeling. 


Study and Data is done by 

Loveless, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.


## Exploratory Data Analyses

The Input Data that is gathered by study consists of 19622 exercises done by 6 participants in the study. Each exercises is measured by 160 metrics. Each exercise is been asked to be done as a specific fashion classified as A,B,C,D and E: 

exactly according to the specification (Class A), 

throwing the elbows to the front (Class B), 

lifting the dumbbell only halfway (Class C), 

lowering the dumbbell only halfway (Class D) and 

throwing the hips to the front (Class E)




```r
library(caret)

setwd("c:/temp")

training <- read.table("pml-training.csv", 
                  sep=",", 
                  header=TRUE, 
                  na.strings=c("NA","#DIV/0!", ""), 
                  stringsAsFactors=FALSE)



testing <- read.table("pml-testing.csv", 
                       sep=",", 
                       header=TRUE, 
                       na.strings=c("NA","#DIV/0!", ""), 
                       stringsAsFactors=FALSE)


dim ( training)
```

```
## [1] 19622   160
```

```r
dim ( testing)
```

```
## [1]  20 160
```

Because training Data has a lot of measures that have NAs, keeping these measures reduces the accuracy of predictive model. Therefore we determined what those measures are and we removed them from the model 



```r
training$classe <- factor(training$classe)


keep.vars <- (apply(is.na(training), 2, mean) <= .95)
training <- training[,keep.vars]
testing <- testing[,keep.vars]
training$X <- NULL
training$user_name <- NULL
training$raw_timestamp_part_1<- NULL
training$raw_timestamp_part_2<- NULL
training$cvtd_timestamp<- NULL
training$new_window <- NULL
training$num_window <- NULL

testing$X <- NULL
testing$user_name <- NULL
testing$raw_timestamp_part_1<- NULL
testing$raw_timestamp_part_2<- NULL
testingcvtd_timestamp<- NULL
testing$new_window <- NULL
testing$num_window <- NULL
```


We also removed measures that have a high co-relation to each other. 





```r
correlationMatrix <- cor ( training[,1:52])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.8)
training <- training [ , - highlyCorrelated]
```

We ended up to training data with 


```r
str( training)
```

```
## 'data.frame':	19622 obs. of  41 variables:
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
##  $ gyros_belt_y        : num  0 0 0 0 0.02 0 0 0 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ magnet_belt_x       : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
##  $ magnet_belt_y       : int  599 608 600 604 600 603 599 603 602 609 ...
##  $ magnet_belt_z       : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm           : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
##  $ gyros_arm_z         : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
##  $ accel_arm_y         : int  109 110 110 111 111 111 111 111 109 110 ...
##  $ accel_arm_z         : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
##  $ magnet_arm_x        : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
##  $ magnet_arm_z        : int  516 513 513 512 506 513 509 510 518 516 ...
##  $ roll_dumbbell       : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell      : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
##  $ yaw_dumbbell        : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 0 0 -0.02 0 0 0 0 0 0 ...
##  $ accel_dumbbell_y    : int  47 47 46 48 48 48 47 46 47 48 ...
##  $ magnet_dumbbell_x   : int  -559 -555 -561 -552 -554 -558 -551 -555 -549 -558 ...
##  $ magnet_dumbbell_y   : int  293 296 298 303 292 294 295 300 292 291 ...
##  $ magnet_dumbbell_z   : num  -65 -64 -63 -60 -68 -66 -70 -74 -65 -69 ...
##  $ roll_forearm        : num  28.4 28.3 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 ...
##  $ yaw_forearm         : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -152 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
##  $ gyros_forearm_y     : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
##  $ accel_forearm_x     : int  192 192 196 189 189 193 195 193 193 190 ...
##  $ accel_forearm_y     : int  203 203 204 206 206 203 205 205 204 205 ...
##  $ accel_forearm_z     : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
##  $ magnet_forearm_x    : int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
##  $ magnet_forearm_y    : num  654 661 658 658 655 660 659 660 653 656 ...
##  $ magnet_forearm_z    : num  476 473 469 469 473 478 470 474 476 473 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
dim ( training)
```

```
## [1] 19622    41
```

To validate our model based on training data that is provided, we divided the training data to two pieces, we allocated %75 of rows chosen randomly to training data and remaining %25 t validation data to validate our model and compare prediction with validation data. 



```r
set.seed(825)


inTrain <- createDataPartition(y=training$classe, p=.75, list=FALSE )                           

src <- training

training <- src[inTrain,]
validation <- src [-inTrain,]
```


For our predictive modeling, we used 3 fold cross validation. We used random Foresting model method for training with 100 trees.



```r
control1 <- trainControl(method = "cv", number = 3, allowParallel = TRUE)


model1 <- train(classe~., data = training, method="rf", ntrees= 100, trControl = control1, prox= TRUE)


model1
```

```
## Random Forest 
## 
## 14718 samples
##    40 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## 
## Summary of sample sizes: 9811, 9812, 9813 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9879739  0.9847850  0.001079091  0.001364059
##   21    0.9880416  0.9848704  0.002769163  0.003503697
##   40    0.9811789  0.9761873  0.004329492  0.005478465
## 
## Accuracy was used to select the optimal model using 
##  the largest value.
## The final value used for the model was mtry = 21.
```

```r
model1$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE,      ntrees = 100) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 21
## 
##         OOB estimate of  error rate: 0.79%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4181    3    1    0    0 0.0009557945
## B   18 2822    6    0    2 0.0091292135
## C    0   25 2529   13    0 0.0148032723
## D    2    1   30 2377    2 0.0145107794
## E    0    0    2   12 2692 0.0051736881
```


                
To measure accuracy of our model, We applied the model to validation dataset 




```r
pred <- predict ( model1, validation )

table ( pred, validation$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 1391    7    0    1    0
##    B    3  937    6    0    0
##    C    0    5  847    9    0
##    D    0    0    2  792    4
##    E    1    0    0    2  897
```

The accuracy of our model against Validation data was


```r
sum ( pred == validation$classe) / nrow(validation)
```

```
## [1] 0.9918434
```

Now, we are in position to apply our model to final Testing Data that was provided


```r
prediction <- predict ( model1, testing)

prediction 
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
