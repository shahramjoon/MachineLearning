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



```{r, echo=TRUE }

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

dim ( testing)

```

Because training Data has a lot of measures that have NAs, keeping these measures reduces the accuracy of predictive model. Therefore we determined what those measures are and we removed them from the model 


```{r, echo=TRUE }

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




```{r, echo=TRUE }

correlationMatrix <- cor ( training[,1:52])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.8)
training <- training [ , - highlyCorrelated]


```

We ended up to training data with 

```{r, echo=TRUE }

str( training)

dim ( training)

```

To validate our model based on training data that is provided, we divided the training data to two pieces, we allocated %75 of rows chosen randomly to training data and remaining %25 t validation data to validate our model and compare prediction with validation data. 


```{r, echo=TRUE }

set.seed(825)


inTrain <- createDataPartition(y=training$classe, p=.75, list=FALSE )                           

src <- training

training <- src[inTrain,]
validation <- src [-inTrain,]


```


For our predictive modeling, we used 3 fold cross validation. We used random Foresting model method for training with 100 trees.


```{r, echo=TRUE }

control1 <- trainControl(method = "cv", number = 3, allowParallel = TRUE)


model1 <- train(classe~., data = training, method="rf", 
                ntrees= 100, trControl = control1, prox


model1


model1$finalModel
```


                
To measure accuracy of our model, We applied the model to validation dataset 



```{r, echo=TRUE }

pred <- predict ( model1, validation )

table ( pred, validation$classe)


```

The accuracy of our model against Validation data was

```{r, echo=TRUE }

sum ( pred == validation$classe) / nrow(validation)

```

Now, we are in position to apply our model to final Testing Data that was provided

```{r, echo=TRUE }

prediction <- predict ( model1, testing)

prediction 

```
