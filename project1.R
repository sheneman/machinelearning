#
#
#  Coursera - Practical Machine Learning
# 
#  Class Assignment
#
#  Luke Sheneman
#
#------------------------------------------------------------------------------------
#
#  This project was possible by using the Weight Lifting Exercise (WLE) Human Activity 
#  Recognition Dataset:
#
#   Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity 
#   Recognition of Weight Lifting Exercises. Proceedings of 4th International 
#   Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: 
#   ACM SIGCHI, 2013.
#
#    http://groupware.les.inf.puc-rio.br/har
#
#
#  SOME NOTES:
#  -----------
#  Six young health participants were asked to perform one set of 10 repetitions
#  of the Unilateral Dumbbell Biceps Curl in five different fashions (the classe variable): 
#  exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting
#  the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and 
#  throwing the hips to the front (Class E).


library(ggplot2)
library(plyr)
library(caret)
library(ipred)

set.seed(1234)

#
# preprocess_data() -
#
# A function that removed unneeded columns, normalized data classes,
# imputes missing values for the training data set and the testing data set
#
preprocess_data <- function(d, dtype, test_col_names) {
  
  # limit our columns to the ones we know are not entirely missing values in the testing dataset
  if (dtype == "training") {
    tmp_classe <- as.factor(d$classe)
  }
  nd <- d[,names(d) %in% test_col_names]
  
  # save the factor variable "user_name"
  tmp_user_name   = as.factor(nd$user_name)
  
  # convert all integer columns to numeric columns for better interoperability
  for (i in 1:ncol(nd)) {
    if(class(nd[,i])=="integer") {
      nd[,i] <- as.numeric(nd[,i])
    }
  }
  
  # temporarily remove the factor variables
  nd$user_name  <- NULL;
  nd$classe     <- NULL;
  
  # impute missing values across all columns by computing the mean of the column
  # and replacing missing values with that mean
  for(i in 1:ncol(nd)) {
    col_mean <- mean(nd[,i], na.rm = TRUE)
    nd[!complete.cases(nd[,i]),i] <- col_mean
  }

  # re-add a subset of the removed factor variables
  nd$user_name <- tmp_user_name
  
  if(dtype == "training") {
    nd$classe <- tmp_classe
  }
  
  # return the processed data frame
  return(nd)
}



# Read the training dataset and mark things as NA a needed
training_data <- 
    read.csv("pml-training.csv", 
    stringsAsFactors=FALSE,na.strings=c("NA","","#DIV/0!"))

testing_data <- 
  read.csv("pml-testing.csv", 
           stringsAsFactors=FALSE,na.strings=c("NA","","#DIV/0!"))

# remove all columns in the testing dataset for which there are ONLY missing values and 
# remove a couple other unhelpful columns as well
new_testing_data <- testing_data[,colSums(is.na(testing_data)) != nrow(testing_data)]
new_testing_data$new_window     <- NULL
new_testing_data$X              <- NULL
new_testing_data$num_window     <- NULL
new_testing_data$problem_id     <- NULL
new_testing_data$cvtd_timestamp <- NULL

# pre-process the datasets to set the stage for modeling and prediction
new_training_data <- preprocess_data(training_data,dtype="training",test_col_names=names(new_testing_data))
new_testing_data  <- preprocess_data(new_testing_data,dtype="testing",test_col_names=names(new_testing_data))

# build a model by training on the training set using the random forest 
# classification method.   We use the built-in k-fold cross validation capability 
# in caret's train() function to do cross validation with k folds.

control <- trainControl(method="cv", number=5, verboseIter=TRUE)
modelfit <- train(classe ~ ., data=new_training_data, trControl=control, method="rpart")

# predict new results given the model and the small testing data
predictions <- predict(modelfit, new_training_data)

cm <- confusionMatrix(predictions,new_training_data$classe)
print(cm)

#
# BAGGING
#

control <- trainControl(method="cv", number=5, verboseIter=TRUE)
modelfit <- train(classe ~ ., data=new_training_data, trControl=control, method="treebag")

# predict new results given the model and the small testing data
predictions <- predict(modelfit, new_training_data)

cm <- confusionMatrix(predictions,new_training_data$classe)
print(cm)



#
# Since bagging worked far better than rpart, lets use cross-validated treebag model 
# to predict the outcomes in the test dataset
#
final_prediction <- predict(modelfit, new_testing_data)
print(final_prediction)

#
# lets create a model that is not cross validated and see if we get the same prediction
#
control <- trainControl(method="none", verboseIter=TRUE)
modelfit <- train(classe ~ ., data=new_training_data, trControl=control, method="treebag")
final_prediction <- predict(modelfit, new_testing_data)
print(final_prediction)