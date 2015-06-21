setwd("~/Coursera/MachineLearning/project")
library(ggplot2); library(reshape); library(caret); library(kernlab); library(randomForest)
library(doParallel)

data <- read.csv("pml-training.csv")
data_test <- read.csv("pml-testing.csv")
registerDoParallel(cores=2)

Names_keep_1 <- colnames(data)[grep("gyros|accel|magnet|pitch|roll|yaw|picth", colnames(data))]
Names_keep_2 <- Names_keep_1[-grep("avg|max|min|var|stddev|skewness|amplitude|skewness|kurtosis", 
                                   Names_keep_1)]
set.seed(8484)
#Use 60% of the data for model and predictor selection (cross validating etc)
inTrain_F <- createDataPartition(y=data$classe, p=0.6, list=F)
# set aside 40% of data for sole use in estimating out of sample error after model and predictors are selectted
final_test_data <- data[ -inTrain_F,]
model_building_data <- data[ inTrain_F , c("classe" , Names_keep_2)]
# model_building_data is 60% of the data so 
inTrain <- createDataPartition(y=model_building_data$classe, p=0.8, list=F)
Training_data <- model_building_data[inTrain ,]; Testing_data <- model_building_data[-inTrain ,]

#look at correlation to remove predictors (columns)
drops <- c("classe"); drops2 <- c("user_name", "classe")
sub_data <- Training_data[ ,!(names(Training_data) %in% drops)]; col_sub_data <- colnames(sub_data)
M <- abs(cor(sub_data))
sub_data <- sub_data[ , -findCorrelation(M, cutoff = 0.8)]
Training_data <- cbind(Training_data$classe, sub_data)
colnames(Training_data) <- c("classe", colnames(sub_data))

cortest <- function(fit, pred){ # function for calculating residuals/ predictors cor,
        df <- data.frame()
        TDclasse <- as.numeric(data$classe)
        for(i in 2:length(pred)){
                cor.obj <- cor.test(TDclasse, as.vector(data[ , pred[i] ]))
                df[i,"corellaton: fit residual vs. predictors"] <- as.numeric(cor.obj$estimate)
                df[i, "P-value"] <- cor.obj$p.value
        }
        rownames(df) <- pred
        df
}

cor_DF <- cortest(1,colnames(Training_data))
cor_index <- which(abs(cor_DF[ ,1]) > 0.05)

new_colnames <- colnames(Training_data)[cor_index]
Training_data <- Training_data[ ,c("classe", new_colnames)]
# and one more partition just for computational speed
inTrain_s <- createDataPartition(y=Training_data$classe, p=0.1, list=F)

Folds <- createFolds(y=Training_data[inTrain_s,]$classe, k=6 )
#?createFolds()
ctrl <- trainControl(method = "repeatedcv", repeats = 6, number = 6, classProbs = T, index = Folds)

modelFit <- train(classe ~ ., method = "rf", 
        trainControl = ctrl, data = Training_data[inTrain_s,], parallel = T)

#modelFit$modelInfo
#save(modelFit, file="fit_at_60of60")

#load("fit_at_60of60")

modelFit$finalModel

data_temp_test <- Training_data[ -inTrain_s, ]
confusionMatrix(predict(modelFit, data_temp_test) , data_temp_test$classe )

data_temp_test <- final_test_data[ , colnames(Training_data) ]
confusionMatrix(predict(modelFit, data_temp_test) , final_test_data$classe )

plot_confusion <- confusionMatrix(predict(modelFit, data_temp_test) , Training_data[ -inTrain_s, ]$classe )


confusion <- as.table(prop.table(plot_confusion$table, margin = 2))
c2 <- round(confusion, 3)


confusion
#results submission
# answers = predict(modelFit, data_test)
# 
# pml_write_files = function(x){
#         n = length(x)
#         for(i in 1:n){
#                 filename = paste0("problem_id_",i,".txt")
#                 write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
#         }
# }
# 
# setwd("~/Coursera/MachineLearning/project/answers")
# pml_write_files(answers)

answers
