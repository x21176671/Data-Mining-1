########################
##      NBA            #
########################

library(ggthemes)
library(scales)
library(Amelia)
library(dplyr)
library(leaps)
library(MASS)
library(caret)
library(car)
library(haven)
library(ggplot2)
library(gridExtra)
library(regclass)
library(AICcmodavg)
#library(xlsx)
library(magrittr)

setwd("college/nci/semester2/data mining and machine learning 1/project")
NBA_data <- read.csv("NBA12_18_teamBoxScore.csv",  # full kepler dataset of exoplanet candidates 
                        header=T, na.strings=c(""), stringsAsFactors = T)
NBA_data

var_4_analysis <- names(NBA_data) %in%  # list of variable names we're interested in for analysis
  c("teamRslt",
    "teamPTS",
    "teamAST" ,  
     "teamTO" ,
    "teamSTL" ,
    "teamBLK"  ,
    "teamPF"  ,
    "teamFGA",   
     "teamFGM",
     "teamFG.",
    "team2PA"  ,
    "team2PM"   ,
    "team2P."   ,
     "team3PA"   ,
    "team3PM" ,
    "team3P."  ,
    "teamFTA" ,
    "teamFTM" ,  
     "teamFT." ,
    "teamTRB"  ,
     "opptAbbr", 
    "opptPTS"   , 
    "opptAST" ,  
     "opptTO"  ,
    "opptSTL"   ,
    "opptBLK" ,
    "opptPF"  ,
    "opptFGA" ,  
     "opptFGM" ,
    "opptFG.",
    "oppt2PA" ,
    "oppt2PM" , 
    "oppt2P."   ,
     "oppt3PA" , 
    "oppt3PM" ,
    "oppt3P."  ,
    "opptFTA"  ,
    "opptFTM"  , 
     "opptFT." ,
    "opptTRB"
  )

NBA_subset1 <- NBA_data[var_4_analysis] # subset of the data for analysis
write.csv(NBA_subset1, "NBA_subset1")
# still taking 41 features here, find a way to slim it down 
# opp result is just inverse of team result, carrying same info
# opp abbreviation doesn't carry much weight without also knowing team abbreviation 


#find missing attributes
sapply(NBA_subset1,function(x) sum(is.na(x)))
# read in data
NBA_data <- read.csv("NBA_subset1", header=T, na.strings=c(""), stringsAsFactors = T)
# no na's
# in reality, the opposition has a great impact of the outcome of the game, but for simplicity we will ignore it here 
# and just look at the general trends league wide 
NBA_data <- subset(NBA_data, select = c(-opptAbbr))
# need to futher trim down variable selection to ease interpretation
# lets reduce shooting to % rather than attempts and mades 
NBA_data <- subset(NBA_data, select = c(-teamFGA,-teamFGM,-team2PA,-team2PM,-team3PA,-team3PM,-teamFTA,-teamFTM,
                                        -opptFGA,-opptFGM,-oppt2PA,-oppt2PM,-oppt3PA,-oppt3PM,-opptFTA,-opptFTM))
# lets also ignore pts as this will carry too much weight to leave any interpretation on the other stats
NBA_data <- subset(NBA_data, select = c(-teamPTS, -opptPTS))

boxplot(subset(NBA_data, select = c(-teamRslt, -X)))
boxes <- subset(NBA_data, select = c(-teamRslt, -X))

# remove outliers 
library(plyr)
is_outlier <- function(x, na.rm = FALSE) {
  qs = quantile(x, probs = c(0.25, 0.75), na.rm = na.rm)
  
  lowerq <- qs[1]
  upperq <- qs[2]
  iqr = upperq - lowerq 
  
  extreme.threshold.upper = (iqr * 3) + upperq
  extreme.threshold.lower = lowerq - (iqr * 3)
  
  # Return logical vector
  x > extreme.threshold.upper | x < extreme.threshold.lower
}

#' Remove rows with outliers in given columns
#' 
#' Any row with at least 1 outlier will be removed
#' 
#' @param df A data.frame
#' @param cols Names of the columns of interest. Defaults to all columns.
#' 
#' 
remove_outliers <- function(df, cols = names(df)) {
  for (col in cols) {
    cat("Removing outliers in column: ", col, " \n")
    df <- df[!is_outlier(df[[col]]),]
  }
  df
}


NBA_data <- remove_outliers(NBA_data, c("teamAST", "teamTO" , "teamSTL", "teamBLK", "teamPF", 
                                          "teamFG.", 'team2P.', "team3P.", "teamFT.", "teamTRB",
                                        "opptAST",  "opptTO" ,  "opptSTL",  "opptBLK",  "opptPF", 
                                          "opptFG.",  "oppt2P.",  "oppt3P.",  "opptFT.",  "opptTRB"))

boxplot(subset(NBA_data, select = c(-teamRslt, -X)))
write.csv(NBA_data, "NBA_subset_no_out.csv")
# still taking 41 features here, find a way to slim it down 
# opp result is just inverse of team result, carrying same info
# opp abbreviation doesn't carry much weight without also knowing team abbreviation 

NBA_data <- read.csv("NBA_subset_no_out.csv", header=T, na.strings=c(""), stringsAsFactors = T)
#split dataset for train and test 80:20
testsize <- floor(0.8*nrow(NBA_data))
set.seed(21176671) #set seed to split dataset to keep the same random sample 
picked <- sample(seq_len(nrow(NBA_data)), size = testsize )
train <- NBA_data[picked,] # set to create model with (random 80% of the default set)
test <- NBA_data[-picked,] # set to test model with (other 20%)

#fit linear model with all IV first 
fit0 <- glm(train$teamRslt ~ ., data = train, family = binomial) # ACC = 94.2%
summary(fit0)
# only home stats
fit1 <- glm(train$teamRslt ~ train$teamAST+train$teamTO +train$teamSTL+train$teamBLK+train$teamPF+
              train$teamFG.+train$team2P.+train$team3P.+train$teamFT.+train$teamTRB, data = train, family = binomial)  # ACC = 81.4%
summary(fit1)
# only oppt stats
fit2 <- glm(train$teamRslt ~ train$opptAST+ train$opptTO + train$opptSTL+ train$opptBLK+ train$opptPF+
              train$opptFG.+ train$oppt2P.+ train$oppt3P.+ train$opptFT.+ train$opptTRB, data = train, family = binomial)  # ACC = 81.9%
summary(fit2)

#check for multicollinearity 
#correlation matrix ### checking variables for multicoliniarity  
library(ggcorrplot)
#Data needs to be all numeric
train$teamRslt <- as.numeric(train$teamRslt)
corr <- round(cor(train, method = 'pearson'), digits = 2)
head(corr[, 1:6])
ggcorrplot(corr, hc.order = TRUE, type = "lower",
           lab = TRUE)
# VIF 
VIF(fit0)
VIF(fit1)
VIF(fit2)

NBA_data <- subset(NBA_data, select = c(-team2P.,-team3P.,-oppt2P., - oppt3P.))
write.csv(NBA_data, "NBA_subset_no_out_no_multi_col.csv")


# run it back bud

# read in data
NBA_data <- read.csv("NBA_subset_no_out_no_multi_col.csv", header=T, na.strings=c(""), stringsAsFactors = T)
NBA_data <- subset(NBA_data, select = c(-X.2,-X.1, -X)) # removes row id cols 
 
#split dataset for train and test 80:20
testsize <- floor(0.8*nrow(NBA_data))
set.seed(21176671) #set seed to split dataset to keep the same random sample 
picked <- sample(seq_len(nrow(NBA_data)), size = testsize )
train_final <- NBA_data[picked,] # set to create model with (random 80% of the default set)
test_final <- NBA_data[-picked,] # set to test model with (other 20%)

#fit linear model with all IV first 
fit0 <- glm(train_final$teamRslt ~ ., data = train_final, family = binomial) # ACC = 92%
summary(fit0)
# only home stats
attach(train_final)
fit1 <- glm(teamRslt ~ teamAST+teamTO +
              teamSTL+teamBLK+teamPF+
              teamFG.+teamFT.+teamTRB, data = train_final, family = binomial)  # ACC = %
summary(fit1)
# only oppt stats
fit2 <- glm(teamRslt ~ opptAST+ opptTO + 
              opptSTL+ opptBLK+ opptPF+
              opptFG.+opptFT.+ opptTRB, data = train_final, family = binomial)  # ACC =%
summary(fit2)

#check for multicollinearity 
#correlation matrix ### checking variables for multicoliniarity  
library(ggcorrplot)
#Data needs to be all numeric
train$teamRslt <- as.numeric(train$teamRslt)
corr <- round(cor(train, method = 'pearson'), digits = 2)
head(corr[, 1:6])
ggcorrplot(corr, hc.order = TRUE, type = "lower",
           lab = TRUE)
# VIF 
VIF(fit0)
VIF(fit1)
VIF(fit2)


# evaluate model 
library(ISLR)
library(stats)
predicted0 <- predict(fit0, test_final, type ="response")
pred_binary <- ifelse(predicted0>0.5, 1, 0) # sets to 1 if >0.5 , else sets to 0 
pred_tab <- table(Predicted=pred_binary, Actual=test_final$teamRslt)
paste("fit0")
pred_tab
accuracy0 <- sum(diag(pred_tab))/sum(pred_tab)
accuracy0
predicted1 <- predict(fit1, test_final, type ="response")
pred_binary <- ifelse(predicted1>0.5, 1, 0) # sets to 1 if >0.5 , else sets to 0 
pred_tab <- table(Predicted=pred_binary, Actual=test_final$teamRslt)
paste("fit1")
pred_tab
accuracy1 <- sum(diag(pred_tab))/sum(pred_tab)
accuracy1
predicted2 <- predict(fit2, test_final, type ="response")
pred_binary <- ifelse(predicted2>0.5, 1, 0) # sets to 1 if >0.5 , else sets to 0 
pred_tab <- table(Predicted=pred_binary, Actual=test_final$teamRslt)
paste("fit2")
pred_tab
accuracy2 <- sum(diag(pred_tab))/sum(pred_tab)
accuracy2

#roc curves

library("ROCR")
pred <- predict(fit0, test_final, type = "response")
pred <- prediction(pred, test_final$teamRslt)
roc <- performance(pred, "tpr", "fpr")
plot(roc, colorize = T, lwd = 3,lty = 1, main = "test data ROC curve ", label = "all stats")
pred <- predict(fit1, test_final, type = "response")
pred <- prediction(pred, test_final$teamRslt)
roc <- performance(pred, "tpr", "fpr")
plot(roc, colorize = T, lwd = 2, lty = 2, label = "all stats", add=T)
pred <- predict(fit2, test_final, type = "response")
pred <- prediction(pred, test_final$teamRslt)
roc <- performance(pred, "tpr", "fpr")
plot(roc, colorize = T, lwd = 1, lty = 3, label = "all stats", add=T)
legend(0.6, 0.3,lwd = c(3,2,1), legend = c("all stats", "home only", "away only"))
abline(a = 0, b = 1) 

# pseudo R^2
library(DescTools)
cox_snell <- PseudoR2(fit0, which = "CoxSnell")
nagelkerke <- PseudoR2(fit0, which = "Nagelkerke")
cox_snell
nagelkerke
cox_snell <- PseudoR2(fit1, which = "CoxSnell")
nagelkerke <- PseudoR2(fit1, which = "Nagelkerke")
cox_snell
nagelkerke
cox_snell <- PseudoR2(fit2, which = "CoxSnell")
nagelkerke <- PseudoR2(fit2, which = "Nagelkerke")
cox_snell
nagelkerke


