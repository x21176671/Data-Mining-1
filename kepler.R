########################
## NASA EXOPLANET DISPO#
########################
  
setwd("college/nci/semester2/data mining and machine learning 1/project")
kepler_data <- read.csv("kepler_data_no_desc.csv",  # full kepler dataset of exoplanet candidates 
                        header=T, na.strings=c(""), stringsAsFactors = T)
kepler_data 

var_4_analysis <- names(kepler_data) %in%  # list of variable names we're interested in for analysis
      c("kepoi_name", 
        "koi_disposition",   "koi_score" , # current disposition and score (confidence) of the disposition
        "koi_period"  , # orbital period of the planet around the host star      
         "koi_time0bk",     # Transit epoch, when was the first detection
         "koi_duration",  # duration of the orbit    
        "koi_depth"    ,    # depth of the transit by fraction of stellar flux loss
         "koi_ror",    # ratio of planet radius to star radius 
        "koi_srho",  # fitted stellar density derived from host star's light curve 
        "koi_prad" ,   # radius of the exoplanet candidate          
         "koi_teq" ,   # approx temperature of the planet 
         "koi_dor" ,   # distance between planet and star at mid transit over stellar radius 
         "koi_count",   # no. of candidates found in system
         "koi_steff",   # stellar effective temperature in K
         "koi_slogg",   # stellar surface gravity ==
         "koi_smet",   # stellar metallicity          
         "koi_srad",   # stellar radius 
         "koi_smass"   # stellar 
         )

kepler_subset1 <- kepler_data[var_4_analysis] # subset of the data for analysis
kepler_subset1$koi_count <- as.integer(kepler_subset1$koi_count) # take count of planet candidates as factor, not numeric
write.csv(kepler_subset1, "kepler_subset1")
# eccentricity all 0/NA
# age all NA                 check kepler source for updated dataset/corrections, otherwise exclude these
# remove eccentricity and age, no data 

#find missing attributes
sapply(kepler_subset1,function(x) sum(is.na(x)))
# several NAs found, decide which need removing and which are tolerable (KOI score fine, transit depth etc maybe not )
# several features with 363 NAs, likely removing rows with NAs from 1 of these features knocks out all the other features with 363 NAs
kepler_subset2 <- kepler_subset1[!is.na(kepler_subset1$koi_depth),]
sapply(kepler_subset2,function(x) sum(is.na(x)))
# only KOI disposition score (1206) and metallicity (23) NAs left
# KOI score is not something we will be considering as a prediction parameter as we would like to focus more on observational data rather than
# features generated from NASA's own data mining
# metallicity has so few NA's and may yet be excluded from our analysis, we will not remove these rows for the time being 
# no way to reliably estimate metalicity from the given other features, could be possible to make a reasonable attempt using ML 
# there is also so few features with NA here that it is not worth it and they will simply be removed 
kepler_subset3 <- kepler_subset1[!is.na(kepler_subset1$koi_smet),]
# remove the koi name 
kepler_subset3 <- subset(kepler_subset3, select = -c(kepoi_name))

##########################################################################
########                   EVALUATION                          ###########
##########################################################################
# LETS CONSIDER A FEW DIFFERENT METHODS: (PICK 1/2 FOR PAPER)
# BAYESIAN CLASSIFIER
# LOG REGRESSION 
# DECISION TREES 

#BAYES
set.seed(21176671)
#split data 80:20
index <- sample(seq_len(nrow(kepler_subset3)),size = 0.8*nrow(kepler_subset3))
train <- kepler_subset3[index,]
test <- kepler_subset3[-index,]
#isolate the target variable
target_varaible <- train$koi_disposition
#remove target variable from the training set
train <- subset(train, select=-c(koi_disposition))
# also should variables of no machine learning values (names)  
train <- subset(train, select=-c(kepoi_name))
#analysis with all other variables in
library(e1071)
Bayes_classifier_koi_score <- naiveBayes(train, target_varaible)
test_pred_koi_score <- predict(Bayes_classifier_koi_score, test)
library(gmodels)
CrossTable(test_pred_koi_score, test$koi_disposition,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))
library(caret)
confusionMatrix(test_pred_koi_score,test$koi_disposition)
#REMOVE DISPO SCORE 
#(TRY GET A CLOSE ML MODEL TO WHATEVER WAY NASA HAVE CREATED THIS SCORE: 
#Monte Carlo technique such that the score's value is equivalen
#where the Robovetter yields a disposition of CANDIDATE.)  
train_no_koi_score <- subset(train, select=-c( koi_score))
library(e1071)
Bayes_classifier_no_koi_score <- naiveBayes(train_no_koi_score, target_varaible)
test_pred_no_koi_score <- predict(Bayes_classifier_no_koi_score, test)
library(gmodels)
CrossTable(test_pred_no_koi_score, test$koi_disposition,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))
confusionMatrix(test_pred_no_koi_score,test$koi_disposition)


# DECISION TREES 
library(tree)
library(rpart)
library(rpart.plot)
library(ISLR)
library(stats)
#need the response variable (disposition) back in. easiest to just reset the train:test with the same seed 
#remove koi, score to only look at meaningful variables
kepler_dt <- subset(kepler_subset3, select=-c(koi_score))
set.seed(21176671)
#split data 80:20
index <- sample(seq_len(nrow(kepler_dt)),size = 0.8*nrow(kepler_dt))
train <- kepler_dt[index,]
test <- kepler_dt[-index,]
#isolate the target variable
target_varaible <- train$koi_disposition
tree.kepler <-  tree(koi_disposition~., data=train) # build largest tree w/ all variables from dataset 
summary(tree.kepler)
print(tree.kepler,pretty=0)
plot(tree.kepler)
text(tree.kepler , pretty =0)
exo_tree <- rpart(koi_disposition~., data = train, method = "class")

#pruning the tree using the CP factor
# CP factor is a normalized version of alpha
cp_tab <- printcp(exo_tree)

#xerror
min_cp = exo_tree$cptable[which.min(exo_tree$cptable[,"xerror"]),"CP"]
ptree<- prune(exo_tree, cp= min_cp)
print(ptree)
summary(ptree)
print(ptree,pretty=0)
plot(ptree)
text(ptree, pretty =0)

#predictions of the tree
predictions <- predict(exo_tree, test,type="class")
p_predictions <- predict(ptree, test,type="class")
#table the predicitons
table_full <- table(as.factor(test$koi_disposition), predictions)
table_prune <- table(as.factor(test$koi_disposition), p_predictions)
table_prune
#accuracy 
tree_acc <- mean(predictions==test$koi_disposition)
ptree_acc <- mean(p_predictions==test$koi_disposition)
confusionMatrix(predictions, test$koi_disposition)
confusionMatrix(p_predictions, test$koi_disposition)

