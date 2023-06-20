
#script to analyze use/availability for NWT bison
#all animals F
#period restricted to September - March
#data has used and available

library(lubridate)
library(dplyr)
library(corrplot)
library(lme4)
library(MuMIn)
library(goeveg)

rm(list = ls())
gc()




#### prepare data for analysis ####

summary(dat <- readRDS("dat_NahanniDehcho.rds"))

# split into two sets - used and available
dat.used <- dat %>% filter(detection == 1)
dat.available <- dat %>% filter(detection == 0)

# check for correlation among variables
dat.available %>% select(where(is.numeric)) %>%
  cor() %>% corrplot(method = 'number')
# slope and elevations have 0.73 correlation: remove slope
dat.available <- dat.available %>% select(-slope)
dat.used <- dat.used %>% select(-slope)

# decide which predictors to include based on their distribution
dat.available %>% select(where(is.numeric)) %>%
  summarise_all(list(mean = mean, cv = cv,
                     zeros = function(x){return(mean(x == 0))}))
# remove veg categories with < 1% cover (veg_100 and veg_40),
# and log_t2fire due to low variability 
dat.available <- dat.available %>% select(-veg_100, -veg_40, -log_t2fire)
dat.used <- dat.used %>% select(-veg_100, -veg_40, -log_t2fire)

# decide which individuals to include based on space-use variability
dat.used %>% group_by(ID) %>%
  summarise(across(where(is.numeric), sd)) %>%
  print(n = 9)
plot(longitude ~ latitude, data = dat.used)
points(longitude ~ latitude, data = dat.used,
       subset = (ID == "WOBI500"), pch = 19, cex = 1, col = "red")
points(longitude ~ latitude, data = dat.used,
       subset = (ID == "WOBI508"), pch = 19, cex = 1, col = "blue")
# WOBI500 and WOBI508 seem to be invalid - remove
dat.used <- dat.used %>% filter(!(ID == "WOBI500" | ID == "WOBI508"))

# calculate means and SDs for scaling and centering all predictors
means <- dat.available %>% select(where(is.numeric)) %>% summarise_all(mean)
SDs <- dat.available %>% select(where(is.numeric)) %>% summarise_all(sd)




#### fit individual RSFs ####

unique.IDs <- unique(dat.used$ID)
results.ML <- array(dim = c(length(unique.IDs), 14))
results.SE <- array(dim = c(length(unique.IDs), 14))
for (i in 1:length(unique.IDs)) { # loop over individuals
  
  #form an individual dataset
  dat.ind <- dat.used %>% filter(ID == unique.IDs[i]) %>%
    # sample one point per day
    group_by(day) %>% slice_sample(n = 1) %>%
    # add the available points
    rbind(dat.available)
  
  mymodel <- glm(detection ~ 1 +
                   I((veg_20 - means$veg_20) / SDs$veg_20) +
                   # I((veg_35 - means$veg_35) / SDs$veg_35) +
                   I((veg_55 - means$veg_55) / SDs$veg_55) +
                   I((veg_85 - means$veg_85) / SDs$veg_85) +
                   # I((veg_211 - means$veg_211) / SDs$veg_211) +
                   I((veg_212 - means$veg_212) / SDs$veg_212) +
                   # I((veg_213 - means$veg_213) / SDs$veg_213) +
                   I((veg_225 - means$veg_225) / SDs$veg_225) +
                   #I((veg_235 - means$veg_235) / SDs$veg_235) +
                   # I((dem_3581 - means$dem_3581) / SDs$dem_3581) +
                   # I((northness - means$northness) / SDs$northness) +
                   I((log_roughness - means$log_roughness) / SDs$log_roughness) +
                   I((log_d2road  - means$log_d2road) / SDs$log_d2road),
                 weights = 10000 ^ as.numeric(detection == 0),
                 family = binomial(link = 'logit'),
                 data = dat.ind)
  
  results.ML[i, 1:length(summary(mymodel)$coefficients[, 1])] <- summary(mymodel)$coefficients[, 1] # record coefficient estimates 
  results.SE[i, 1:length(summary(mymodel)$coefficients[, 2])] <- summary(mymodel)$coefficients[, 2] # record coefficient standard errors
}

# examine overlap of inverse variance-weighted selection coefficients with 0
# which implies lack of consensus among individuals 
w <- sweep(results.SE ^ -2, 2, apply((results.SE ^ -2), 2, sum), FUN = "/") # calculate inverse-variance weights  
boxplot(results.ML * w, ylim = c(-0.1, 0.1))  
abline(0,0)
# after fitting the model to all individuals,
# decide on a single predictor to remove -
# the one that performs worst in terms of overlap with 0.
# Now rerun and repeat until a reasonable consensus model is identified

# examine the correlation between parameters - high correlation may indicate overfitting 
results.ML %>% cor() %>% corrplot(method = 'number')





#### final model ####

dat.daily <- dat.used %>%
  group_by(ID, day) %>%
  slice_sample(n = 1) %>%
  rbind(dat.available)
summary(final.model <- glm(detection ~ 1 +
                             I((veg_20 - means$veg_20) / SDs$veg_20) +
                             I((veg_55 - means$veg_55) / SDs$veg_55) +
                             I((veg_85 - means$veg_85) / SDs$veg_85) +
                             I((veg_212 - means$veg_212) / SDs$veg_212) +
                             I((veg_225 - means$veg_225) / SDs$veg_225) +
                             I((log_roughness - means$log_roughness) / SDs$log_roughness) +
                             I((log_d2road  - means$log_d2road) / SDs$log_d2road),
                           family = binomial(link = 'logit'),
                           weights = 10000 ^ as.numeric(detection == 0),
                           data = dat.daily))




#### cross validate ####

# test the model's predictive capacity. First ignoring individual ID, and then accounting for it.
# In every iteration of this loop, used data will be split into a training and testing set,
# the model will be fitted to the training set, and then the resulting predictions
# will be checked against the testing set

validation.results <- array(dim = c(100, 1))
for (i in 1:nrow(validation.results)) { 
  
  # assemble the training data
  dat.sample <- dat.used %>%
    group_by(ID, day) %>%
    slice_sample(n = 1) 
  train.indices <- nrow(dat.sample) %>% sample.int(., size = round(. / 2))
  dat.sample.train <- rbind(dat.sample[train.indices, ], dat.available)
  
  # fit the model
  mymodel <- glm(detection ~ 1 +
                   I((veg_20 - means$veg_20) / SDs$veg_20) +
                   I((veg_55 - means$veg_55) / SDs$veg_55) +
                   I((veg_85 - means$veg_85) / SDs$veg_85) +
                   I((veg_212 - means$veg_212) / SDs$veg_212) +
                   I((veg_225 - means$veg_225) / SDs$veg_225) +
                   I((log_roughness - means$log_roughness) / SDs$log_roughness) +
                   I((log_d2road  - means$log_d2road) / SDs$log_d2road),
                 family = binomial(link = 'logit'),
                 weights = 10000 ^ as.numeric(detection == 0),
                 data = dat.sample.train)
  
  # assemble the test data
  dat.sample.test <- rbind(dat.sample[-train.indices, ], dat.available)
  
  # predict
  dat.sample.test$logRSF <- predict(mymodel, newdata = dat.sample.test)
  
  # bin
  dat.sample.test$logRSF.cat <-  cut(dat.sample.test$logRSF, breaks = 10)
  test.table <- dat.sample.test %>% group_by(logRSF.cat) %>%
    summarise(density = sum(detection == 1) / sum(detection == 0))
  
  # record correlation between bin used-point density and bin log RSF rank
  validation.results[i] <- cor(test.table$density, 1:nrow(test.table),
                               method = "spearman")
  
}

summary(validation.results)
# Min.   :0.9879  
# 1st Qu.:0.9879  
# Median :1.0000  
# Mean   :0.9954  
# 3rd Qu.:1.0000  
# Max.   :1.0000   

sd(validation.results)
# 0.005913112

# cross validate between individuals
validation.results.ind <- array(dim = c(100, 1))
for (i in 1:nrow(validation.results.ind)) {
  
  # assemble the training data
  dat.sample <- dat.used %>%
    group_by(ID, day) %>%
    slice_sample(n = 1) 
  train.IDs.indices <- length(unique.IDs) %>%
    sample.int(., size = round(. / 2))
  dat.sample.train <- dat.sample %>% 
    filter(ID %in% unique.IDs[train.IDs.indices]) %>%
    rbind(dat.available)
  
  # fit the model
  mymodel <- glm(detection ~ 1+
                   I((veg_20 - means$veg_20) / SDs$veg_20) +
                   I((veg_55 - means$veg_55) / SDs$veg_55) +
                   I((veg_85 - means$veg_85) / SDs$veg_85) +
                   I((veg_212 - means$veg_212) / SDs$veg_212) +
                   I((veg_225 - means$veg_225) / SDs$veg_225) +
                   I((log_roughness - means$log_roughness) / SDs$log_roughness) +
                   I((log_d2road  - means$log_d2road) / SDs$log_d2road),
                 family = binomial(link = 'logit'),
                 weights = 10000 ^ as.numeric(detection == 0),
                 data = dat.sample.train)
  
  # assemble the test data
  dat.sample.test <- dat.sample %>% 
    filter(!(ID %in% unique.IDs[train.IDs.indices])) %>%
    rbind(dat.available)
  
  # predict
  dat.sample.test$logRSF <- predict(mymodel, newdata = dat.sample.test)
  
  # bin
  dat.sample.test$logRSF.cat <-  cut(dat.sample.test$logRSF, breaks = 10)
  test.table <- dat.sample.test %>% group_by(logRSF.cat) %>%
    summarise(density = sum(detection == 1) / sum(detection == 0))
  
  # record correlation between bin used-point density and bin log RSF rank
  validation.results.ind[i] <- cor(test.table$density, 1:nrow(test.table),
                                   method = "spearman")
  
}

summary(validation.results.ind)
# Min.   :0.3526  
# 1st Qu.:0.9636  
# Median :0.9879  
# Mean   :0.9496  
# 3rd Qu.:1.0000  
# Max.   :1.0000  

sd(validation.results.ind)
# 0.1303202
