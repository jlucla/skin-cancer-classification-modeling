# =========================================================
# SKIN CANCER CLASSIFICATION MODELING
# =========================================================
# Predicting malignant vs. benign skin lesions
# using generalized additive models (GAMs)
# with MICE-based multiple imputation.
# =========================================================

# =========================================================
# DATA PREPROCESSING
# =========================================================
# -----------------------------------
# Load Libraries
# -----------------------------------

library(mice)
library(mgcv)

# -----------------------------------
# Load Training and Test Data
# -----------------------------------

train <- read.csv("SkinCancerTrain.csv", row.names = 1)
X_test <- read.csv("SkinCancerTestNoY.csv", row.names = 1)

# -----------------------------------
# Convert Binary Indicators
# -----------------------------------

# Classify outdoor_job as categorical
train$outdoor_job <- as.character(train$outdoor_job) 
X_test$outdoor_job <- as.character(X_test$outdoor_job)

train$outdoor_job[train$outdoor_job == "0"] <- "No"
train$outdoor_job[train$outdoor_job == "1"] <- "Yes"
X_test$outdoor_job[X_test$outdoor_job == "0"] <- "No"
X_test$outdoor_job[X_test$outdoor_job == "1"] <- "Yes"

# Convert categorical type variables to factor
train[sapply(train, is.character)] <- 
  lapply(train[sapply(train, is.character)], as.factor)

X_test[sapply(X_test, is.character)] <- 
  lapply(X_test[sapply(X_test, is.character)], as.factor)

# -----------------------------------
# Multiple Imputation
# -----------------------------------

# Impute using mice
imp <- mice(train, 
            m = 5,          
            method = "pmm", 
            maxit = 3,     
            seed = 123)

imp_test <- mice(X_test, 
                 m = 5,          
                 method = "pmm", 
                 maxit = 3,     
                 seed = 123)

# Export imputed datasets
for (i in 1:imp$m) {
  imputed_train <- complete(imp, i)
  
  write.csv(
    imputed_train,
    file = paste0("train_mice_imputed_", i, "_of_5.csv"),
    row.names = TRUE
  )
}

for (i in 1:imp_test$m) {
  imputed_test <- complete(imp_test, i)
  
  write.csv(
    imputed_test,
    file = paste0("X_test_mice_imputed_", i, "_of_5.csv"),
    row.names = TRUE
  )
}

# =========================================================
# WINNING MODEL IMPLEMENTATION: GAM WITH 14 PREDICTORS
# =========================================================

# -----------------------------------
# Load Cleaned and Imputed Datasets
# -----------------------------------

train_mice5 <- paste0("train_mice_imputed_", 1:5, "_of_5.csv")
train_mice5 <- lapply(train_mice5, read.csv)

X_test_mice5 <- paste0("X_test_mice_imputed_", 1:5, "_of_5.csv")
X_test_mice5 <- lapply(X_test_mice5, read.csv)

# -----------------------------------
# GAM Model Fitting
# -----------------------------------

# Fit GAM models for each imputed dataset
gam14.model <- lapply(train_mice5, function(d) {
  gam(as.factor(Cancer) ~ 
        s(age) + 
        skin_tone + 
        s(avg_daily_uv) + 
        sunscreen_freq + 
        hat_use + 
        clothing_protection + 
        tanning_bed_use + 
        outdoor_job + 
        family_history + 
        immunosuppressed + 
        s(lesion_size_mm) + 
        number_of_lesions + 
        skin_photosensitivity + 
        sunburns_last_year, 
      
      data = d, 
      family = binomial
      )
})

# -----------------------------------
# Prediction and Output
# -----------------------------------

# Predict probabilities on the test sets
gam14.model.preds <- mapply(function(model, test) {
  predict(model, newdata = test, type = "response")  # probabilities
}, gam14.model, X_test_mice5, SIMPLIFY = FALSE)

# Average predictions across imputations
gam14.model.pred.probs <- Reduce("+", gam14.model.preds) / length(gam14.model.preds)

# Convert probabilities to classes
gam14.model.pred.class <- ifelse(gam14.model.pred.probs < 0.5, "Benign", "Malignant")

# Output dataframe
gam14.model.test.mat <- data.frame(
  ID = 1:20000,
  Cancer = gam14.model.pred.class
)

# Export test predictions
write.csv(gam14.model.test.mat, "skincancerpred_mice_gam14_model.csv", row.names = FALSE)