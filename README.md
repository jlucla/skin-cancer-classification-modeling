# Skin Cancer Classification Modeling

## Overview
This project explores statistical learning approaches for predicting malignant versus benign skin lesions using demographic and lesion-level predictors.

The final modeling workflow used generalized additive models (GAMs) with MICE-based multiple imputation to address missing data and improve predictive robustness across datasets.

## Methods Used
- Multiple Imputation via MICE
- Generalized Additive Models (GAMs)
- Logistic Regression
- Random Forest
- Gradient Boosting
- Classification Modeling
- Feature Engineering

## Workflow
1. Data preprocessing and categorical variable conversion
2. Missing-data imputation using predictive mean matching
3. Model fitting across multiple imputed datasets
4. Probability prediction and pooling across imputations
5. Final malignant/benign classification output

## Repository Contents

| File | Description |
|---|---|
| `skin_cancer_preprocessing_gam_modeling.R` | Data preprocessing, multiple imputation, and final GAM modeling workflow |
| `skincancerslides.pdf` | Presentation summarizing exploratory analysis, modeling approaches, and project findings |

## Tools and Libraries
- R
- mice
- mgcv

## Key Findings
- Generalized additive models (GAMs) achieved the best balance of predictive performance and interpretability
- Flexible nonlinear terms improved modeling performance for predictors such as age, UV exposure, and lesion size
- Simpler statistical learning methods performed competitively relative to more flexible models on this dataset
