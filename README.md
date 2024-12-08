# Logistic Regression Model Builder for Credit Card Fraud Detection

**Repo Author**: Tom D'Angelo (WPI MS DS)
  - For DS501-F24-F01, Intro to Data Science, Case Study 3.
  - *Instructor*: Prof. Narhara Chari Dingari

**Data Source**: Kaggle
  - *URL*: https://www.kaggle.com/datasets/adityakadiwal/credit-card-fraudulent-transactions
  - *Primary Contributor*: [Aditya Kadiwal](https://www.kaggle.com/adityakadiwal)
  - *Brief Description*: Dataset contains ~90k credit card transactions with binary indicator variable identifying the transaction as "FRAUD" or "LEGIT". Data also contains transaction metadata including: masked domain name of customer's email, state and zip code of customer, time of transaction, amount of transaction, and anonymized data on the customer and transaction.

**EDA**
- See *case_study_3_eda.Rmd* in repo.

**App Description**:
This is a Shiny app that allows users to build and evaluate a logistic regression model on a credit card fraud detection dataset. It leverages the `glmnet` package for regularized logistic regression and includes functionality for target encoding, tuning hyperparameters, adjusting split sizes, and adjusting the prediction threshold.

## Overview of Logistic Regression

**Logistic Regression** is a supervised learning method used for binary classification tasks. In this case, predicting fraudlent vs. legitimate credit card transactions. Instead of directly modeling the outcome (0 or 1), logistic regression models the probability that the outcome is 1 given a set of input features.

### Model Equations

1. **Linear Combination of Features:**
   ```text
   z = w_0 + (w_1 x_1) + (w_2 x_2) + ... + (w_n x_n)
   ```
     - z is the raw score (logit)
     - w_0 is the error term
     - w_1, w_2, ... w_n are the model coefficients
     - x_1, x_2, ... x_n are the feature values for a given observation

3. **Sigmoid (Logistic) Function:**
   ```text
   σ(z) = 1 / (1 + e^(-z))
   ```
   - Logistic regression transforms the raw score ```z``` into a probability using the sigmoid function.
   - The sigmoid function maps ```z``` to a value between 0 and 1, which represents the probability that the observation belongs to the positive class.

4. **Predicted Probability:**
   ```text
   ŷ = P(y=1|x) = σ(z)
   ```
   - The model outputs a predicted probability ```ŷ```.
   - The predicted probability ```ŷ``` is then compared to a threshold to classify each observation as either the positive class (fraud) or the negative class (non-fraud). Observations with a predicted probability below the cutoff threshold are classified as members of the negative class. For instance, if the threshold is set to 0.5 and the predicted probability is 0.45, then the observation will be classified as a non-fraud (legitimate) transaction.
  
### Loss Function
Logistic regression uses a loss function to quantify the differnce between the predicted probability of the positive class and the actual class labels. This app uses the most common loss function for this algorithm, **Log-Loss**.
```text
LogLoss = - (1/N) * Σ [y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i)]
```
- N is the number of observations
- y_i is the true label
- ŷ_i is the predicted probability for the positive class

- One cool thing about Log-Loss is it penalizes incorrect predictions relative to the size of the error. That is, the "more wrong" the model is, the heavier the penalty.

### Optimization
The overall goal of logistic regression is to find the model parameters (w_0, w_1_, ... w_n) that minimize the Log-Loss function. The coefficients are updated iteravely during training using **gradient descent**. The process of gradient descent updates the weights such that the Log-Loss is minimized during training.
```text
∂LogLoss/∂w_j = (1/N) * Σ [(ŷ_i - y_i) * x_{j,i}]
```
- ŷ_i - y_i is the error for observation i (difference between predicted value and actual value)
- x_{j,i} is the value of the jth feature for observation i.
- ∂LogLoss/∂w_j is the partial derivative of the Log-Loss with respect to w_j (the weight for the jth coefficient)

- Using the gradient, the weights are updated iteravely with the following rule:
```text
w_j = w_j - α * ∂LogLoss/∂w_j
```
- Where α is the "learning rate". The learning rate controls how quickly the model weights are updated during training. Smaller values of α allow the model to converge more slowly towards the global minimum of the Log-Loss. Larger values of α allow the model to converge faster, but the danger is the model could overshoot the global minimum.

## Regularization
This app uses regularization parameters, which adds a penalty term to the Log-Loss function which also changes the gradient. There are two basic kinds of regularization, **Ridge** and **Lasso**.

### Ridge Regularization
Ridge regularization adds the squared magnitude of the coefficients as a penalty term to the loss function. Ridge ensures no coefficients are exactly 0, shrinks them significantly.
```text
Penalty_Ridge = λ * Σ (w_j^2)
```
Which adjusts the gradient:
```text
∂TotalLoss/∂w_j = ∂LogLoss/∂w_j + 2 * λ * w_j
```
### Lasso Regularization
Lasso regularization adds a penalty proportional to the absolute value of the coefficients. Some weights are pushed to 0.
```text
Penalty_Lasso = λ * Σ |w_j|
```
Which adjusts the gradient:
```text
∂TotalLoss/∂w_j = ∂LogLoss/∂w_j + λ * sign(w_j)
```
- For both Ridge and Lasso regularization, the ```λ``` controls the strength of the penalty applied to the coefficients. Larger ```λ``` leads to simpler models. Smaller ```λ``` allows the model to fit to the training data more closely.

## Target Encoding
For certain categorical variables (e.g., `DOMAIN`, `STATE`, `ZIPCODE`), target encoding replaces these categories with the mean value of the target variable observed in the training data. This encoding was done for the following reasons: 1) to transform the values of character variables to numeric so they could be ingested by the model, 2) to keep the variance of the categorical data without resorting to standard one-hot encoding, which would have led to too many features and **the curse of dimensionality**. Essentially, the curse of dimensionality refers to cases where there are so many features in a dataset that models cannot capture any patterns or trends in the observations.

## How the App Works

1. **Data Loading and Preprocessing:**  
   The application loads a pre-processed dataset (`CC_FRAUD_cleaned.csv`), processed from the original dataset sourced from Kaggle (link above). The target variable TRN_TYPE_ENCODED is an integer variable with categories 1 (FRAUD) and 0 (LEGIT).

2. **User Inputs:**
   - **Predictors:** Select which variables to include in the model.
   - **Alpha (0 to 1):** Choose between Ridge, Lasso, or a blend.
   - **Lambda:** Choose the strength of regularization.
   - **Train-Test Split:** Set the proportion of data used for training.
   - **Random Seed:** Ensure reproducibility of results.
   - **Prediction Threshold:** Adjust the cutoff for classifying a prediction as fraud vs. non-fraud.

3. **Model Training and Evaluation:**  
   When user clicks "Run Model":
   - The data is split into training and testing sets.
   - Target encoding is applied if required (i.e., at least one high-cardinality nominal variable is selected)
   - A logistic regression model with regularization is trained using `glmnet`.
   - Predictions are made on the test set.
   - Model performance is displayed via confusion matrix, metrics (accuracy, precision, recall, etc.), and an ROC curve.

## Evaluation Metrics

- **Confusion Matrix:** Shows the counts of true positives, true negatives, false positives, and false negatives.
- **Accuracy, Precision, Recall, F1-score:** Evaluate different aspects of the model’s classification performance.
- **ROC Curve and AUC:** The Receiver Operating Characteristic (ROC) curve shows the trade-off between true positive rate and false positive rate at different thresholds. AUC (Area Under the Curve) summarizes this trade-off as a single number. Closer to 1 indicates better discrimination.

## Running the App

1. Clone repo locally:
   ```bash
   git clone https://github.com/dangelotom/DAngelo_DS501_F24_F01_Case_Study_3.git
   ```
2. Run using ```shiny``` package in R Console:
   ```text
   shiny::runGitHub("DAngelo_DS501_F24_F01_Case_Study_3", "dangelotom", subdir = "app.R")
   ```

## Requirements
```text
install.packages(c("shiny", "glmnet", "ggplot2", "dplyr", "pROC", "caret"))
```

## References
1. Burger, S. (2018). *Introduction to Machine Learning with R.* O'Reilly Media.
2. Bruce, P., & Bruce, A. (2017). *Practical Statistics for Data Scientists.* O'Reilly Media.
4. Freedman, D., Pisani, R., and Roger Purves. (2018). *Statistics* (4th ed). W. W. Nortion & Company.
