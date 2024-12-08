library(shiny)
library(glmnet)
library(ggplot2)
library(dplyr)
library(pROC)
library(caret)
library(rmarkdown)

# Load cleaned data file
data <- read.csv("./data/CC_FRAUD_cleaned.csv")

# Define target variable and variables to target encode
target_variable <- "TRN_TYPE_ENCODED"
target_encoded_vars <- c("DOMAIN", "STATE", "ZIPCODE")

# Feature variables
exclude_vars <- c(target_variable, "TRN_TYPE")
feature_vars <- setdiff(names(data), exclude_vars)

# UI
ui <- fluidPage(
  titlePanel("Logistic Regression Model Builder for Credit Card Fraud Detection Dataset"),
  
  tabsetPanel(
    # Main App Tab
    tabPanel(
      "Model Builder",
      sidebarLayout(
        sidebarPanel(
          checkboxGroupInput("predictors", "Select Predictor Variables", choices = feature_vars),
          sliderInput("alpha", "Regularization Alpha (0=Ridge, 1=Lasso)", min = 0, max = 1, value = 0.5),
          numericInput("seed", "Random Seed (for reproducibility)", value = 42, min = 1),
          sliderInput("train_size", "Train-Test Split (Train Proportion)", min = 0.5, max = 0.9, value = 0.8, step = 0.05),
          sliderInput("lambda", "Regularization Lambda", min = 0.01, max = 10, value = 1, step = 0.1),
          sliderInput("threshold", "Prediction Threshold", min = 0.1, max = 0.9, value = 0.5, step = 0.05),
          actionButton("run_model", "Run Model")
        ),
        
        mainPanel(
          verbatimTextOutput("model_coefs"),
          verbatimTextOutput("metrics"),
          plotOutput("roc_curve")
        )
      )
    ),
    # Source Code Tab
    tabPanel(
      "App Source Code",
      h4("Source Code for app.R"),
      verbatimTextOutput("appCode")
    ),
    # EDA Tab (R Markdown)
    tabPanel(
      "EDA",
      includeMarkdown("case_study_3_eda.Rmd")
    )
  )
)

# Server
server <- function(input, output, session) {
  
  # Splits
  split_data <- reactive({
    req(input$predictors)
    set.seed(input$seed)
    train_indices <- createDataPartition(data[[target_variable]], p = input$train_size, list = FALSE)
    list(
      train = data[train_indices, ],
      test = data[-train_indices, ]
    )
  })
  
  encoded_data_reactive <- reactiveValues(train = NULL, test = NULL)
  updated_predictors_reactive <- reactiveVal(NULL)
  
  # Fit model
  model <- eventReactive(input$run_model, {
    req(input$predictors)
    if (length(input$predictors) < 2) {
      stop("Please select at least two predictors.")
    }
    
    split <- split_data()
    train <- split$train
    test <- split$test
    
    y_train <- factor(train[[target_variable]], levels = c("0", "1"))
    y_train <- relevel(y_train, ref = "1")
    
    x_train <- as.matrix(as.data.frame(lapply(train[, input$predictors, drop = FALSE], as.numeric)))
    weights <- ifelse(y_train == "1", sum(y_train == "0") / sum(y_train == "1"), 1)
    
    glmnet(x_train, y_train, family = "binomial", alpha = input$alpha, weights = weights)
  })
  
  # Outputs
  output$model_coefs <- renderPrint({
    req(model())
    print(coef(model(), s = input$lambda))
  })
  
  output$metrics <- renderPrint({
    req(model())
    test <- split_data()$test
    preds <- predict(model(), newx = as.matrix(test[, input$predictors]), s = input$lambda, type = "response")
    pred_classes <- ifelse(preds > input$threshold, 1, 0)
    confusionMatrix(factor(pred_classes, levels = c(0,1)), factor(test[[target_variable]], levels = c(0,1)))
  })
  
  output$roc_curve <- renderPlot({
    req(model())
    test <- split_data()$test
    preds <- predict(model(), newx = as.matrix(test[, input$predictors]), s = input$lambda, type = "response")
    plot(roc(test[[target_variable]], preds), main = "ROC Curve")
  })
  
  output$appCode <- renderText({
    paste(readLines("app.R"), collapse = "\n")
  })
}

shinyApp(ui, server)