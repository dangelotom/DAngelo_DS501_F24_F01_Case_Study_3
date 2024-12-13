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

# Target encoding function
target_encode <- function(df_train, df_test, vars_to_encode, target) {
  df_train_encoded <- df_train
  df_test_encoded <- df_test
  
  for (var in vars_to_encode) {
    if (!var %in% colnames(df_train_encoded)) {
      next
    }
    encodings <- df_train_encoded %>%
      group_by_at(var) %>%
      summarize(Target_Mean = mean(.data[[target]], na.rm = TRUE), .groups = "drop")
    
    df_train_encoded <- df_train_encoded %>%
      left_join(encodings, by = setNames(var, var)) %>%
      mutate("{var}_encoded" := coalesce(Target_Mean, mean(.data[[target]], na.rm = TRUE))) %>%
      select(-Target_Mean, -all_of(var))
    
    df_test_encoded <- df_test_encoded %>%
      left_join(encodings, by = setNames(var, var)) %>%
      mutate("{var}_encoded" := coalesce(Target_Mean, mean(.data[[target]], na.rm = TRUE))) %>%
      select(-Target_Mean, -all_of(var))
  }
  
  list(train = df_train_encoded, test = df_test_encoded)
}

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
    # EDA Tab (HTML)
    tabPanel(
      "EDA",
      tags$iframe(src = "case_study_3_eda.html",
                  width = "100%",
                  height = "800px",
                  frameborder = 0)
    )
  )
)

# Server
server <- function(input, output, session) {
  
  # Reactive data splitting
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
  
  # Model fitting
  model <- eventReactive(input$run_model, {
    req(input$predictors)
    if (length(input$predictors) < 2) {
      stop("Please select at least two predictors.")
    }
    
    split <- split_data()
    train <- split$train
    test <- split$test
    
    # Target encode if necessary
    selected_encoded_vars <- intersect(input$predictors, target_encoded_vars)
    encoded_data <- if (length(selected_encoded_vars) > 0) {
      target_encode(train, test, selected_encoded_vars, target_variable)
    } else {
      list(train = train, test = test)
    }
    train <- encoded_data$train
    test <- encoded_data$test
    
    # Update predictor names for encoded columns
    updated_predictors <- input$predictors
    for (var in selected_encoded_vars) {
      encoded_var <- paste0(var, "_encoded")
      updated_predictors <- gsub(var, encoded_var, updated_predictors, fixed = TRUE)
    }
    
    # Remove rows with missing values in predictors
    train <- train[complete.cases(train[, updated_predictors]), ]
    test <- test[complete.cases(test[, updated_predictors]), ]
    
    encoded_data_reactive$train <- train
    encoded_data_reactive$test <- test
    updated_predictors_reactive(updated_predictors)
    
    y_train <- factor(as.character(train[[target_variable]]), levels = c("0", "1"))
    y_train <- relevel(y_train, ref = "1")
    
    # Use model.matrix for safe numeric conversion
    x_train <- model.matrix(~ . - 1, data = train[, updated_predictors, drop = FALSE])
    
    # Compute weights
    weights <- ifelse(y_train == "1", sum(y_train == "0") / sum(y_train == "1"), 1)
    
    # Fit glmnet
    glmnet(x_train, y_train, family = "binomial", alpha = input$alpha, weights = weights)
  })
  
  # Coefficients
  output$model_coefs <- renderPrint({
    req(model())
    print(coef(model(), s = input$lambda))
  })
  
  # Metrics (Confusion Matrix, Accuracy, Specificity, etc.)
  output$metrics <- renderPrint({
    req(model())
    test <- encoded_data_reactive$test
    updated_predictors <- updated_predictors_reactive()
    
    x_test <- model.matrix(~ . - 1, data = test[, updated_predictors, drop = FALSE])
    preds <- predict(model(), newx = x_test, s = input$lambda, type = "response")
    preds <- as.numeric(preds)
    
    pred_classes <- ifelse(preds > input$threshold, 1, 0)
    confusion <- confusionMatrix(
      factor(pred_classes, levels = c(0,1)),
      factor(test[[target_variable]], levels = c(0,1))
    )
    confusion
  })
  
  # ROC Curve
  output$roc_curve <- renderPlot({
    req(model())
    test <- encoded_data_reactive$test
    updated_predictors <- updated_predictors_reactive()
    
    x_test <- model.matrix(~ . - 1, data = test[, updated_predictors, drop = FALSE])
    preds <- predict(model(), newx = x_test, s = input$lambda, type = "response")
    preds <- as.numeric(preds)
    
    roc_obj <- roc(test[[target_variable]], preds)
    plot(roc_obj, main = "ROC Curve")
  })
  
  # Display App Source Code
  output$appCode <- renderText({
    paste(readLines("app.R"), collapse = "\n")
  })
}

shinyApp(ui, server)