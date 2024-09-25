# Load necessary libraries
install.packages("glmnet")
library(glmnet)

train_data = train
# Set seed for reproducibility
set.seed(123)

# Transform data
# Step 1: Create a named vector for mapping variables to their types
var_type_mapping <- setNames(codebook$Type, codebook$'Variable Name')

# Step 2: Identify continuous and count variables
continuous_and_count_vars <- names(train_data)[names(train_data) %in% names(var_type_mapping[var_type_mapping %in% c("continuous", "count")])]

# Step 3: Standardize continuous and count variables
# Function to standardize a numeric vector (subtract mean and divide by standard deviation)
standardize <- function(x) {
  return((x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE))
}

# Apply standardization only to continuous and count variables
train_data[continuous_and_count_vars] <- lapply(train_data[continuous_and_count_vars], standardize)

# Step 4: The binary variables remain unchanged, and only continuous and count variables are standardized

# Step 1: Outer holdout split (e.g., 80% training, 20% test)
fraction_test <- 0.2
n <- nrow(train_data)
rows_to_keep <- floor((1 - fraction_test) * n)

# Outer training and test split
train_data_outer <- train_data[1:rows_to_keep, ]
test_data_outer <- train_data[(rows_to_keep + 1):n, ]

# Separate predictors and target
x_train_outer <- as.matrix(train_data_outer[, -1])  # Predictors (exclude the target)
y_train_outer <- train_data_outer$target            # Target (classification labels)
x_test_outer <- as.matrix(test_data_outer[, -1])
y_test_outer <- test_data_outer$target

# Step 2: Inner k-fold cross-validation (k = 10)
k <- 10
n_inner <- nrow(train_data_outer)
shuffled_indices <- sample(1:n_inner)  # Shuffle the data
split_size <- floor(n_inner / k)

# Split the data into k equal parts
split_data <- split(train_data_outer[shuffled_indices, ], 
                    ceiling(seq_along(shuffled_indices) / split_size))

# Handle the case where the number of splits exceeds k
if (length(split_data) > k) {
  split_data[[k]] <- rbind(split_data[[k]], split_data[[k+1]])
  split_data <- split_data[-(k+1)]  # Remove extra part
}

# Grid of hyperparameters to search over (alpha and lambda)
alpha_grid <- seq(0, 1, by = 0.1)  # Alpha varies from 0 (Ridge) to 1 (Lasso)
lambda_grid <- seq(0, 10,by = 1)  # Lambda values (regularization strength)

best_hyperparams <- NULL  # Placeholder for the best hyperparameters
best_log_loss <- Inf      # Placeholder for the lowest log-loss

# Function to calculate log-loss
log_loss <- function(y_true, y_pred) {
  return(-mean( y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)))
}

# Hyperparameter selection using inner k-fold CV
for (alpha in alpha_grid) {
  print(alpha)
  for (lambda in lambda_grid) {
    cv_log_losses <- numeric(k)  # Store log-losses for each fold
    
    for (i in 1:k) {
      # Split into training and validation sets for the i-th fold
      validation_fold <- split_data[[i]]
      train_folds <- split_data[-i]
      train_folds <- do.call(rbind, train_folds)
      
      # Prepare training and validation sets
      x_train_inner <- as.matrix(train_folds[, -1])  # Exclude target
      y_train_inner <- train_folds$target
      x_val_inner <- as.matrix(validation_fold[, -1])
      y_val_inner <- validation_fold$target
      
      # Train Elastic Net logistic regression model on the inner training set
      elastic_net_model <- glmnet(x_train_inner, y_train_inner, alpha = alpha, 
                                  lambda = lambda, family = "binomial")
      
      # Make predictions on the validation set (class probabilities)
      val_predictions <- predict(elastic_net_model, s = lambda, newx = x_val_inner, type = "response")
      
      # Calculate log-loss on the validation set
      cv_log_losses[i] <- log_loss(y_val_inner, val_predictions)
    }
    
    # Average cross-validation log-loss for this combination of alpha and lambda
    avg_cv_log_loss <- mean(cv_log_losses)
    
    # Update the best hyperparameters if this combination has the lowest log-loss
    if (avg_cv_log_loss < best_log_loss) {
      best_log_loss <- avg_cv_log_loss
      best_hyperparams <- list(alpha = alpha, lambda = lambda)
    }
  }
}

# Step 3: Train final model on the full outer training set using the best hyperparameters
final_elastic_net_model <- glmnet(x_train_outer, y_train_outer, alpha = best_hyperparams$alpha, 
                                  lambda = best_hyperparams$lambda, family = "binomial")

# Step 4: Evaluate the final model on the outer test set
test_predictions <- predict(final_elastic_net_model, s = best_hyperparams$lambda, 
                            newx = x_test_outer, type = "response")

# Calculate log-loss on the test set
test_log_loss <- log_loss(y_test_outer, test_predictions)

# Output the results
print(paste("Best alpha:", best_hyperparams$alpha))
print(paste("Best lambda:", best_hyperparams$lambda))
print(paste("Final test log-loss:", test_log_loss))
