############################################################
# DoorDash Delivery Duration Prediction
#
# Author: Divraj Singh
# Tools: R, DuckDB, glmnet, XGBoost
#
# Pipeline:
# 1. Load data into DuckDB
# 2. Clean timestamps + create target
# 3. Feature engineering (SQL + R)
# 4. Train Linear Regression
# 5. Train Elastic Net
# 6. Train XGBoost
# 7. Evaluate performance
############################################################


# DBI provides dbConnect(), dbExecute(), dbGetQuery(), etc.
library(DBI)
# duckdb provides duckdb::duckdb() which is the DuckDB driver/engine
library(duckdb)
library(dplyr)
library(dbplyr)


# Here we create a database file named "doordash.duckdb" in our project folder.
db_file <- "doordash.duckdb"


############################
# Connect to DuckDB
############################

# dbConnect() opens a connection to the database.
# If the file doesn't exist yet, DuckDB will create it automatically.
con <- dbConnect(duckdb::duckdb(), dbdir = db_file)


############################
# Quick sanity checks 
############################

# Run a simple SQL query.
# If this returns 1, your database connection is working properly.
dbGetQuery(con, "SELECT 1 AS connection_test;")


# List tables currently in the database.
# Right after setup, this will likely be an empty character vector.
dbListTables(con)


# Making Data more readable
options(dplyr.width = Inf)

############################
# Point to your CSV file 
############################

csv_path <- "data/historical_data.csv"

# Checking if file at this location.
file.exists(csv_path)


############################
# Create a "raw" table in DuckDB from the CSV
############################

# We use read_csv_auto() which lets DuckDB automatically infer column types.
create_raw_table_sql <- paste0("
  CREATE OR REPLACE TABLE deliveries_raw AS
  SELECT * FROM read_csv_auto('", csv_path, "');
")

# Execute the SQL to create the table
dbExecute(con, create_raw_table_sql)

# List of Tables
dbListTables(con)

# Count how many rows were loaded.
dbGetQuery(con, "SELECT COUNT(*) AS n_rows FROM deliveries_raw;")

# More information about our data and its corresponding value types
dbGetQuery(con, "DESCRIBE deliveries_raw;")

# Preview a few rows
dbGetQuery(con, "SELECT * FROM deliveries_raw LIMIT 5;")


# We want to see if created_at or actual_delivery_time have missing values,
# because that will affect target creation later.
dbGetQuery(con, "
  SELECT
    SUM(CASE WHEN created_at IS NULL THEN 1 ELSE 0 END) AS missing_created_at,
    SUM(CASE WHEN actual_delivery_time IS NULL THEN 1 ELSE 0 END) AS missing_actual_delivery_time
  FROM deliveries_raw;
")


############################################################
# Create a cleaned table + target label in DuckDB
############################################################

# Notes on choices:
# - created_at is already TIMESTAMP, so we can use it directly.
# - actual_delivery_time is VARCHAR, so we parse it into a TIMESTAMP.
#   We use try_strptime() so rows with bad formats become NULL (not an error).
# - We cast "dashers/orders/driving_duration" columns from VARCHAR -> DOUBLE.
#   DOUBLE is safe because these fields may contain missing values or decimals.
# - epoch(timestamp) gives seconds since 1970; difference = duration in seconds.

dbExecute(con, "
  CREATE OR REPLACE TABLE deliveries_clean AS
  WITH typed AS (
    SELECT
      -- Keep identifiers / categorical fields
      market_id,
      store_id,
      store_primary_category,
      order_protocol,

      -- Timestamps
      created_at AS created_ts,

      -- Parse actual_delivery_time from VARCHAR -> TIMESTAMP
      -- This format usually matches 'YYYY-MM-DD HH:MM:SS'
      try_strptime(actual_delivery_time, '%Y-%m-%d %H:%M:%S') AS delivered_ts,

      -- Order features (already numeric in your schema)
      total_items,
      subtotal,
      num_distinct_items,
      min_item_price,
      max_item_price,

      -- Market features (currently VARCHAR -> cast to DOUBLE)
      try_cast(total_onshift_dashers AS DOUBLE)      AS total_onshift_dashers,
      try_cast(total_busy_dashers AS DOUBLE)         AS total_busy_dashers,
      try_cast(total_outstanding_orders AS DOUBLE)   AS total_outstanding_orders,

      -- Model prediction features
      estimated_order_place_duration,
      try_cast(estimated_store_to_consumer_driving_duration AS DOUBLE)
        AS estimated_store_to_consumer_driving_duration

    FROM deliveries_raw
  )
  SELECT
    *,
    -- Target label: delivery duration in seconds
    epoch(delivered_ts) - epoch(created_ts) AS delivery_duration_seconds
  FROM typed
  WHERE
    created_ts IS NOT NULL
    AND delivered_ts IS NOT NULL
    -- Filter out non-positive durations (bad data)
    AND (epoch(delivered_ts) - epoch(created_ts)) > 0;
")

# Row count after filtering bad/missing timestamp rows
dbGetQuery(con, "SELECT COUNT(*) AS n_rows_clean FROM deliveries_clean;")

# Quick distribution check for the target
dbGetQuery(con, "
  SELECT
    MIN(delivery_duration_seconds) AS min_sec,
    AVG(delivery_duration_seconds) AS avg_sec,
    MAX(delivery_duration_seconds) AS max_sec
  FROM deliveries_clean;
")

# Optional: to see what % of rows were dropped due to timestamp parsing / invalid durations
dbGetQuery(con, "
  SELECT
    (SELECT COUNT(*) FROM deliveries_raw)  AS raw_rows,
    (SELECT COUNT(*) FROM deliveries_clean) AS clean_rows,
    (SELECT COUNT(*) FROM deliveries_raw) - (SELECT COUNT(*) FROM deliveries_clean) AS dropped_rows;
")

# Preview a few rows to make sure timestamps + target look correct
dbGetQuery(con, "
  SELECT
    created_ts,
    delivered_ts,
    delivery_duration_seconds
  FROM deliveries_clean
  LIMIT 5;
")


############################################################
# Investigate extreme delivery durations
############################################################

# Look at top 10 largest durations
dbGetQuery(con, "
  SELECT 
    delivery_duration_seconds,
    created_ts,
    delivered_ts
  FROM deliveries_clean
  ORDER BY delivery_duration_seconds DESC
  LIMIT 10;
")

# Examine distribution percentiles of delivery duration
dbGetQuery(con, "
  SELECT
    percentile_cont(0.50) WITHIN GROUP (ORDER BY delivery_duration_seconds) AS p50,
    percentile_cont(0.90) WITHIN GROUP (ORDER BY delivery_duration_seconds) AS p90,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY delivery_duration_seconds) AS p95,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY delivery_duration_seconds) AS p99,
    percentile_cont(0.999) WITHIN GROUP (ORDER BY delivery_duration_seconds) AS p999
  FROM deliveries_clean;
")

# Remove extreme outliers for modeling
#
# We remove deliveries longer than 8000 seconds (~2.2 hours)
# This threshold is justified by percentile analysis

dbExecute(con, "
  CREATE OR REPLACE TABLE deliveries_model AS
  SELECT *
  FROM deliveries_clean
  WHERE delivery_duration_seconds <= 8000;
")

# Check how many rows remain
dbGetQuery(con, "
  SELECT
    (SELECT COUNT(*) FROM deliveries_clean) AS before_trim,
    (SELECT COUNT(*) FROM deliveries_model) AS after_trim,
    (SELECT COUNT(*) FROM deliveries_clean) -
    (SELECT COUNT(*) FROM deliveries_model) AS removed_rows;
")

# Confirm new max duration
dbGetQuery(con, "
  SELECT
    MAX(delivery_duration_seconds) AS new_max_sec
  FROM deliveries_model;
")

############################################################
# Step 5: Create Feature Engineering Table in SQL
############################################################

# We build additional predictive features directly in DuckDB.
# This keeps the pipeline structured and production-like.

dbExecute(con, "
  CREATE OR REPLACE TABLE deliveries_features AS
  SELECT
    -- Keep identifiers
    market_id,
    store_id,
    store_primary_category,
    order_protocol,

    -- Timestamps
    created_ts,
    delivered_ts,

    -- Target variable
    delivery_duration_seconds,

    -- ================================
    -- 1) TIME-BASED FEATURES
    -- ================================

    EXTRACT('hour' FROM created_ts) AS order_hour,

    EXTRACT('dow' FROM created_ts) AS order_dow,

    CASE 
      WHEN EXTRACT('dow' FROM created_ts) IN (0,6)
      THEN 1 ELSE 0
    END AS is_weekend,

    -- ================================
    -- 2) MARKET PRESSURE FEATURES
    -- ================================

    total_onshift_dashers,
    total_busy_dashers,
    total_outstanding_orders,

    CASE 
      WHEN total_onshift_dashers > 0
      THEN total_busy_dashers / total_onshift_dashers
      ELSE NULL
    END AS busy_ratio,

    CASE
      WHEN total_onshift_dashers > 0
      THEN total_outstanding_orders / total_onshift_dashers
      ELSE NULL
    END AS orders_per_dasher,

    -- ================================
    -- 3) ORDER / BASKET FEATURES
    -- ================================

    total_items,
    subtotal,
    num_distinct_items,
    min_item_price,
    max_item_price,

    CASE 
      WHEN total_items > 0
      THEN subtotal * 1.0 / total_items
      ELSE NULL
    END AS avg_item_price,

    (max_item_price - min_item_price) AS item_price_range,

    CASE
      WHEN total_items > 0
      THEN num_distinct_items * 1.0 / total_items
      ELSE NULL
    END AS distinct_item_ratio,

    -- ================================
    -- 4) MODEL PREDICTION FEATURES
    -- ================================

    estimated_order_place_duration,
    estimated_store_to_consumer_driving_duration

  FROM deliveries_model;
")

# Preview a few engineered columns
dbGetQuery(con, "
  SELECT
    delivery_duration_seconds,
    order_hour,
    is_weekend,
    busy_ratio,
    avg_item_price
  FROM deliveries_features
  LIMIT 5;
")

############################################################
# Load Feature Table into R for Modeling
############################################################

# Use dbplyr to reference the table
deliveries_tbl <- tbl(con, "deliveries_features")

# Collect into R memory as a dataframe
deliveries_df <- deliveries_tbl %>% collect()

# Check dimensions
dim(deliveries_df)

# Structure check
str(deliveries_df)

# Check missing values per column
colSums(is.na(deliveries_df))

############################################################
#. Clean Dataset for Modeling
############################################################

# Start from trimmed SQL dataset (no extreme outliers)
temp_df <- deliveries_df

# Sort by time
temp_df <- temp_df[order(temp_df$created_ts), ]

# Spliting the data into training and testing
# 80% data in training and 20% for testing
split_index <- floor(0.8 * nrow(temp_df))

train_df <- temp_df[1:split_index, ]
test_df  <- temp_df[(split_index + 1):nrow(temp_df), ]

# Cap unrealistic ratios 

# busy_ratio = busy_dashers / total_dashers (doesn't make sense to be more than 1)
train_df$busy_ratio[train_df$busy_ratio > 1] <- 1

# values above 5 are rare and likely noise/outliers
train_df$orders_per_dasher[train_df$orders_per_dasher > 5] <- 5

# Also replacing Na values with the respective col medians 
# Compute medians from training data ONLY
numeric_cols <- sapply(train_df, is.numeric)

train_medians <- c()
# create vector of col medians where we have na values.
for (col in names(train_df)[numeric_cols]) {
  train_medians[col] <- median(train_df[[col]], na.rm = TRUE)
}

# Impute training data
for (col in names(train_medians)) {
  train_df[[col]][is.na(train_df[[col]])] <- train_medians[col]
}

# Applying training-based preprocessing to TEST data

# Cap ratios
test_df$busy_ratio[test_df$busy_ratio > 1] <- 1
test_df$orders_per_dasher[test_df$orders_per_dasher > 5] <- 5

# Impute using TRAIN medians 
for (col in names(train_medians)) {
  test_df[[col]][is.na(test_df[[col]])] <- train_medians[col]
}

############################################################
# Final Modeling Preparation
############################################################

# Remove timestamps (not used directly in regression)
train_df$created_ts <- NULL
train_df$delivered_ts <- NULL
test_df$created_ts <- NULL
test_df$delivered_ts <- NULL

# Convert categorical variables to factors
categorical_cols <- c("market_id",
                      "store_primary_category",
                      "order_protocol")

for (col in categorical_cols) {
  train_df[[col]] <- as.factor(train_df[[col]])
  test_df[[col]]  <- as.factor(test_df[[col]])
}

# Final check
str(train_df)

############################################################
# Train Baseline Linear Regression
############################################################

lm_model <- lm(delivery_duration_seconds ~ ., data = train_df)

summary(lm_model)

############################################################
# Model Evaluation
############################################################

# Predict on test set
predictions <- predict(lm_model, newdata = test_df)

# MAE (mean absolute error)
mae <- mean(abs(predictions - test_df$delivery_duration_seconds))

# RMSE (root mean squared error)
rmse <- sqrt(mean((predictions - test_df$delivery_duration_seconds)^2))

mae
rmse

############################################################
# My intuition on the results 
#   - R² ≈ 0.287
#   - MAE ≈ 676 seconds (~11.3 minutes)
#   - RMSE ≈ 903 seconds (~15 minutes)
#   - looking at the t values tells us that delivery time is mainly driven by
#   - driving time, marketplace congestion, time of the day, order size.
#   - Also the store_primary_category introduces around 70+ dummy variables 
#   - increasing variance, model complexity and causes instability
# 
#   - Test Model B after removing the tore_primary_category
############################################################

# Reduced Model (remove store_primary_category)
train_reduced <- train_df
test_reduced  <- test_df

# Remove category column
train_reduced$store_primary_category <- NULL
test_reduced$store_primary_category  <- NULL

# Train model
lm_reduced <- lm(delivery_duration_seconds ~ ., data = train_reduced)

# Predict
pred_reduced <- predict(lm_reduced, newdata = test_reduced)

# Evaluate
mae_reduced  <- mean(abs(pred_reduced - test_reduced$delivery_duration_seconds))
rmse_reduced <- sqrt(mean((pred_reduced - test_reduced$delivery_duration_seconds)^2))

mae_reduced
rmse_reduced

############################################################
# Dropping store_primary_category had minimal impact on performance,
# suggesting limited predictive contribution.
# we proceed with Elastic Net regularization to control overfitting and
# automatically select the most relevant features.
############################################################

# load glemnet 
library(glmnet)

# Align factor levels between train and test
categorical_cols <- c("market_id",
                      "store_primary_category",
                      "order_protocol")

for (col in categorical_cols) {
  test_df[[col]] <- factor(test_df[[col]],
                           levels = levels(train_df[[col]]))
}

# Prepare data for glmnet

# Response variable
y_train <- train_df$delivery_duration_seconds
y_test  <- test_df$delivery_duration_seconds

# Remove response from predictors
x_train <- model.matrix(delivery_duration_seconds ~ ., train_df)[, -1]
x_test  <- model.matrix(delivery_duration_seconds ~ ., test_df)[, -1]
dim(x_train)
dim(x_test)

############################################################
# Elastic Net (with cross-validation)
############################################################

set.seed(123)

cv_model <- cv.glmnet(
  x_train,
  y_train,
  alpha = 0.5   # 0 = Ridge, 1 = Lasso, 0.5 = Elastic Net
)

# Best lambda chosen by CV
best_lambda <- cv_model$lambda.min
best_lambda


############################################################
# Evaluate Elastic Net
############################################################

pred_enet <- predict(cv_model, s = "lambda.min", newx = x_test)

mae_enet  <- mean(abs(pred_enet - y_test))
rmse_enet <- sqrt(mean((pred_enet - y_test)^2))

mae_enet
rmse_enet

sum(coef(cv_model, s = "lambda.min") != 0)

############################################################
# The linear model was already fairly stable.
# Multicollinearity wasn’t causing major overfitting.
# Feature selection barely happened, only droped like two features and kept 104


## So, now we are going to try Gradient Boosting 
# What this does is built a decision trees that capture non linear behaviour,
# thresholds, interactions and complex structures
# Instead of building one tree,
# Boosting builds many small trees sequentially.

# Gradient Boosting (we move in the direction that reduces error).
############################################################

# Install and load xgboost
library(xgboost)

# Create DMatrix objects
# It stores feature matrix, labels and seeps up the training

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test,  label = y_test)

# A watchlist lets xgboost report performance on both train + test during training
watchlist <- list(train = dtrain, test = dtest)


# Define XGBoost parameters
#
# Key ideas:
# - objective = "reg:squarederror" for regression
# - eta = learning rate (smaller = slower, often better generalization)
# - max_depth controls tree complexity
# - subsample and colsample_bytree help prevent overfitting

params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  
  # We'll evaluate MAE during training because it's easy to interpret (seconds)
  eval_metric = "mae",
  
  # Controls complexity / learning
  eta = 0.05,          # learning rate
  max_depth = 6,       # tree depth
  min_child_weight = 10,  # helps avoid overly specific splits
  subsample = 0.8,     # use 80% of rows per tree
  colsample_bytree = 0.8  # use 80% of columns per tree
)

# Train the model with early stopping
#
# We train up to nrounds, but stop early if test MAE
# doesn't improve for a while.

set.seed(123)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 3000,                 # big upper limit; early stopping will stop sooner
  watchlist = watchlist,
  early_stopping_rounds = 50,     # stop if no improvement for 50 rounds
  verbose = 1                     # prints training progress
)

# Best number of boosting rounds chosen automatically
xgb_model$best_iteration

#  Predict on test set + evaluate
pred_xgb <- predict(xgb_model, newdata = dtest)

# MAE in seconds
mae_xgb <- mean(abs(pred_xgb - y_test))

# RMSE in seconds
rmse_xgb <- sqrt(mean((pred_xgb - y_test)^2))

mae_xgb
rmse_xgb

# Feature importance
importance <- xgb.importance(model = xgb_model)

# View top 15 important features
head(importance, 15)

############################################################
# Model Comparison and Interpretation
#
# Test Performance:
# Linear Regression   → MAE ≈ 675.6 | RMSE ≈ 902.7
# Elastic Net         → MAE ≈ 675.6 | RMSE ≈ 902.8
# XGBoost             → MAE ≈ 628.4 | RMSE ≈ 853.6
#
# XGBoost improved MAE by ~47 seconds (~7%) relative to the
# linear baseline. While not dramatic, this confirms the
# presence of nonlinear structure in the data.
#
# Interpretation:
# - Linear regression captured strong global signals (e.g.,
#   driving duration, congestion metrics).
# - Boosting reduced bias by modeling nonlinear interactions
#   and threshold effects (e.g., congestion tipping points).
# - Improvement was moderate because feature engineering had
#   already captured much of the linear signal.
#
# Top XGBoost Features:
# - orders_per_dasher (strong congestion signal)
# - estimated_store_to_consumer_driving_duration
# - order_hour
# - subtotal
# - store_id (store-specific prep effects; potential
#   generalization risk)
#
# Best iteration ≈ 1370 trees (eta = 0.05).
# Small learning rate → more trees required.
# Early stopping helps control overfitting by limiting
# excessive model complexity.
############################################################
