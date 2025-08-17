# ------------------------------------------
# 1. LOAD & CLEAN
# ------------------------------------------
library(lubridate)
library(recipes)
library(skimr)

data <- read_csv("airbnb_ratings_new.csv")
str(data)
#df <- df %>% 
#  janitor::clean_names() %>%                    # nice snake_case
#  mutate(price = parse_number(price),
#         last_review_date = mdy(last_review_date),
#         is_superhost = if_else(host_is_superhost == "t", 1, 0))
str(data)
y_cls <- "is_superhost"
y_reg <- "price"
# 3.corrected: Create binary Superhost column from logical TRUE/FALSE
data$host_is_superhost <- as.integer(data$`Host Is Superhost` == "TRUE")
# 4. Clean 'Price' column (capital P)
data$Price <- as.numeric(gsub("[$,]", "", data$Price))

# ------------------------------------------
# 2. DESCRIPTIVE STATS
# ------------------------------------------
#num_desc <- df %>%
#  select_if(is.numeric) %>%
#  summary()

#cat_desc <- df %>% 
#  select_if(is.character) %>% 
#  map_df(~ tibble(unique = n_distinct(.x),
#                 most_common = names(sort(table(.x), TRUE)[1]),
#                  missing_pct = mean(is.na(.x))*100), .id = "variable")
#summary(num_desc)
#write_csv(cat_desc, "cat_desc.csv")

# 8. FEATURE ENGINEERING

# A. reviews_per_day_available
data$reviews_per_day_available <- with(data, `Reviews per month` / (`Availability 365` / 30 + 1))

# B. total_rooms
data$total_rooms <- with(data, Bedrooms + Bathrooms)

# C. accommodates_per_room
data$accommodates_per_room <- with(data, Accommodates / (total_rooms + 1))

# D. avg_review_score
review_vars <- c("Review Scores Rating",
                 "Review Scores Cleanliness",
                 "Review Scores Checkin",
                 "Review Scores Communication",
                 "Review Scores Location",
                 "Review Scores Value")

data[review_vars] <- lapply(data[review_vars], function(x) as.numeric(as.character(x)))
data$avg_review_score <- rowMeans(data[, review_vars], na.rm = TRUE)

# E. is_experienced_host
data$is_experienced_host <- ifelse(data$`Host total listings count` > 5, 1, 0)

# F. value_per_dollar
data$value_per_dollar <- data$`Review Scores Value` / (data$Price + 1)

# OPTIONAL: summary
engineered_vars <- c("reviews_per_day_available",
                     "total_rooms",
                     "accommodates_per_room",
                     "avg_review_score",
                     "is_experienced_host",
                     "value_per_dollar")

summary(data[engineered_vars])
str(data)
# --------------------------
# 1. DATA PREPARATION
# --------------------------
library(tidyverse)
library(caret)
library(randomForest)
library(gbm)
library(e1071)
library(ggplot2)

set.seed(123)
train_idx <- sample(seq_len(nrow(data)), size = 0.8 * nrow(data))

data_train <- data[train_idx, ]
data_test  <- data[-train_idx, ]
library(dplyr)
train_numeric <- data_train %>% select_if(is.numeric)
test_numeric  <- data_test %>% select_if(is.numeric)
data_train <- data_train[!is.na(data_train$host_is_superhost), ]
sum(is.na(data_train$host_is_superhost))
colSums(is.na(data_train))
data_train <- na.omit(data_train)
data_train$host_is_superhost <- as.factor(data_train$host_is_superhost)
# --------------------------
# FEATURE SELECTION: Random Forest
# --------------------------
library(janitor)

data_train <- data_train %>% clean_names()
rf_model <- randomForest(host_is_superhost ~ ., data = data_train, importance = TRUE)
varImpPlot(rf_model, main = "Random Forest Variable Importance")

# FEATURE SELECTION: Stepwise (Backward)
null_model <- glm(host_is_superhost ~ 1, data = data_train, family = binomial)
full_model <- glm(host_is_superhost ~ ., data = data_train, family = binomial)
step_model <- step(full_model, direction = "backward", trace = 0)

# Plot coefficients from stepwise
coef_df <- as.data.frame(summary(step_model)$coefficients)
coef_df$Variable <- rownames(coef_df)
ggplot(coef_df[-1, ], aes(x = reorder(Variable, Estimate), y = Estimate)) +
  geom_col() + coord_flip() +
  labs(title = "Stepwise Selected Variables (Coefficients)", y = "Estimate", x = "")

# --------------------------
# 2. KNN MODEL
# --------------------------
knn_fit <- train(
  host_is_superhost ~ ., 
  data = train_scaled, 
  method = "knn",
  tuneLength = 10,
  trControl = trainControl(method = "cv", number = 5)
)

knn_pred <- predict(knn_fit, newdata = test_scaled)

# Confusion matrix
conf_knn <- confusionMatrix(knn_pred, test_scaled$host_is_superhost, positive = "1")
print(conf_knn)

# Plot confusion matrix
cm_knn_df <- as.data.frame(conf_knn$table)
ggplot(cm_knn_df, aes(Prediction, Reference, fill = Freq)) +
  geom_tile() + geom_text(aes(label = Freq), color = "white", size = 5) +
  labs(title = "KNN Confusion Matrix")

# Metrics
knn_metrics <- conf_knn$byClass[c("Precision", "Recall", "F1")]
cat("KNN Accuracy:", conf_knn$overall["Accuracy"], "\n")
print(knn_metrics)

# --------------------------
# 3. RANDOM FOREST MODEL
# --------------------------
rf_fit <- randomForest(host_is_superhost ~ ., data = train_scaled, importance = TRUE)
rf_pred <- predict(rf_fit, newdata = test_scaled)

# Confusion matrix
conf_rf <- confusionMatrix(rf_pred, test_scaled$host_is_superhost, positive = "1")
print(conf_rf)

# Plot confusion matrix
cm_rf_df <- as.data.frame(conf_rf$table)
ggplot(cm_rf_df, aes(Prediction, Reference, fill = Freq)) +
  geom_tile() + geom_text(aes(label = Freq), color = "white", size = 5) +
  labs(title = "Random Forest Confusion Matrix")

# Metrics
rf_metrics <- conf_rf$byClass[c("Precision", "Recall", "F1")]
cat("Random Forest Accuracy:", conf_rf$overall["Accuracy"], "\n")
print(rf_metrics)

# Plot variable importance
varImpPlot(rf_fit, main = "Random Forest Variable Importance")

# --------------------------
# 4. GRADIENT BOOSTED TREES
# --------------------------
gbm_fit <- gbm(
  host_is_superhost ~ ., 
  data = train_scaled, 
  distribution = "bernoulli",
  n.trees = 1000, 
  interaction.depth = 3,
  shrinkage = 0.01,
  cv.folds = 5,
  verbose = FALSE
)

best_iter <- gbm.perf(gbm_fit, method = "cv")
cat("Best number of trees (GBM):", best_iter, "\n")

gbm_pred_probs <- predict(gbm_fit, newdata = test_scaled, n.trees = best_iter, type = "response")
gbm_pred <- ifelse(gbm_pred_probs > 0.5, "1", "0") %>% factor(levels = c("0", "1"))

# Confusion matrix
conf_gbm <- confusionMatrix(gbm_pred, test_scaled$host_is_superhost, positive = "1")
print(conf_gbm)

# Plot confusion matrix
cm_gbm_df <- as.data.frame(conf_gbm$table)
ggplot(cm_gbm_df, aes(Prediction, Reference, fill = Freq)) +
  geom_tile() + geom_text(aes(label = Freq), color = "white", size = 5) +
  labs(title = "GBM Confusion Matrix")

# Metrics
gbm_metrics <- conf_gbm$byClass[c("Precision", "Recall", "F1")]
cat("GBM Accuracy:", conf_gbm$overall["Accuracy"], "\n")
print(gbm_metrics)

# Plot variable importance
gbm_importance <- summary(gbm_fit, plotit = FALSE)
ggplot(gbm_importance, aes(x = reorder(var, rel.inf), y = rel.inf)) +
  geom_col() + coord_flip() +
  labs(title = "GBM Variable Importance", y = "Relative Importance", x = "")

