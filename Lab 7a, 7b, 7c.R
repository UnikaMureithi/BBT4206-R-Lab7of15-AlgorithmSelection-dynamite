
## Installing and loading packages ====
if (require("stats")) {
  require("stats")
} else {
  install.packages("stats", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## mlbench 
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret 
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## MASS 
if (require("MASS")) {
  require("MASS")
} else {
  install.packages("MASS", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## glmnet 
if (require("glmnet")) {
  require("glmnet")
} else {
  install.packages("glmnet", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## kernlab 
if (require("kernlab")) {
  require("kernlab")
} else {
  install.packages("kernlab", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## rpart 
if (require("rpart")) {
  require("rpart")
} else {
  install.packages("rpart", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

if (require("arules")) {
  require("arules")
} else {
  install.packages("arules", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## arulesViz ----
if (require("arulesViz")) {
  require("arulesViz")
} else {
  install.packages("arulesViz", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## tidyverse ----
if (require("tidyverse")) {
  require("tidyverse")
} else {
  install.packages("tidyverse", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readxl ----
if (require("readxl")) {
  require("readxl")
} else {
  install.packages("readxl", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}





## lubridate ----
if (require("lubridate")) {
  require("lubridate")
} else {
  install.packages("lubridate", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## plyr ----
if (require("plyr")) {
  require("plyr")
} else {
  install.packages("plyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naniar ----
if (require("naniar")) {
  require("naniar")
} else {
  install.packages("naniar", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## RColorBrewer ----
if (require("RColorBrewer")) {
  require("RColorBrewer")
} else {
  install.packages("RColorBrewer", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## A. Linear Algorithms ----
### 1a. Logistic Regression without caret ----
# Loading and splitting the dataset
library(readr)
Loan_Default <- read_csv("data/Loan_Default.csv", 
                         col_types = cols(Employed = col_factor(levels = c("1", 
                                                                           "0")), Default = col_factor(levels = c("1", 
                                                                                                                  "0"))))
View(Loan_Default) 

# An 80:20 split of the dataset
train_index <- createDataPartition(Loan_Default$Default,
                                   p = 0.8,
                                   list = FALSE)
Loan_Default_train <- Loan_Default[train_index, ]
Loan_Default_test <- Loan_Default[-train_index, ]

#Training the model
Loan_Default_model_glm <- glm(Default ~ ., data = Loan_Default_train,
                              family = binomial(link = "logit"))

# Displaying the model's details
print(Loan_Default_model_glm)

# Making predictions on the test data
probabilities <- predict(Loan_Default_model_glm, Loan_Default_test, type = "response")
print(probabilities)
# A probability greater than 0.5 = 1(default), less than 0.5 = 0(non-default)
predictions <- ifelse(probabilities > 0.5, 1, 0)
print(predictions)
# Displaying the model's evaluation metrics
table(predictions, Loan_Default_test$Default)

### 1b. Logistic Regression Using caret ----
# Since we've already loaded and split the dataset, we go straight to training the model
# Applying 10-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 10)
set.seed(7)
Loan_Default_caret_model_logistic <-
  train(Default ~ ., data = Loan_Default_train,
        method = "regLogistic", metric = "Accuracy",
        preProcess = c("center", "scale"), trControl = train_control)
# Displaying the model
print(Loan_Default_caret_model_logistic)
# Make Predictions
predictions <- predict(Loan_Default_caret_model_logistic,
                       Loan_Default_test[, 1:4])
# Displaying the model's evaluation metrics
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

### 2a.Linear Discriminant Analysis without caret ----
# Training the model
Loan_default_model_lda <- lda(Default ~ ., data = Loan_Default_train)
# Displaying the model
print(Loan_default_model_lda)
# Making predictions
predictions <- predict(Loan_default_model_lda,
                       Loan_Default_test[, 1:4])$class
# Display Model's evaluation metrics
table(predictions, Loan_Default_test$Default)

### 2b. Linear Discriminant Analysis using caret ----
# Train the model
set.seed(10)
## applying Leave One Out Cross Validation resampling method
train_control <- trainControl(method = "LOOCV")
Loan_default_caret_model_lda <- train(Default ~ .,
                                      data = Loan_Default_train,
                                      method = "lda", metric = "Accuracy",
                                      preProcess = c("center", "scale"),
                                      trControl = train_control)
# Display the model's details
print(Loan_default_caret_model_lda)

# Making predictions on the test dataset
predictions <- predict(Loan_default_caret_model_lda,
                       Loan_Default_test[, 1:4])

# Display the model's evaluation metrics 
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

### 3a. Regularized Linear Regression without caret ----
# Splitting to get the feature matrix and target matrix
x <- as.matrix(Loan_Default[, 1:4])
y <- as.matrix(Loan_Default[, 5])

# Training the model; using elastic net 
Loan_default_model_glm <- glmnet(x, y, family = "binomial",
                                 alpha = 0.5, lambda = 0.001)

# Displaying the model's details 
print(Loan_default_model_glm)

# Making predictions 
predictions <- predict(Loan_default_model_glm, x, type = "class")

# Displaying the model's evaluation metrics 
table(predictions, Loan_Default$Default)

### 3b. Regularized Linear Regression using caret ----
# Training the model
set.seed(7)

# Resampling using 10 fold cross validation
train_control <- trainControl(method = "cv", number = 10)
Loan_default_caret_model_glmnet <-
  train(Default ~ ., data = Loan_Default_train,
        method = "glmnet", metric = "Accuracy",
        preProcess = c("center", "scale"), trControl = train_control)

#Display the model
print(Loan_default_caret_model_glmnet)

# Make predictions
predictions <- predict(Loan_default_caret_model_glmnet,
                       Loan_Default_test[, 1:4])

# Display the model's evaluation metrics
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")


## B. Non-Linear Algorithms ----
### 1. CART - Decision trees without caret ----
#Training the model
Loan_default_model_rpart <- rpart(Default ~ ., data = Loan_Default_train)

#Displaying model
print(Loan_default_model_rpart)

#Making predictions using the test dataset
predictions <- predict(Loan_default_model_rpart,
                       Loan_Default_test[, 1:4],
                       type = "class")

#Displaying the evaluation metrics
table(predictions, Loan_Default_test$Default)

confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")



### 2. Naïve Bayes Classifier without Caret
#Training the model
Loan_default_model_nb <- naiveBayes(Default ~ .,
                                    data = Loan_Default_train)

#Displaying the model's details
print(Loan_default_model_nb)

#Making predictions
predictions <- predict(Loan_default_model_nb,
                       Loan_Default_test[, 1:4])

# Displaying the evaluation metrics
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")




### 3. kNN using caret ----
# Training the model
set.seed(7)
# Resampling using 10 - fold cross validation
train_control <- trainControl(method = "cv", number = 10)
Loan_default_caret_model_knn <- train(Default ~ ., data = Loan_Default,
                                      method = "knn", metric = "Accuracy",
                                      preProcess = c("center", "scale"),
                                      trControl = train_control)
#Displaying the model
print(Loan_default_caret_model_knn)

# Making predictions
predictions <- predict(Loan_default_caret_model_knn,
                       Loan_Default_test[, 1:4])

# Displaying evaluation metrics
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")


### 4a. Support Vector Machine without CARET ----
# Training the model 
Loan_default_model_svm <- ksvm(Default ~ ., data = Loan_Default_train,
                               kernel = "rbfdot")

#Displaying the model
print(Loan_default_model_svm)

# Make predictions 
predictions <- predict(Loan_default_model_svm, Loan_Default_test[, 1:4],
                       type = "response")

# Displaying the evaluation metrics 
table(predictions, Loan_Default_test$Default)

confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

### 4b. Support Vector Machine using CARET ----
# Training the model 
set.seed(7)
# Resampling using 10-fold cross validation
train_control <- trainControl(method = "cv", number = 10)
Loan_default_caret_model_svm_radial <- 
  train(Default ~ ., data = Loan_Default_train, method = "svmRadial",
        metric = "Accuracy", trControl = train_control)

# Display the model
print(Loan_default_caret_model_svm_radial)

# Making predictions 
predictions <- predict(Loan_default_caret_model_svm_radial,
                       Loan_Default_test[, 1:4])

# Display the evaluation metrics 
table(predictions, Loan_Default_test$Default)
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         Loan_Default_test[, 1:5]$Default)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")



## CLUSTERING----
# Load the Breast Cancer dataset
data("BreastCancer")

# View the structure of the dataset
str(BreastCancer)

# Get the dimensions of the dataset (number of rows and columns)
dim(BreastCancer)

# View the first few rows of the dataset
head(BreastCancer)

# Get summary statistics for the dataset
summary(BreastCancer)

# Check for missing data in the Breast Cancer dataset
# Are there missing values in the dataset?
any_na(BreastCancer)

# How many missing values?
n_miss(BreastCancer)

# What is the proportion of missing data in the entire dataset?
prop_miss(BreastCancer)

# What is the number and percentage of missing values grouped by each variable?
miss_var_summary(BreastCancer)

# Which variables contain the most missing values?
gg_miss_var(BreastCancer)

# Which combinations of variables are missing together?
#gg_miss_upset(BreastCancer)

# Where are missing values located (the shaded regions in the plot)?
vis_miss(BreastCancer) +
  theme(axis.text.x = element_text(angle = 80))

# Check for missing values in 'Bare.nuclei'
any(is.na(BreastCancer$Bare.nuclei))

# Convert 'Bare.nuclei' to numeric (if not already)
BreastCancer$Bare.nuclei <- as.numeric(BreastCancer$Bare.nuclei)

# Impute missing values with the mean of the non-missing values
mean_value <- mean(BreastCancer$Bare.nuclei, na.rm = TRUE)
BreastCancer$Bare.nuclei[is.na(BreastCancer$Bare.nuclei)] <- mean_value

# Check if missing values have been imputed
any(is.na(BreastCancer$Bare.nuclei))

# Check the column names of the dataset
colnames(BreastCancer)

#EDA
# Visualize the distribution of the target variable
ggplot(BreastCancer, aes(x = Class)) +
  geom_bar(fill = "skyblue") +
  labs(title = "Distribution of Diagnosis (Malignant and Benign)")

# Identify columns that are not numeric or integer
non_numeric_cols <- sapply(BreastCancer, function(x) !is.numeric(x) && !is.integer(x))

# Convert non-numeric columns to numeric
BreastCancer[, non_numeric_cols] <- lapply(BreastCancer[, non_numeric_cols], as.numeric)

# Compute the correlation matrix
correlation_matrix <- cor(BreastCancer)

# Visualize the correlation matrix
corrplot(correlation_matrix, method = "color")

# Select only the numeric columns for the scatter plot
numeric_cols <- sapply(BreastCancer, is.numeric)
numeric_data <- BreastCancer[, numeric_cols]

# Create scatter plots
pairs(numeric_data)

# Select only the numeric columns for standardization
numeric_cols <- sapply(BreastCancer, is.numeric)
numeric_data <- BreastCancer[, numeric_cols]

# Standardize the data
scaled_data <- scale(numeric_data)

# Convert the scaled data back to a data frame
scaled_data <- as.data.frame(scaled_data)

# add the column names back to the scaled_data data frame
colnames(scaled_data) <- colnames(numeric_data)


# Specify the number of clusters you want to create (e.g., 2 for benign and malignant)
num_clusters <- 2

# Perform K-Means clustering
kmeans_result <- kmeans(scaled_data, centers = num_clusters)

# Add the cluster assignments to your original dataset
BreastCancer$cluster <- kmeans_result$cluster

# The 'cluster' column in 'BreastCancer' now contains the cluster assignments

# view the cluster centers using:
kmeans_result$centers

# To visualize the clusters we use scatter plot
ggplot(BreastCancer, aes(x = Cl.thickness, y = Cell.size, color = factor(cluster))) +
  geom_point() +
  labs(title = "K-Means Clustering of Breast Cancer Data")

# Calculate the total within-cluster variance for different numbers of clusters
wcss <- numeric(length = 10)  # Initialize a vector to store within-cluster variance

for (i in 1:10) {
  kmeans_model <- kmeans(scaled_data, centers = i)
  wcss[i] <- kmeans_model$tot.withinss
}

wcss <- numeric(length = 10)  # Initialize a vector to store within-cluster variance

# Create a scree plot to identify the optimal number of clusters
plot(1:10, wcss, type = "b", xlab = "Number of Clusters", ylab = "Total Within-Cluster Variance", main = "Scree Plot")



# After identifying the elbow point, you can choose the optimal number of clusters.
# For example, if the elbow point is at k=3, you can perform K-Means clustering with 3 clusters.
optimal_clusters <- 3  

# Perform K-Means clustering with the optimal number of clusters
kmeans_result <- kmeans(scaled_data, centers = optimal_clusters)

# Add the cluster assignments to your original dataset
BreastCancer$cluster <- kmeans_result$cluster

#  view the cluster centers using:
kmeans_result$centers

# To visualize the clusters, you can create a scatter plot
# using a subset of the variables (e.g., 'Cl.thickness' and 'Cell.size')
library(ggplot2)
ggplot(BreastCancer, aes(x = Cl.thickness, y = Cell.size, color = factor(cluster))) +
  geom_point() +
  labs(title = "K-Means Clustering of Breast Cancer Data")


#  ASSOCIATION ----
# STEP 1. Install and Load the Required Packages ----
## arules ----
if (require("arules")) {
  require("arules")
} else {
  install.packages("arules", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## arulesViz ----
if (require("arulesViz")) {
  require("arulesViz")
} else {
  install.packages("arulesViz", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## tidyverse ----
if (require("tidyverse")) {
  require("tidyverse")
} else {
  install.packages("tidyverse", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readxl ----
if (require("readxl")) {
  require("readxl")
} else {
  install.packages("readxl", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## knitr ----
if (require("knitr")) {
  require("knitr")
} else {
  install.packages("knitr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## lubridate ----
if (require("lubridate")) {
  require("lubridate")
} else {
  install.packages("lubridate", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## plyr ----
if (require("plyr")) {
  require("plyr")
} else {
  install.packages("plyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naniar ----
if (require("naniar")) {
  require("naniar")
} else {
  install.packages("naniar", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## RColorBrewer ----
if (require("RColorBrewer")) {
  require("RColorBrewer")
} else {
  install.packages("RColorBrewer", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# STEP 2. Load and pre-process the dataset ----
## Using Basket Format----
transactions_basket_format <-
  read.transactions("data/transactions_basket_format.csv",
                    format = "basket", sep = ",", cols = 2)
View(transactions_basket_format)
print(transactions_basket_format)

# Reading the set
retail_2 <- read_csv("data/new_online_retail.csv")
dim(retail_2)

### Handle missing values ----
# Are there missing values in the dataset?
any_na(retail_2)

# How many?
n_miss(retail_2)

# What is the proportion of missing data in the entire dataset?
prop_miss(retail_2)

# What is the number and percentage of missing values grouped by
# each variable?
miss_var_summary(retail_2)

# Which variables contain the most missing values?
gg_miss_var(retail_2)

# Which combinations of variables are missing together?
gg_miss_upset(retail_2)

#### Remove the variables with missing values ----
# Using 'InvoiceNo' instead of 'CustomerID-
retail_2_removed_vars <- retail_2 %>% dplyr::filter(complete.cases(.))

dim(retail_2_removed_vars)

# Are there missing values in the dataset?
any_na(retail_2_removed_vars)

# What is the number and percentage of missing values grouped by
# each variable?
miss_var_summary(retail_2_removed_vars)

# We now remove the observations that do not have a value for the description
# variable.
retail_2_removed_vars_obs <- retail_2_removed_vars %>% filter(complete.cases(.))

dim(retail_2_removed_vars_obs)

## Identify categorical variables ----
# Ensure the customer's country is recorded as categorical data
retail_2_removed_vars_obs %>% mutate(Country = as.factor(Country))

# Also ensure that the description (name of the product purchased) is recorded
# as categorical data
retail_2_removed_vars_obs %>% mutate(Description = as.factor(Description))
str(retail_2_removed_vars_obs)

dim(retail_2_removed_vars_obs)
head(retail_2_removed_vars_obs)

## Record the date and time variables in the correct format ----
# Ensure that InvoiceDate is stored in the correct date format.
# We can separate the date and the time into 2 different variables.
retail_2_removed_vars_obs$trans_date <-
  as.Date(retail_2_removed_vars_obs$InvoiceDate)

# Extract time from InvoiceDate and store it in another variable
retail_2_removed_vars_obs$trans_time <-
  format(retail_2_removed_vars_obs$InvoiceDate, "%H:%M:%S")

## Record the InvoiceNo in the correct format (numeric) ----
# Convert InvoiceNo into numeric
retail_2_removed_vars_obs$invoice_no <-
  as.numeric(as.character(retail_2_removed_vars_obs$InvoiceNo))

# The NAs introduced by coercion represent cancelled invoices. The OLTP system
# of the business represents cancelled invoice with the prefix "C", e.g.
# "C536391".

# Are there missing values in the dataset?
any_na(retail_2_removed_vars_obs)

# What is the number and percentage of missing values grouped by
# each variable?
miss_var_summary(retail_2_removed_vars_obs)
dim(retail_2_removed_vars_obs)

