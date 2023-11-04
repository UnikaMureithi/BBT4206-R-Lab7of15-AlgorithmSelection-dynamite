if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naniar ----
if (require("naniar")) {
  require("naniar")
} else {
  install.packages("naniar", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## corrplot ----
if (require("corrplot")) {
  require("corrplot")
} else {
  install.packages("corrplot", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## ggcorrplot ----
if (require("ggcorrplot")) {
  require("ggcorrplot")
} else {
  install.packages("ggcorrplot", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

if (!require("mlbench")) {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
library(mlbench)

if (!requireNamespace("GGally", quietly = TRUE)) {
  install.packages("GGally")
}
library(GGally)

if (!requireNamespace("stats", quietly = TRUE)) {
  install.packages("stats")
}


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

#Creator: Dr. WIlliam H. Wolberg (physician); University of Wisconsin Hospital ;Madison; Wisconsin; USA

#Donor: Olvi Mangasarian (mangasarian@cs.wisc.edu)

#Received: David W. Aha (aha@cs.jhu.edu)

#These data have been taken from the UCI Repository Of Machine Learning Databases at

#ftp://ftp.ics.uci.edu/pub/machine-learning-databases

#http://www.ics.uci.edu/~mlearn/MLRepository.html

#and were converted to R format by Evgenia Dimitriadou.