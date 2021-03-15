# Loading Packages, Setting Seed ------------------------------------------
# loading packages
library(dplyr)
library(tidyverse)
library(skimr)
library(janitor)
library(lubridate)
library(patchwork)
library(tidymodels)
library(car)
library(lmtest)
library(multcomp)
library(stringr)


# setting seed
set.seed(3012)


# Reading in Dataset ------------------------------------------------------
# loading dataset
airbnb <- read_csv("Data/unprocessed/AB_NYC_2019.csv") %>%
  clean_names()
airbnb

# inspecting missingness
airbnb %>% skim_without_charts()


# EDA ---------------------------------------------------------------------
# looking at distribution of accident severity
# histogram
p1 <- airbnb %>% 
  ggplot(aes(x = price)) +
  geom_histogram() +
  xlim(0, 2500)

# boxplot
p2 <- airbnb %>% 
  ggplot(aes(x = price)) +
  geom_boxplot()

p1 / p2

# log-transforming
airbnb_log <- airbnb %>% 
  mutate(price = log(price, base = 10)) %>% 
  filter(price > 0)

# viewing data after transformation
p3 <- airbnb_log %>% 
  ggplot(aes(x = price)) +
  geom_histogram() +
  xlim(0.8,4.5)

# inspecting by borough
airbnb_log %>% 
  ggplot(aes(x = price)) +
  geom_histogram() +
  xlim(0, 4.5) +
  facet_grid(rows = "neighbourhood_group")

# looking at distribution of min_nights between 0 and 10
airbnb_log %>% 
  ggplot(aes(x = minimum_nights)) +
  geom_histogram() +
  xlim(0, 10)

p1 + p3


# Splitting Data ----------------------------------------------------------
# splitting data
airbnb_split <- initial_split(airbnb_log, prop = 0.75, strata = price)
airbnb_split

# training data
airbnb_train <- training(airbnb_split)

# testing data
airbnb_test <- testing(airbnb_split)

# verifying dimensions
dim(airbnb_train)
dim(airbnb_test)


# Folds -------------------------------------------------------------------
# folding data using 10 folds, 5 repeats
airbnb_fold <- vfold_cv(airbnb_train, v = 5, repeats = 3, strata = price)
airbnb_fold


# Recipe, Prep, Bake ------------------------------------------------------------------
# Recipe for NON MACHINE LEARNING
airbnb_recipe_lm_only <- recipe(
  price ~  neighbourhood_group + room_type + minimum_nights + number_of_reviews + reviews_per_month +
    calculated_host_listings_count + availability_365, # including variables
  data = airbnb_train) %>%
  step_impute_linear(reviews_per_month) %>%  # imputing the (few) missing data
  step_dummy(all_nominal()) %>% # creating dummy vars from categorical data
  step_normalize(all_predictors())

# Recipe for MACHINE LEARNING
airbnb_recipe <- recipe(
  price ~  neighbourhood_group + room_type + minimum_nights + number_of_reviews + reviews_per_month +
    calculated_host_listings_count + availability_365, # including variables
  data = airbnb_train) %>%
  step_impute_linear(reviews_per_month) %>%  # imputing the (few) missing data
  step_dummy(all_nominal(), one_hot = TRUE) %>%  # creating dummy vars from categorical data with one-hot encoding
  step_normalize(all_predictors())

# prepping, baking
samp_bake <- airbnb_recipe %>% 
  prep(airbnb_train) %>% 
  bake(new_data = NULL) 
samp_bake

# Preparing Output File ---------------------------------------------------
# objects required for tuning
save(airbnb_fold, airbnb_recipe, airbnb_split, file = "Data/setups/airbnb_setup_log.rda")

# objects required for tuning LM ONLY
save(airbnb_fold, airbnb_recipe_lm_only, airbnb_split, file = "Data/setups/airbnb_setup_lm_only_log.rda")


# Loading Processed Data --------------------------------------------------
load(file = "data/processed/log/rf_tune_log.rda")
load(file = "data/processed/log/knn_tune_log.rda")
load(file = "data/processed/log/bt_tune_log.rda")
load(file = "data/processed/log/lm_tune_log.rda")

# Inspecting Tuned Info ---------------------------------------------------
# autoplots
rf_tune %>% 
  autoplot(metric = "rmse")

knn_tune %>% 
  autoplot(metric = "rmse")

bt_tune %>% 
  autoplot(metric = "rmse")


# Overall Results -----------------------------------------------------
# creating tibble with tune results
tune_results <- tibble(
  model_type = c("rf", "knn", "boost"),
  tune_info = list(rf_tune, knn_tune, bt_tune),
  assessment_info = map(tune_info, collect_metrics),
  best_model = map(tune_info, ~ select_best(.x, metric = "rmse")))
tune_results

head(tune_results %>% 
       dplyr::select(model_type, assessment_info) %>% 
       unnest(assessment_info) %>% 
       filter(.metric == "rmse") %>% 
       arrange(desc(mean)))

summary(lm_results)


# Predictions & Individual Results ---------------------------------------------------
# lm
lm_results <- fit(lm_workflow, airbnb_train)
predict(lm_results, new_data = airbnb_test) %>%
  bind_cols(airbnb_test %>% dplyr::select(price)) %>%
  rsq(truth = price, estimate = .pred)
summary(lm_results)

# rf
# finalizing workflow on whole training set
rf_workflow_tuned <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tune, metric = "rmse"))

# looking at results from fitting training set
rf_results <- fit(rf_workflow_tuned, airbnb_train)
rf_results

# using predict function to fit testing set
predict(rf_results, new_data = airbnb_test) %>% 
  bind_cols(airbnb_test %>% dplyr::select(price)) %>% 
  rmse(truth = price, estimate = .pred)
