# Boosted Tree tuning

# load packages
library(tidyverse)
library(tidymodels)

# set seed
set.seed(2021)

# load necessary items
load("airbnb_setup_log.rda")

# define model
knn_model <- nearest_neighbor(
  mode = "regression",
  neighbors = tune()) %>% 
  set_engine("kknn")

# setup tuning grid
knn_params <- parameters(knn_model)

# define grid
knn_grid <- grid_regular(knn_params, levels = 5) # trying out every single combination of mtry and min_n

# workflow
knn_workflow <- 
  workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(airbnb_recipe)

# tuning/fitting
knn_tune <- knn_workflow %>% 
  tune_grid(resamples = airbnb_fold, grid = knn_grid)

# write out results & workflow
save(knn_tune, knn_workflow, file = "knn_tune_log.rda")




