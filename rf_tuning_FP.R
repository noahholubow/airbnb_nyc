# Random Forest tuning

# load packages
library(tidyverse)
library(tidymodels)
library(ranger)

# set seed
set.seed(123)

# load necessary items
load("airbnb_setup_log.rda")

# define model
rf_model <- rand_forest(mode = "regression",
    mtry = tune(),
    min_n = tune()) %>% 
  set_engine("ranger")

# check tuning parameters
# parameters(rf_model)

# setup tuning grid
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(2, 5))) # use 5 as upper bound instead of 8 vars

# define grid
rf_grid <- grid_regular(rf_params, levels = 5) # trying out every single combination of mtry and min_n

# workflow
rf_workflow <- 
  workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(airbnb_recipe)

# tuning/fitting
rf_tune <- rf_workflow %>% 
  tune_grid(resamples = airbnb_fold, grid = rf_grid)

# write out results & workflow
save(rf_tune, rf_workflow, file = "rf_tune_log.rda")

