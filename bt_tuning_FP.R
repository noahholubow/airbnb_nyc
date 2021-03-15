# Random Forest tuning

# load packages
library(tidyverse)
library(tidymodels)
library(ranger)

# set seed
set.seed(2468)

# load necessary items
load("airbnb_setup_log.rda")

# define model
bt_model <- boost_tree(mode = "regression",
    mtry = tune(),
    min_n = tune(), 
    learn_rate = tune()) %>% 
  set_engine("xgboost", importance = "impurity")

# setup tuning grid
bt_params <- parameters(bt_model) %>% 
  update(mtry = mtry(range = c(2, 5)),
         learn_rate = learn_rate(range = c(-5, 0)))

# define grid
bt_grid <- grid_regular(bt_params, levels = 5) # trying out every single combination of mtry and min_n

# workflow
bt_workflow <- 
  workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(airbnb_recipe)

# tuning/fitting
bt_tune <- bt_workflow %>% 
  tune_grid(resamples = airbnb_fold, grid = bt_grid)

# write out results & workflow
save(bt_tune, bt_workflow, file = "bt_tune_log.rda")

