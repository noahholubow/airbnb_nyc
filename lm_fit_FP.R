# Boosted Tree tuning

# load packages
library(tidyverse)
library(tidymodels)

# set seed
set.seed(1234)

# load necessary items
load("airbnb_setup_lm_only_log.rda")

# define model
lm_model <- linear_reg() %>% 
  set_mode("regression") %>% 
  set_engine("lm")

# workflow
lm_workflow <- 
  workflow() %>% 
  add_recipe(airbnb_recipe_lm_only) %>% 
  add_model(lm_model)


# fitting
lm_fit_folds <- fit_resamples(lm_workflow, resamples = airbnb_fold)

# write out results & workflow
save(lm_fit_folds, lm_workflow, file = "lm_tune_log.rda")


