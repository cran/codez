#' codez
#'
#' @param df A data frame with time features on columns. They could be numeric variables or categorical, but not both.
#' @param seq_len Positive integer. Time-step number of the forecasting sequence. Default: NULL (random selection within 2 to max preset boundary).
#' @param n_windows Positive integer. Number of validation windows to test prediction error. Default: 10.
#' @param latent Positive integer. Dimensions of the latent space for encoding-decoding operations. Default: NULL (random selection within preset boundaries)
#' @param smoother Logical. Perform optimal smoothing using standard loess for each time feature. Default: FALSE
#' @param n_samp Positive integer. Number of samples for random search. Default: 30.
#' @param autoencoder_layers_n Positive integer. Number of layers for the encoder-decoder model. Default: NULL (random selection within preset boundaries)
#' @param autoencoder_layers_size Positive integer. Numbers of nodes for the encoder-decoder model. Default: NULL (random selection within preset boundaries)
#' @param autoencoder_activ String. Activation function to be used by the encoder-decoder model. Implemented functions are: "linear", "relu", "leaky_relu", "selu", "elu", "sigmoid", "tanh", "swish", "gelu". Default: NULL (random selection within standard activations)
#' @param forward_net_layers_n Positive integer. Number of layers for the forward net model. Default: NULL (random selection within preset boundaries)
#' @param forward_net_layers_size Positive integer. Numbers of nodes for the forward net model. Default: NULL (random selection within preset boundaries)
#' @param forward_net_activ String. Activation function to be used by the forward net model. Implemented functions are: "linear", "relu", "leaky_relu", "selu", "elu", "sigmoid", "tanh", "swish", "gelu". Default: NULL (random selection within standard activations)
#' @param forward_net_reg_L1 Positive numeric between. Weights for L1 regularization. Default: NULL (random selection within preset boundaries).
#' @param forward_net_reg_L2 Positive numeric between. Weights for L2 regularization. Default: NULL (random selection within preset boundaries).
#' @param forward_net_drop Positive numeric between 0 and 1. Value for the dropout parameter for each layer of the forward net model (for example, a neural net with 3 layers may have dropout = c(0, 0.5, 0.3)). Default: NULL (random selection within preset boundaries).
#' @param loss_metric String. Loss function for both models. Available metrics: "mse", "mae", "mape". Default: "mae".
#' @param autoencoder_optimizer String. Optimization method for autoencoder. Implemented methods are: "adam", "adadelta", "adagrad", "rmsprop", "sgd", "nadam", "adamax". Default: NULL (random selection within standard optimizers).
#' @param forward_net_optimizer String. Optimization method for forward net. Implemented methods are: "adam", "adadelta", "adagrad", "rmsprop", "sgd", "nadam", "adamax". Default: NULL (random selection within standard optimizers).
#' @param epochs Positive integer. Default: 100.
#' @param patience Positive integer. Waiting time (in epochs) before evaluating the overfit performance. Default: 10.
#' @param holdout Positive numeric between 0 and 1. Holdout sample for validation. Default: 0.5.
#' @param verbose Logical. Default: FALSE.
#' @param ci Positive numeric. Confidence interval. Default: 0.8
#' @param error_scale String. Scale for the scaled error metrics (for continuous variables). Two options: "naive" (average of naive one-step absolute error for the historical series) or "deviation" (standard error of the historical series). Default: "naive".
#' @param error_benchmark String. Benchmark for the relative error metrics (for continuous variables). Two options: "naive" (sequential extension of last value) or "average" (mean value of true sequence). Default: "naive".
#' @param dates Date. Vector with dates for time features.
#' @param seed Positive integer. Random seed. Default: 42.
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#' @return This function returns a list including:
#' \itemize{
#' \item history: a table with the sampled models, hyper-parameters, validation errors
#' \item best_model: results for the best selected model according to the weighted average rank, including:
#' \itemize{
#' \item predictions: for continuous variables, min, max, q25, q50, q75, quantiles at selected ci, mean, sd, mode, skewness, kurtosis, IQR to range, risk ratio, upside probability and divergence for each point fo predicted sequences; for factor variables, min, max, q25, q50, q75, quantiles at selected ci, proportions, difformity (deviation of proportions normalized over the maximum possible deviation), entropy, upgrade probability and divergence for each point fo predicted sequences
#' \item testing_errors: testing errors for each time feature for the best selected model (for continuous variables: me, mae, mse, rmsse, mpe, mape, rmae, rrmse, rame, mase, smse, sce, gmrae; for factor variables: czekanowski, tanimoto, cosine, hassebrook, jaccard, dice, canberra, gower, lorentzian, clark)
#' \item plots: standard plots with confidence interval for each time feature
#' }
#' \item time_log
#' }
#'
#'
#' @export
#'
#' @importFrom fANCOVA loess.as
#' @importFrom imputeTS na_kalman
#' @import purrr
#' @import abind
#' @import keras
#' @import tensorflow
#' @import ggplot2
#' @import tictoc
#' @importFrom readr parse_number
#' @importFrom lubridate seconds_to_period is.Date
#' @importFrom scales number
#' @importFrom narray split
#' @importFrom stats ecdf loess predict sd lm median na.omit quantile
#' @importFrom utils head tail
#' @import fastDummies
#' @importFrom modeest mlv1
#' @importFrom moments skewness kurtosis
#' @import entropy
#' @import philentropy
#' @import greybox
#'

codez <- function(df, seq_len = NULL, n_windows = 10, latent = NULL, smoother = FALSE, n_samp = 30,
                  autoencoder_layers_n = NULL, autoencoder_layers_size = NULL, autoencoder_activ = NULL,
                  forward_net_layers_n = NULL, forward_net_layers_size = NULL, forward_net_activ = NULL,
                  forward_net_reg_L1 = NULL, forward_net_reg_L2 = NULL, forward_net_drop = NULL,
                  loss_metric = "mae", autoencoder_optimizer = NULL, forward_net_optimizer = NULL,
                  epochs = 100, patience = 10, holdout = 0.5, verbose = FALSE, ci = 0.8,
                  error_scale = "naive", error_benchmark = "naive", dates = NULL, seed = 42)
{
  tic.clearlog()
  tic("time")

  set.seed(seed)
  n_length <- nrow(df)

  class_index <- any(map_lgl(df, ~ is.factor(.x) | is.character(.x)))
  all_classes <- all(class_index)
  numeric_index <- map_lgl(df, ~ is.integer(.x) | is.numeric(.x))
  all_numerics <- all(numeric_index)
  if(!(all_classes | all_numerics)){stop("only all numerics or all classes, not both")}

  if(all_classes){df <- dummy_cols(df, select_columns = NULL, remove_first_dummy = FALSE, remove_most_frequent_dummy = TRUE, ignore_na = FALSE, split = NULL, remove_selected_columns = TRUE); binary_class <- rep(TRUE, ncol(df))}
  if(all_numerics){binary_class <- rep(FALSE, ncol(df))}

  if(anyNA(df) & all_numerics){df <- as.data.frame(na_kalman(df)); message("kalman imputation on time features\n")}
  if(anyNA(df) & all_classes){df <- floor( as.data.frame(na_kalman(df))); message("kalman imputation on time features\n")}
  if(smoother == TRUE & all_numerics){df <- as.data.frame(purrr::map(df, ~ suppressWarnings(loess.as(x=1:n_length, y=.x)$fitted))); message("performing optimal smoothing\n")}

  n_feats <- ncol(df)
  feat_names <- colnames(df)

  if(all_numerics){deriv <- min(map_dbl(df, ~ best_deriv(.x)))}
  if(all_classes){deriv <- 0}

  max_limit <- floor((n_length/(n_windows + 1)/4 - max(deriv) + 2))###AT LEAST TWO ROW IN THE SEGMENTED FRAME
  if(max_limit < 2){stop("not enough data for the validation windows")}

  standard_activations <- c("linear", "relu", "leaky_relu", "selu", "elu", "sigmoid", "tanh", "swish", "gelu")
  standard_optimizations <- c("adam", "adadelta", "adagrad", "rmsprop", "sgd", "nadam", "adamax")
  if(!is.null(autoencoder_activ) && !all(autoencoder_activ %in% standard_activations)){stop("non-standard activation in autoencoder")}
  if(!is.null(forward_net_activ) && !all(forward_net_activ %in% standard_activations)){stop("non-standard activation in forward net")}
  if(!is.null(autoencoder_optimizer) && !(autoencoder_optimizer %in% standard_optimizations)){stop("non-standard optimizer in autoencoder")}
  if(!is.null(forward_net_optimizer) && !(forward_net_optimizer %in% standard_optimizations)){stop("non-standard optimizer in forward net")}

  finite_solution <- (length(seq_len) == 1) && (length(latent) == 1) && (length(autoencoder_layers_n)==1) && (length(autoencoder_layers_size)==autoencoder_layers_n) && (length(autoencoder_activ)==autoencoder_layers_n) && (length(forward_net_layers_n)==1) && (length(forward_net_layers_size)==forward_net_layers_n) && (length(forward_net_activ)==forward_net_layers_n) && (length(forward_net_reg_L1)==forward_net_layers_n) && (length(forward_net_reg_L2)==forward_net_layers_n) && (length(forward_net_drop)==forward_net_layers_n)
  if(finite_solution & n_samp != 1)
  {
    n_samp <- 1
    history <- NULL

    best_model <- windower(df, seq_len, n_windows, latent, autoencoder_form = autoencoder_layers_size, autoencoder_activ, forward_net_form = forward_net_layers_size, forward_net_activ, forward_net_reg_L1, forward_net_reg_L2, forward_net_drop,
                           loss_metric, autoencoder_optimizer, epochs, patience, holdout, verbose, ci, error_scale, error_benchmark, dates,  binary_class, n_samp, seed)
    if(n_feats > 1){best_model$errors <- Reduce(rbind, best_model$errors); rownames(best_model$errors) <- feat_names}
    if(n_feats == 1){best_model$errors <- round(t(as.data.frame(best_model$errors)), 4)}
  }

  if(!finite_solution & n_samp >= 1)
  {
    sqln_set <- sampler(seq_len, n_samp, range = c(3, max_limit), integer = TRUE)
    ltt_set <- sampler(latent, n_samp, range = c(2, max(sqln_set)), integer = TRUE)

    if(any(sqln_set < 2)){sqln_set[sqln_set < 2] <- 2; message("setting min seq_len to 2")}
    if(any(sqln_set > max_limit)){sqln_set[sqln_set > max_limit] <- max_limit; message(paste0("setting max seq_len to ", max_limit))}
    ltt_set[ltt_set > sqln_set] <- sqln_set[ltt_set > sqln_set]
    ltt_set[ltt_set >= sqln_set] <- sqln_set[ltt_set >= sqln_set] - 1

    autoencoder_lyr_set <- sampler(autoencoder_layers_n, n_samp,  range = c(1, 5), integer = TRUE) ###PROBLEMI CON IL SINGLE LAYER
    autoencoder_lsz_set <- map(autoencoder_lyr_set, ~ unlist(sampler(autoencoder_layers_size, 1, range = seq(8, 1024, by = 8), integer = TRUE, multi = .x)))
    autoencoder_act_set <- map(autoencoder_lyr_set, ~ unlist(sampler(autoencoder_activ, 1, range = standard_activations, integer = FALSE, multi = .x)))

    forward_net_lyr_set <- sampler(forward_net_layers_n, n_samp,  range = c(1, 5), integer = TRUE) ###PROBLEMI CON IL SINGLE LAYER
    forward_net_lsz_set <- map(forward_net_lyr_set, ~ unlist(sampler(forward_net_layers_size, 1, range = seq(8, 1024, by = 8), integer = TRUE, multi = .x)))
    forward_net_act_set <- map(forward_net_lyr_set, ~ unlist(sampler(forward_net_activ, 1, range = standard_activations, integer = FALSE, multi = .x)))
    forward_net_rL1_set <- map(forward_net_lyr_set, ~ round(unlist(sampler(forward_net_reg_L1, 1, range = c(0.001, 1000), integer = FALSE, multi = .x)), 4))
    forward_net_rL2_set <- map(forward_net_lyr_set, ~ round(unlist(sampler(forward_net_reg_L1, 1, range = c(0.001, 1000), integer = FALSE, multi = .x)), 4))
    forward_net_drp_set <- map(forward_net_lyr_set, ~ round(unlist(sampler(forward_net_drop, 1, range = c(0.1, 0.9), integer = FALSE, multi = .x)), 4))
    autoencoder_opt_set <- sampler(autoencoder_optimizer, n_samp, range = standard_optimizations)
    forward_net_opt_set <- sampler(forward_net_optimizer, n_samp, range = standard_optimizations)

    hyper_param <- list(sqln_set, ltt_set, autoencoder_lsz_set, autoencoder_act_set, forward_net_lsz_set, forward_net_act_set, forward_net_rL1_set, forward_net_rL2_set, forward_net_drp_set, autoencoder_opt_set, forward_net_opt_set, 1:n_samp)

    exploration <- pmap(hyper_param, ~ windower(df, seq_len = ..1, n_windows, latent = ..2,
                                                autoencoder_form = ..3, autoencoder_activ = ..4, forward_net_form = ..5, forward_net_activ = ..6,
                                                forward_net_reg_L1 = ..7, forward_net_reg_L2 = ..8, forward_net_drop = ..9,
                                                loss_metric, autoencoder_optimizer = ..10, forward_net_optimizer = ..11,
                                                epochs, patience, holdout, verbose, ci, error_scale, error_benchmark,
                                                dates, binary_class, deriv, n_samp = ..12, seed))

    exploration <- purrr::transpose(exploration)

    models <- exploration$quant_pred
    errors <- exploration$errors
    plots <- exploration$plot

    if(n_feats > 1){aggr_errors <- map(errors, ~ colMeans(Reduce(rbind, .x)))} else {aggr_errors <- errors}
    aggr_errors <- t(as.data.frame(aggr_errors))

    history <- data.frame(seq_len = unlist(sqln_set))
    history$latent <- ltt_set
    history$autoencoder_layers <- autoencoder_lsz_set
    history$autoencoder_activations <- autoencoder_act_set
    history$autoencoder_optimizer <- autoencoder_opt_set
    history$forward_net_layers <- forward_net_lsz_set
    history$forward_net_activations <- forward_net_act_set
    history$forward_net_reg_L1 <- forward_net_rL1_set
    history$forward_net_reg_L2 <- forward_net_rL2_set
    history$forward_net_dropout <- forward_net_drp_set
    history$forward_net_optimizer <- forward_net_opt_set

    history <- data.frame(history, round(aggr_errors, 4))
    rownames(history) <- NULL

    if(all_numerics){history <- ranker(history, focus = -c(1:11), inverse = NULL, absolute = c("me", "mpe", "sce"), reverse = FALSE)}
    if(all_classes){history <- ranker(history, focus = -c(1:11), inverse = NULL, absolute = NULL, reverse = FALSE)}

    best_index <- as.numeric(rownames(history[1,]))
    predictions <- models[[best_index]]
    errors <- t(as.data.frame(errors[[best_index]]))
    plot <- plots[[best_index]]

    names(predictions) <- feat_names
    rownames(errors) <- feat_names
    names(plot) <- feat_names

    best_model <- list(predictions = predictions, errors = errors, plot = plot)
  }

  toc(log = TRUE)
  time_log <- seconds_to_period(round(parse_number(unlist(tic.log())), 0))

  outcome <- list(history = history, best_model = best_model, time_log = time_log)

  return(outcome)
}


###
windower <- function(df, seq_len, n_windows = 10, latent, autoencoder_form = c(128, 64),
                     autoencoder_activ = c("gelu", "gelu"), forward_net_form = c(128, 64), forward_net_activ = c("gelu", "gelu"),
                     forward_net_reg_L1 = c(0.5, 0.5), forward_net_reg_L2 = c(0.5, 0.5), forward_net_drop = c(0.5, 0.5),
                     loss_metric = "mae", autoencoder_optimizer = "adam", forward_net_optimizer = "adam", epochs = 100, patience = 10, holdout = 0.5,
                     verbose = FALSE, ci = 0.8, error_scale = "naive", error_benchmark = "naive",
                     dates = NULL, binary_class = rep(FALSE, ncol(df)), deriv = 0, n_samp, seed = 42)
{
  n_length <- nrow(df)
  fnames <- colnames(df)

  idx <- c(base::rep(1, n_length%%(n_windows + 1)), base::rep(1:(n_windows + 1), each = n_length/(n_windows + 1)))

  window_results <- map(1:n_windows, ~ engine(df[idx <= .x, , drop = FALSE], seq_len, holdout_truth = head(df[idx > .x, , drop = FALSE], seq_len), latent, autoencoder_form, autoencoder_activ, forward_net_form, forward_net_activ, forward_net_reg_L1, forward_net_reg_L2, forward_net_drop, loss_metric, autoencoder_optimizer, forward_net_optimizer, epochs, patience, holdout, verbose, ci, error_scale, error_benchmark, binary_class, deriv, seed))
  window_results <- transpose(window_results)
  errors <- map(map_depth(window_results, 2, ~ .x$testing_error), ~ colMeans(t(as.data.frame(.x))))
  pred_scores <- map(map_depth(window_results, 2, ~ .x$quant_pred$pred_scores), ~ colMeans(t(as.data.frame(.x))))

  model <- engine(df, seq_len, holdout_truth = NULL, latent, autoencoder_form, autoencoder_activ, forward_net_form, forward_net_activ, forward_net_reg_L1, forward_net_reg_L2, forward_net_drop, loss_metric, autoencoder_optimizer, forward_net_optimizer, epochs, patience, holdout, verbose, ci, error_scale, error_benchmark, binary_class, deriv, seed)
  message(paste0("model n. ", n_samp))

  quant_pred <- transpose(model)[[1]]
  quant_pred <- map2(quant_pred, pred_scores, ~ cbind(.x, pred_scores = .y))
  plot <- pmap(list(quant_pred, df, fnames), ~ plotter(quant_pred = ..1, ci, ts = ..2, n_class = NULL, level_names = NULL, dates, feat_name = ..3))
  quant_pred <- map(quant_pred, ~ dater(.x, dates))
  names(quant_pred) <- fnames
  names(errors) <- fnames
  names(plot) <- fnames

  outcome <- list(quant_pred = quant_pred, errors = errors, plot = plot)

  return(outcome)
}

###
engine <- function(df, seq_len, holdout_truth = NULL,
                   latent, autoencoder_form = c(128, 64), autoencoder_activ = c("gelu", "gelu"),
                   forward_net_form = c(128, 64), forward_net_activ = c("gelu", "gelu"),
                   forward_net_reg_L1 = c(0.5, 0.5), forward_net_reg_L2 = c(0.5, 0.5), forward_net_drop = c(0.5, 0.5),
                   loss_metric = "mae", autoencoder_optimizer = "Adam", forward_net_optimizer = "Adam",
                   epochs = 100, patience = 10, holdout = 0.5, verbose = FALSE, ci = 0.8,
                   error_scale = "naive", error_benchmark = "naive", binary_class, deriv = 0, seed = 42)
{

  seq_len <- seq_len + deriv
  segment_tensor <- abind::abind(map(df, ~ smart_reframer(.x, seq_len, seq_len)), along = 3)
  difframed_model <- reframed_differentiation(segment_tensor, deriv)
  difframed <- difframed_model$reframed

  autoencoder_model <- tf_autoencoder(tensor = difframed, n_layers = length(autoencoder_form),
                                      activations = autoencoder_activ, layer_sizes = autoencoder_form, latent_space = latent,
                                      latent_activation = "linear", output_activation = "linear",
                                      epochs, batch_size = 32, validation_split = 0.1,
                                      optimizer = autoencoder_optimizer, loss_metric,
                                      min_delta = 0.001, span = 0.1, seed)

  encoded <- autoencoder_model$encoding(difframed)

  network_model <- tf_snap(input_tensor = head(encoded, -1), output_tensor = tail(encoded, -1), holdout = holdout,
                           layers = length(forward_net_activ), activations = forward_net_activ, regularization_L1 = forward_net_reg_L1, regularization_L2 = forward_net_reg_L2, nodes = forward_net_form, dropout = forward_net_drop,
                           span = 0.2, min_delta = 0.001,  batch_size = 32, epochs, output_activation = "linear",
                           optimizer = forward_net_optimizer, loss_metric, seed, verbose = 0)


  ###TEST ERROR ESTIMATION
  test_index <- network_model$test_index
  encoded_input <- autoencoder_model$encoding(difframed)
  encoded_output <- network_model$pred_fun(encoded_input)
  decoded_output <- autoencoder_model$decoding(encoded_output)
  raw_errors <- segment_tensor[test_index,,,drop = FALSE] - reframed_integration(decoded_output, difframed_model$head_list, add = TRUE)[test_index,,,drop = FALSE]
  if(deriv > 0){raw_errors <- raw_errors[, - c(1:deriv),, drop = FALSE]}
  seq_len <- seq_len - deriv
  dimnames(raw_errors)[[2]] <- paste0("t", 1:seq_len)

  ###PREDICTION
  encoded_input <- autoencoder_model$encoding(tail(difframed, 1))
  encoded_output <- network_model$pred_fun(encoded_input)
  decoded_output <- autoencoder_model$decoding(encoded_output)

  diff_model <- map(df, ~ recursive_diff(.x, deriv))
  diff_tails <- map(diff_model, ~ .x$tail_value)
  seeds <- map2(narray::split(decoded_output, along = 3), diff_tails, ~ invdiff(.x, .y))
  raw_errors <- map(narray::split(raw_errors, along = 3), ~ as.data.frame(.x))

  raw_predictions <- map2(seeds, raw_errors, ~ prediction_integration(.x, .y))
  if(is.null(holdout_truth)){quantile_predictions <- pmap(list(raw_predictions, df, binary_class), ~ qpred(raw_pred = ..1, holdout_truth = NULL, ts = ..2, ci, error_scale, error_benchmark, binary_class = ..3, seed))}
  if(!is.null(holdout_truth)){quantile_predictions <- pmap(list(raw_predictions, df, binary_class, holdout_truth), ~ qpred(raw_pred = ..1, holdout_truth = ..4, ts = ..2, ci, error_scale, error_benchmark, binary_class = ..3, seed))}

  return(quantile_predictions)
}

###
prediction_integration <- function(seeds, raw_errors){as.matrix(as.data.frame(map2(seeds, raw_errors, ~ .x + sample(.y, size = 1000, replace = TRUE))))}

###
qpred <- function(raw_pred, holdout_truth = NULL, ts, ci, error_scale = "naive", error_benchmark = "naive", binary_class = FALSE, seed = 42)
{
  set.seed(seed)

  raw_pred <- doxa_filter(ts, raw_pred, binary_class)
  quants <- sort(unique(c((1-ci)/2, 0.25, 0.5, 0.75, ci+(1-ci)/2)))

  if(binary_class == FALSE)
  {
    p_stats <- function(x){c(min = suppressWarnings(min(x, na.rm = TRUE)), quantile(x, probs = quants, na.rm = TRUE), max = suppressWarnings(max(x, na.rm = TRUE)), mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE), mode = suppressWarnings(mlv1(x[is.finite(x)], method = "shorth")), kurtosis = suppressWarnings(kurtosis(x[is.finite(x)], na.rm = TRUE)), skewness = suppressWarnings(skewness(x[is.finite(x)], na.rm = TRUE)))}
    quant_pred <- as.data.frame(t(as.data.frame(apply(raw_pred, 2, p_stats))))
    p_value <- apply(raw_pred, 2, function(x) ecdf(x)(seq(min(raw_pred), max(raw_pred), length.out = 1000)))
    divergence <- c(max(p_value[,1] - seq(0, 1, length.out = 1000)), apply(p_value[,-1, drop = FALSE] - p_value[,-ncol(p_value), drop = FALSE], 2, function(x) abs(max(x, na.rm = TRUE))))
    upside_prob <- c(mean((raw_pred[,1]/tail(ts, 1)) > 1, na.rm = TRUE), apply(apply(raw_pred[,-1, drop = FALSE]/raw_pred[,-ncol(raw_pred), drop = FALSE], 2, function(x) x > 1), 2, mean, na.rm = TRUE))
    iqr_to_range <- (quant_pred[, "75%"] - quant_pred[, "25%"])/(quant_pred[, "max"] - quant_pred[, "min"])
    above_to_below_range <- (quant_pred[, "max"] - quant_pred[, "50%"])/(quant_pred[, "50%"] - quant_pred[, "min"])
    quant_pred <- round(cbind(quant_pred, iqr_to_range, above_to_below_range, upside_prob, divergence), 4)
  }

  if(binary_class == TRUE)
  {
    p_stats <- function(x){c(min = suppressWarnings(min(x, na.rm = TRUE)), quantile(x, probs = quants, na.rm = TRUE), max = suppressWarnings(max(x, na.rm = TRUE)), prop = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE), entropy = entropy(x))}
    quant_pred <- as.data.frame(t(as.data.frame(apply(raw_pred, 2, p_stats))))
    p_value <- apply(raw_pred, 2, function(x) ecdf(x)(c(0, 1)))
    divergence <- c(max(p_value[,1] - c(0, 1)), apply(p_value[,-1, drop = FALSE] - p_value[,-ncol(p_value), drop = FALSE], 2, function(x) abs(max(x, na.rm = TRUE))))
    upgrade_prob <- c(mean((raw_pred[,1]/tail(ts, 1)) > 1, na.rm = TRUE), apply(apply(raw_pred[,-1, drop = FALSE]/raw_pred[,-ncol(raw_pred), drop = FALSE], 2, function(x) x > 1), 2, mean, na.rm = TRUE))
    quant_pred <- round(cbind(quant_pred, upgrade_prob = upgrade_prob, divergence = divergence), 4)
  }

  testing_error <- NULL
  if(!is.null(holdout_truth))
  {
    mean_pred <- colMeans(raw_pred)
    testing_error <- custom_metrics(holdout_truth, mean_pred, ts, error_scale, error_benchmark, binary_class)
    pred_scores <- round(prediction_score(raw_pred, holdout_truth), 4)
    quant_pred <- cbind(quant_pred, pred_scores = pred_scores)
  }

  rownames(quant_pred) <- NULL

  outcome <- list(quant_pred = quant_pred, testing_error = testing_error)
  return(outcome)
}

###
doxa_filter <- function(ts, mat, binary_class = FALSE)
{
  discrete_check <- all(ts%%1 == 0)
  all_positive_check <- all(ts >= 0)
  all_negative_check <- all(ts <= 0)
  monotonic_increase_check <- all(diff(ts) >= 0)
  monotonic_decrease_check <- all(diff(ts) <= 0)

  monotonic_fixer <- function(x, mode)
  {
    model <- recursive_diff(x, 1)
    vect <- model$vector
    if(mode == 0){vect[vect < 0] <- 0; vect <- invdiff(vect, model$head_value, add = TRUE)}
    if(mode == 1){vect[vect > 0] <- 0; vect <- invdiff(vect, model$head_value, add = TRUE)}
    return(vect)
  }

  if(all_positive_check){mat[mat < 0] <- 0}
  if(all_negative_check){mat[mat > 0] <- 0}
  if(discrete_check){mat <- round(mat)}
  if(monotonic_increase_check){mat <- t(apply(mat, 1, function(x) monotonic_fixer(x, mode = 0)))}
  if(monotonic_decrease_check){mat <- t(apply(mat, 1, function(x) monotonic_fixer(x, mode = 1)))}

  if(binary_class == TRUE){mat[mat > 1] <- 1; mat[mat < 1] <- 0}
  mat <- na.omit(mat)

  return(mat)
}

###
recursive_diff <- function(vector, deriv)
{
  vector <- unlist(vector)
  head_value <- vector("numeric", deriv)
  tail_value <- vector("numeric", deriv)
  if(deriv==0){head_value = NULL; tail_value = NULL}
  if(deriv > 0){for(i in 1:deriv){head_value[i] <- head(vector, 1); tail_value[i] <- tail(vector, 1); vector <- diff(vector)}}
  outcome <- list(vector = vector, head_value = head_value, tail_value = tail_value)
  return(outcome)
}

###
invdiff <- function(vector, heads, add = FALSE)
{
  vector <- unlist(vector)
  if(is.null(heads)){return(vector)}
  for(d in length(heads):1){vector <- cumsum(c(heads[d], vector))}
  if(add == FALSE){return(vector[-c(1:length(heads))])} else {return(vector)}
}

###
custom_metrics <- function(holdout, forecast, actuals, error_scale = "naive", error_benchmark = "naive", binary_class = FALSE)
{

 if(binary_class == FALSE)
  {
    scale <- switch(error_scale, "deviation" = sd(actuals), "naive" = mean(abs(diff(actuals))))
    benchmark <- switch(error_benchmark, "average" = rep(mean(forecast), length(forecast)), "naive" = rep(tail(actuals, 1), length(forecast)))
    me <- ME(holdout, forecast, na.rm = TRUE)
    mae <- MAE(holdout, forecast, na.rm = TRUE)
    mse <- MSE(holdout, forecast, na.rm = TRUE)
    rmsse <- RMSSE(holdout, forecast, scale, na.rm = TRUE)
    mre <- MRE(holdout, forecast, na.rm = TRUE)
    mpe <- MPE(holdout, forecast, na.rm = TRUE)
    mape <- MAPE(holdout, forecast, na.rm = TRUE)
    rmae <- rMAE(holdout, forecast, benchmark, na.rm = TRUE)
    rrmse <- rRMSE(holdout, forecast, benchmark, na.rm = TRUE)
    rame <- rAME(holdout, forecast, benchmark, na.rm = TRUE)
    mase <- MASE(holdout, forecast, scale, na.rm = TRUE)
    smse <- sMSE(holdout, forecast, scale, na.rm = TRUE)
    sce <- sCE(holdout, forecast, scale, na.rm = TRUE)
    gmrae <- GMRAE(holdout, forecast, benchmark, na.rm = TRUE)
    out <- round(c(me = me, mae = mae, mse = mse, rmsse = rmsse, mpe = mpe, mape = mape, rmae = rmae, rrmse = rrmse, rame = rame, mase = mase, smse = smse, sce = sce, gmrae = gmrae), 3)
  }

  if(binary_class == TRUE)
  {
    dice <- suppressMessages(distance(rbind(holdout, forecast), method = "dice"))
    jaccard <- suppressMessages(distance(rbind(holdout, forecast), method = "jaccard"))
    cosine <- suppressMessages(distance(rbind(holdout, forecast), method = "cosine"))
    canberra <- suppressMessages(distance(rbind(holdout, forecast), method = "canberra"))
    gower <- suppressMessages(distance(rbind(holdout, forecast), method = "gower"))
    tanimoto <- suppressMessages(distance(rbind(holdout, forecast), method = "tanimoto"))
    hassebrook <- 1 - suppressMessages(distance(rbind(holdout, forecast), method = "hassebrook"))
    taneja <- suppressMessages(distance(rbind(holdout, forecast), method = "taneja"))
    lorentzian <- suppressMessages(distance(rbind(holdout, forecast), method = "lorentzian"))
    clark <- suppressMessages(distance(rbind(holdout, forecast), method = "clark"))
    sorensen <- suppressMessages(distance(rbind(holdout, forecast), method = "sorensen"))
    harmonic_mean <- suppressMessages(distance(rbind(holdout, forecast), method = "harmonic_mean"))
    avg <- suppressMessages(distance(rbind(holdout, forecast), method = "avg"))

    out <- round(c(dice, jaccard, cosine, canberra, gower, tanimoto, hassebrook, taneja, lorentzian, clark, sorensen, harmonic_mean, avg), 4)
  }

  return(out)
}

###
prediction_score <- function(integrated_preds, ground_truth)
{
  pfuns <- apply(integrated_preds, 2, ecdf)
  pvalues <- map2_dbl(pfuns, ground_truth, ~ .x(.y))
  scores <- 1 - 2 * abs(pvalues - 0.5)
  return(scores)
}

###

best_deriv <- function(ts, max_diff = 3, thresh = 0.001)
{
  pvalues <- vector(mode = "double", length = as.integer(max_diff))

  for(d in 1:(max_diff + 1))
  {
    model <- lm(ts ~ t, data.frame(ts, t = 1:length(ts)))
    pvalues[d] <- with(summary(model), pf(fstatistic[1], fstatistic[2], fstatistic[3],lower.tail=FALSE))
    ts <- diff(ts)
  }

  best <- tail(cumsum(pvalues < thresh), 1)

  return(best)
}

###

ranker <- function(df, focus, inverse = NULL, absolute = NULL, reverse = FALSE)
{
  rank_set <- df[, focus, drop = FALSE]
  if(!is.null(inverse)){rank_set[, inverse] <- - rank_set[, inverse]}###INVERSION BY COL NAMES
  if(!is.null(absolute)){rank_set[, absolute] <- abs(rank_set[, absolute])}###ABS BY COL NAMES
  index <- apply(scale(rank_set), 1, mean, na.rm = TRUE)
  if(reverse == FALSE){df <- df[order(index),]}
  if(reverse == TRUE){df <- df[order(-index),]}
  return(df)
}

###
ts_graph <- function(x_hist, y_hist, x_forcat, y_forcat, lower = NULL, upper = NULL, line_size = 1.3, label_size = 11,
                     forcat_band = "seagreen2", forcat_line = "seagreen4", hist_line = "gray43", label_x = "Horizon", label_y= "Forecasted Var", dbreak = NULL, date_format = "%b-%d-%Y")
{
  if(is.character(y_hist)){y_hist <- as.factor(y_hist)}
  if(is.character(y_forcat)){y_forcat <- factor(y_forcat, levels = levels(y_hist))}
  if(is.character(lower)){lower <- factor(lower, levels = levels(y_hist))}
  if(is.character(upper)){upper <- factor(upper, levels = levels(y_hist))}

  n_class <- NULL
  if(is.factor(y_hist)){class_levels <- levels(y_hist); n_class <- length(class_levels)}

  all_data <- data.frame(x_all = c(x_hist, x_forcat), y_all = as.numeric(c(y_hist, y_forcat)))
  forcat_data <- data.frame(x_forcat = x_forcat, y_forcat = as.numeric(y_forcat))

  if(!is.null(lower) & !is.null(upper)){forcat_data$lower <- as.numeric(lower); forcat_data$upper <- as.numeric(upper)}

  plot <- ggplot()+ geom_line(data = all_data, aes_string(x = "x_all", y = "y_all"), color = hist_line, size = line_size)
  if(!is.null(lower) & !is.null(upper)){plot <- plot + geom_ribbon(data = forcat_data, aes_string(x = "x_forcat", ymin = "lower", ymax = "upper"), alpha = 0.3, fill = forcat_band)}
  plot <- plot + geom_line(data = forcat_data, aes_string(x = "x_forcat", y = "y_forcat"), color = forcat_line, size = line_size)
  if(!is.null(dbreak)){plot <- plot + scale_x_date(name = paste0("\n", label_x), date_breaks = dbreak, date_labels = date_format)}
  if(is.null(dbreak)){plot <- plot + xlab(label_x)}
  if(is.null(n_class)){plot <- plot + scale_y_continuous(name = paste0(label_y, "\n"), labels = number)}
  if(is.numeric(n_class)){plot <- plot + scale_y_continuous(name = paste0(label_y, "\n"), breaks = 1:n_class, labels = class_levels)}
  plot <- plot + ylab(label_y)  + theme_bw()
  plot <- plot + theme(axis.text=element_text(size=label_size), axis.title=element_text(size=label_size + 2))

  return(plot)
}

###
sampler <- function(vect, n_samp, range = NULL, integer = FALSE, fun = NULL, multi = NULL)
{
  if(is.null(vect) & is.null(fun))
  {
    if(!is.character(range)){if(integer){set <- min(range):max(range)} else {set <- seq(min(range), max(range), length.out = 1000)}} else {set <- range}
    if(is.null(multi)){samp <- sample(set, n_samp, replace = TRUE)}
    if(is.numeric(multi)){samp <- replicate(n_samp, sample(set, multi, replace = TRUE), simplify = FALSE)}
  }

  if(is.null(vect) & !is.null(fun)){samp <- fun}

  if(is.null(multi)){
    if(length(vect)==1){samp <- rep(vect, n_samp)}
    if(length(vect) > 1){samp <- sample(vect, n_samp, replace = TRUE)}}

  if(is.numeric(multi)){
    if(length(vect)==1){samp <- replicate(n_samp, rep(vect, multi), simplify = FALSE)}
    if(length(vect) > 1){samp <- replicate(n_samp, sample(vect, multi, replace = TRUE), simplify = FALSE)}}

  return(samp)
}

###

plotter <- function(quant_pred, ci, ts, n_class = NULL, level_names = NULL, dates = NULL, feat_name)
{

  seq_len <- nrow(quant_pred)
  n_ts <- length(ts)

  if(is.Date(dates))
  {
    new_dates<- seq.Date(tail(dates, 1), tail(dates, 1) + seq_len * mean(diff(dates)), length.out = seq_len)
    x_hist <- dates
    x_forcat <- new_dates
    rownames(quant_pred) <- as.character(new_dates)
  }
  else
  {
    x_hist <- 1:n_ts
    x_forcat <- (n_ts + 1):(n_ts + seq_len)
    rownames(quant_pred) <- paste0("t", 1:seq_len)
  }

  quant_pred <- as.data.frame(quant_pred)
  x_lab <- paste0("Forecasting Horizon for sequence n = ", seq_len)
  y_lab <- paste0("Forecasting Values for ", feat_name)

  if(is.numeric(n_class) & !is.null(level_names)){ts <- level_names[ts + 1]}
  lower_b <- paste0((1-ci)/2 * 100, "%")
  upper_b <- paste0((ci+(1-ci)/2) * 100, "%")

  plot <- ts_graph(x_hist = x_hist, y_hist = ts, x_forcat = x_forcat, y_forcat = quant_pred[, "50%"], lower = quant_pred[, lower_b], upper = quant_pred[, upper_b], label_x = x_lab, label_y = y_lab)
  return(plot)
}

###
smart_reframer <- function(ts, seq_len, stride)
{
  n_length <- length(ts)
  if(seq_len > n_length | stride > n_length){stop("vector too short for sequence length or stride")}
  if(n_length%%seq_len > 0){ts <- tail(ts, - (n_length%%seq_len))}
  n_length <- length(ts)
  idx <- base::seq(from = 1, to = (n_length - seq_len + 1), by = 1)
  reframed <- t(sapply(idx, function(x) ts[x:(x+seq_len-1)]))
  if(seq_len == 1){reframed <- t(reframed)}
  idx <- rev(base::seq(nrow(reframed), 1, - stride))
  reframed <- reframed[idx,,drop = FALSE]
  colnames(reframed) <- paste0("t", 1:seq_len)
  return(reframed)
}


###
reframed_differentiation <- function(reframed, diff)
{
  ddim <- dim(reframed)[2]

  if(diff == 0){return(list(reframed = reframed, head_list = NULL, tail_list = NULL))}

  if(diff > 0)
  {
    if(diff >= ddim){stop("diff is greater/ equal to dim 2")}
    head_list <- list()
    tail_list <- list()

    for(d in 1:diff)
    {
      head_list <- append(head_list, list(reframed[, 1,, drop = FALSE]))
      tail_list <- append(tail_list,  list(reframed[, dim(reframed)[2],, drop = FALSE]))
      reframed <- reframed[,2:(ddim - d + 1),, drop = FALSE] - reframed[,1:(ddim - d + 1 - 1),, drop = FALSE]
    }
  }

  outcome <- list(reframed = reframed, head_list = head_list, tail_list = tail_list)
  return(outcome)
}

###
reframed_integration <- function(reframed, head_list, add = FALSE)
{
  if(is.null(head_list)){return(reframed)}

  diff <- length(head_list)
  for(d in diff:1)
  {
    reframed <- abind(head_list[[d]], reframed, along = 2)
    reframed <- aperm(apply(reframed, - 2, cumsum), c(2, 1, 3))
  }

  if(add == FALSE){reframed <- reframed[,-c(1:diff),, drop = FALSE]}

  return(reframed)
}

###
plotter <- function(quant_pred, ci, ts, n_class = NULL, level_names = NULL, dates = NULL, feat_name)
{
  seq_len <- nrow(quant_pred)
  n_ts <- length(ts)

  if(is.Date(dates))
  {
    new_dates<- seq.Date(tail(dates, 1), tail(dates, 1) + seq_len * mean(diff(dates)), length.out = seq_len)
    x_hist <- dates
    x_forcat <- new_dates
    rownames(quant_pred) <- as.character(new_dates)
  }
  else
  {
    x_hist <- 1:n_ts
    x_forcat <- (n_ts + 1):(n_ts + seq_len)
    rownames(quant_pred) <- paste0("t", 1:seq_len)
  }

  quant_pred <- as.data.frame(quant_pred)
  x_lab <- paste0("Forecasting Horizon for sequence n = ", seq_len)
  y_lab <- paste0("Forecasting Values for ", feat_name)

  if(is.numeric(n_class) & !is.null(level_names)){ts <- level_names[ts + 1]}
  lower_b <- paste0((1-ci)/2 * 100, "%")
  upper_b <- paste0((ci+(1-ci)/2) * 100, "%")

  plot <- ts_graph(x_hist = x_hist, y_hist = ts, x_forcat = x_forcat, y_forcat = quant_pred[, "50%"], lower = quant_pred[, lower_b], upper = quant_pred[, upper_b], label_x = x_lab, label_y = y_lab)
  return(plot)
}

###
dater <- function(quant_pred, dates)
{
  seq_len <- nrow(quant_pred)

  if(is.Date(dates))
  {
    new_dates<- seq.Date(tail(dates, 1), tail(dates, 1) + seq_len * mean(diff(dates)), length.out = seq_len)
    rownames(quant_pred) <- as.character(new_dates)
  }

  else

  {
    rownames(quant_pred) <- paste0("t", 1:seq_len)
  }

  return(quant_pred)
}

###
tf_snap <-function(input_tensor, output_tensor, holdout = 0.2,
                   layers = 1, activations = "relu", regularization_L1 = 0, regularization_L2 = 0, nodes = 32, dropout = 0,
                   span = 0.2, min_delta = 0, batch_size = 32, epochs = 50, output_activation = NULL,
                   optimizer = "Adam", loss_metric = NULL, seed = 42, verbose = 0)

{
  config <- tensorflow::tf$compat$v1$ConfigProto(gpu_options = list(allow_growth = TRUE)) ###per_process_gpu_memory_fraction AGGIUNTO ALLA LISTA
  sess <- tensorflow::tf$compat$v1$Session(config = config)
  tensorflow::tf$compat$v1$set_random_seed(seed) ###LIMITED REPRODUCIBILITY

  ###
  regr_metrics <- function(actual, predicted)
  {
    actual <- unlist(actual)
    predicted <- unlist(predicted)
    if(length(actual) != length(predicted)){stop("different lengths")}

    rmse <- sqrt(mean((actual - predicted)^2))
    mae <- mean(abs(actual - predicted))
    mdae <- median(abs(actual - predicted))
    mape <- mean(abs(actual - predicted)/actual)
    rrse <- sqrt(sum((actual - predicted)^2))/sqrt(sum((actual - mean(actual))^2))
    rae <- sum(abs(actual - predicted))/sum(abs(actual - mean(actual)))

    metrics <- round(c(rmse = rmse, mae = mae, mdae = mdae, mape = mape, rrse = rrse, rae = rae), 4)
    return(metrics)
  }

  ###PRED FUNCTION
  pred_fun <- function(new)
  {
    prediction <- predict(model, new, batch_size = batch_size)
    return(prediction)
  }


  ###
  x_rows <- dim(input_tensor)[1]
  x_cols <- dim(input_tensor)[2]
  y_rows <- dim(output_tensor)[1]
  y_cols <- dim(output_tensor)[2]

  set.seed(seed)
  test_index <- sample(x_rows, ceiling(holdout*x_rows))
  train_index <- setdiff(c(1:x_rows), test_index)

  x_train <- input_tensor[train_index,,, drop=FALSE]
  y_train <- output_tensor[train_index,,, drop=FALSE]
  x_test <- input_tensor[test_index,,, drop=FALSE]
  y_test <- output_tensor[test_index,,, drop=FALSE]

  ###DESIGN OF A SINGLE  NETWORK
  if(length(activations)<layers){activations <- replicate(layers, activations[1])}
  if(length(regularization_L1)<layers){regularization_L1 <- replicate(layers, regularization_L1[1])}
  if(length(regularization_L2)<layers){regularization_L2 <- replicate(layers, regularization_L2[1])}
  if(length(nodes)<layers){nodes <- replicate(layers, nodes[1])}
  if(length(dropout)<layers){dropout <- replicate(layers, dropout[1])}

  configuration<-data.frame(layers = NA, activations = NA, regularization_L1 = NA, regularization_L2 = NA, nodes = NA, dropout = NA)
  configuration$layers <- layers
  configuration$activations <- list(activations)
  configuration$regularization_L1 <- list(regularization_L1)
  configuration$regularization_L2 <- list(regularization_L2)
  configuration$nodes <- list(nodes)
  configuration$dropout <- list(dropout)

  ###CREATION OF KERAS NEURAL NET MODELS
  input <- layer_input(shape = as.integer(dim(input_tensor)[-1]))
  reshaped_in <- layer_reshape(input, target_shape = rev(as.integer(dim(input)[-1])))
  interim <- reshaped_in

  for(l in 1:configuration$layers)
  {
    interim <- layer_dense(object = interim, units = unlist(configuration$nodes)[l],
                           kernel_regularizer = regularizer_l1_l2(l1=unlist(configuration$regularization_L1)[l],
                                                                  l2=unlist(configuration$regularization_L2)[l]))

    standard_activations <- c("linear", "relu", "leaky_relu", "selu", "elu", "sigmoid", "tanh", "swish", "gelu")
    if(unlist(configuration$activations)[l] %in% standard_activations){interim<- layer_activation(object=interim, activation = unlist(configuration$activations)[l])} else stop("non-standard activation")

    interim<-layer_dropout(object=interim, rate=unlist(configuration$dropout)[l])
  }

  if(is.null(output_activation)){output_activation = "linear"}
  output <- layer_dense(object=interim, activation= output_activation, units = dim(output_tensor)[2])
  reshape_out <- layer_reshape(output, target_shape = as.integer(dim(output_tensor)[-1]))

  model <- keras_model(inputs = input, outputs = reshape_out)

  ###DEFAULT VALUES FOR MODEL COMPILE
  if(is.null(loss_metric)){loss_metric <- "mean_absolute_error"}
  compile(object = model, loss = loss_metric, optimizer = optimizer, metrics = loss_metric)

  history <- model %>% fit(x_train, y_train, epochs = epochs, batch_size=batch_size, verbose = verbose,
                           validation_data = list(x_test, y_test), callbacks = list(callback_early_stopping(monitor="val_loss",
                                                                                                            min_delta=min_delta, patience=floor(epochs*span), restore_best_weights=TRUE)))

  test_prediction <- pred_fun(x_test)
  reference <- output_tensor[test_index,,,drop=FALSE]
  test_metrics <- suppressWarnings(regr_metrics(reference, test_prediction))

  history_fixed <- history
  history_fixed$metrics <- map(history_fixed$metrics, ~ c(.x, rep(NA, epochs - length(.x))))
  plot <- plot(history_fixed)

  ###COLLECTED RESULTS
  outcome<-list(configuration = configuration, model = model, pred_fun = pred_fun, test_index = test_index,
                history = history, plot = plot, test_metrics = test_metrics)

  tf$compat$v1$Session$close(sess)
  tf$keras$backend$clear_session

  return(outcome)
}

###
tf_autoencoder <- function(tensor, n_layers = 3, activations = c("relu", "relu", "relu"),
                           layer_sizes = c(128, 64, 32), latent_space, latent_activation = "linear", output_activation = "linear",
                           epochs = 30, batch_size = 32, validation_split = 0.3, optimizer = "Adam", loss_metric = "mae",
                           min_delta = 0.001, span = 0.1, seed = 42)
{
  config <- tensorflow::tf$compat$v1$ConfigProto(gpu_options = list(allow_growth = TRUE)) ###per_process_gpu_memory_fraction AGGIUNTO ALLA LISTA
  sess <- tensorflow::tf$compat$v1$Session(config = config)
  tensorflow::tf$compat$v1$set_random_seed(seed) ###LIMITED REPRODUCIBILITY

  if(!is.array(tensor)){stop("tensor required")}
  dims <- dim(array)

  standard_activations <- c("linear", "relu", "leaky_relu", "selu", "elu", "sigmoid", "tanh", "swish", "gelu")
  if(!all(activations %in% standard_activations)){stop("non-standard activation")}
  if(length(activations)!= n_layers){activations <- rep(activations[1], n_layers)}
  if(length(layer_sizes)!= n_layers){layer_sizes <- rep(layer_sizes[1], n_layers)}

  encoder_input <- layer_input(shape = as.integer(dim(tensor)[-1]), name = "encoder_input")
  reshape_layer <- layer_reshape(encoder_input, target_shape = rev(as.integer(dim(tensor)[-1])), name = "reshape_in")

  encoder_list <- vector(mode = "list", length = as.integer(n_layers))
  for(i in 1:n_layers)
  {
    if(i == 1){encoder_list[[i]] <- layer_dense(reshape_layer, units = layer_sizes[i], activation = activations[i], name = paste0("encoding_layer_",i))}
    if(i > 1){encoder_list[[i]] <- layer_dense(encoder_list[[i - 1]], units = layer_sizes[i], activation = activations[i], name = paste0("encoding_layer_",i))}
  }

  latent_dim <- layer_dense(encoder_list[[n_layers]], units = latent_space, activation = latent_activation, name = "bottleneck")

  encoder <- keras_model(encoder_input, latent_dim, name = "encoder")

  decoder_input <- layer_input(shape = as.integer(unlist(encoder$output_shape)), name = "decoder_input")

  decoder_list <- vector(mode = "list", length = as.integer(n_layers))
  for(i in 1:n_layers)
  {
    if(i == 1){decoder_list[[i]] <- layer_dense(decoder_input, units = rev(layer_sizes)[i], activation = rev(activations)[i], name = paste0("decoding_layer_",i))}
    if(i > 1){decoder_list[[i]] <- layer_dense(decoder_list[[i - 1]], units = rev(layer_sizes)[i], activation = rev(activations)[i], name = paste0("decoding_layer_",i))}
  }

  output_layer <- layer_dense(decoder_list[[n_layers]], units = dim(tensor)[2], activation = output_activation, name = "output")
  reshape_layer <- layer_reshape(output_layer, target_shape = as.integer(dim(tensor)[-1]), name = "reshape_out")

  decoder <- keras_model(decoder_input, reshape_layer, name = "decoder")

  model <- keras_model(encoder_input, decoder(encoder(encoder_input)), name = "autoencoder")

  compile(object = model, loss = loss_metric, optimizer = optimizer, metrics = loss_metric)

  history <- model %>% keras::fit(x = tensor, y = tensor, epochs = epochs, batch_size = batch_size, verbose = 0, validation_split = validation_split,
                                  callbacks = list(callback_early_stopping(monitor = "loss", min_delta = min_delta, patience = floor(epochs*span), restore_best_weights=TRUE)))

  mean_error <- evaluate(model, tensor, tensor, verbose = 0)

  encoding <- function(new) predict(encoder, new)
  decoding <- function(new) predict(decoder, new)

  outcome <-  list(history = plot(history), encoder = encoder, decoder = decoder, model = model, mean_error = mean_error, encoding = encoding, decoding = decoding)

  tf$compat$v1$Session$close(sess)
  tf$keras$backend$clear_session

  return(outcome)
}
