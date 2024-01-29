# Data preprocessing
df <- read.csv("C://Users//wl//Desktop//Work//[Website]//Demand Forecast//Kaggle Challenge//train.csv")
View(df) # 2013-01-01 to 2017-12-31

unique(df$store)
unique(df$item)

df[!complete.cases(df), ] # find how many NA rows


# Exploratory data analysis
library(dplyr)
library(lubridate)
library(ggplot2)
library(scales)
library(ggthemes)

df2 <- df |>
  filter(store == 1 & item == 1) |>
  arrange(date)

df2$date <- ymd(df2$date) # convert to date format

pacf(df2$sales) # we can see that day-of-the-week effect and the previous-day effect are the most important

exploratory <- ggplot() +
  geom_line(data = df2, aes(x = date, y = sales, color = "#CC79A7")) +
  scale_x_date(labels = date_format("%Y-%b"), breaks = date_breaks("12 months")) +
  xlab("Date") +
  ylab("Amount ($)") +
  ggtitle("Daily Sales for item 1 store 1") +
  guides(color = "none")

exploratory


# Feature engineering
pce <- read.csv("C://Users//wl//Desktop//Work//[Website]//Demand Forecast//Kaggle Challenge//PCE.csv")
pce$DATE <- ymd(pce$DATE) # convert to date format


# rename DATE to lowercase and create year and month columns
pce$date <- pce$DATE 
pce <- pce |>
  select(-DATE) |>
  mutate(year = year(date), month = month(date))


# join pce with df2
df3 <- df2 |>
  mutate(year = year(date), month = month(date)) |>
  left_join(pce, by = c("year", "month")) |>
  mutate(date = date.x) |>
  select(date, sales, PCE)


# Data preprocessing
library(keras)
library(tensorflow)

tensorflow::set_random_seed( 
  seed = 333, 
  disable_gpu = FALSE
) 


# scaling the dataset
scale_factors_sales <- c(mean(df3$sales), sd(df3$sales))
scale_factors_consumption <- c(mean(df3$PCE), sd(df3$PCE))

scaled_train <- df3 |>
  mutate(sales = (sales - scale_factors_sales[1]) / scale_factors_sales[2]) |>
  mutate(PCE = (PCE - scale_factors_consumption[1]) / scale_factors_consumption[2]) |>
  select(sales, PCE)


# preparing the three-dimensional arrays
lag <- 364

scaled_train <- as.matrix(scaled_train)
 
x_train_data_1 <- t(sapply(
    1:(nrow(df3) - lag - lag),
    function(x) scaled_train[x:(x + lag - 1), 1]
  ))
 
x_train_data_2 <- t(sapply(
    1:(nrow(df3) - lag - lag),
    function(x) scaled_train[x:(x + lag - 1), 2]
  ))

x_train_arr <- array(
    data = as.numeric(unlist(c(x_train_data_1, x_train_data_2))),
    dim = c(
        nrow(x_train_data_1),
        lag,
        2
    )
)

y_train_data <- scaled_train[(nrow(df3) - nrow(x_train_data_1) + 1 - lag):(nrow(df3) - lag), 1]

y_train_arr <- array(
    data = as.numeric(unlist(c(y_train_data))),
    dim = c(
        length(y_train_data),
        1
    )
)


# Constructing model architecture
# initiating a keras model
lstm_model <- keras_model_sequential()


# adding layers and compiling the model
lstm_model |>
  layer_lstm(
    units = 64,
    batch_input_shape = c(1, lag, 2), # batch size, timesteps, features
    return_sequences = TRUE,
    stateful = TRUE,
    activation = "tanh"
  ) |>
  layer_dropout(rate = 0.4) |>
  bidirectional(layer_lstm(units = 64)) |>
  layer_dropout(rate = 0.4) |>
  layer_dense(units = 1, activation = "tanh") |>
  compile(loss = 'mean_squared_error',
          optimizer = "adam",
          metrics = 'mse')

summary(lstm_model)


# Training the model
# fitting the model to training dataset
history <- lstm_model |> fit(
  x = x_train_arr,
  y = y_train_arr,
  batch_size = 1,
  epochs = 10,
  verbose = 1,
  shuffle = FALSE,
  use_multiprocessing = T,
  validation_split = 0.2,
  callback_early_stopping(
    monitor = "val_loss",
    min_delta = 0,
    patience = 2,
    verbose = 0,
    mode = "auto",
    baseline = NULL,
    restore_best_weights = TRUE
  )
)


# plotting train and validation loss for each epoch
plot(history)


# Checking model performance on test data
# initialize vector to store forecasts
periods_to_pred <- 364
forecasted_store <- c()


# we will first forecast the very next period with observed lagged values
observed_sequence_1 <- scaled_train[(nrow(df3) - periods_to_pred + 1 - periods_to_pred):(nrow(df3) - periods_to_pred), 1]
observed_sequence_2 <- scaled_train[(nrow(df3) - periods_to_pred + 1 - periods_to_pred):(nrow(df3) - periods_to_pred), 2]

input_sequence <- array(
    data = as.numeric(unlist(c(observed_sequence_1, observed_sequence_2))),
    dim = c(
        1,
        periods_to_pred,
        2
    )
)

forecasted_value <- lstm_model |>
    predict(input_sequence)

forecasted_store <- forecasted_store |>
  append(forecasted_value)


# now we do for the rest of the days in the year
for (i in 1:(periods_to_pred - 1)) {

observed_sequence_1 <- scaled_train[(nrow(df3) - periods_to_pred + 1 + i - periods_to_pred):(nrow(df3) - periods_to_pred), 1]

observed_sequence_1 <- observed_sequence_1 |>
  append(forecasted_store)

observed_sequence_2 <- c(observed_sequence_2[-seq_len(1)], observed_sequence_2[seq_len(1)])  

input_sequence <- array(
    data = as.numeric(unlist(c(observed_sequence_1, observed_sequence_2))),
    dim = c(
        1,
        periods_to_pred,
        2
    )
)

forecasted_value <- lstm_model |>
    predict(input_sequence)

forecasted_store <- forecasted_store |>
  append(forecasted_value)
}


# reversing the scaling
forecasted_store2 <- forecasted_store * scale_factors_sales[2] + scale_factors_sales[1]


# creating a dataframe to store results
results <- data.frame(date = df3[(nrow(df3) - periods_to_pred + 1):(nrow(df3)), 1],
                      sales = df3[(nrow(df3) - periods_to_pred + 1):(nrow(df3)), 2],
                      forecast = forecasted_store2)


# summzarizing the accuracy
accuracy <- results |>
  mutate(MAPE_Inv = (results$sales - results$forecast) / results$sales) |>
  mutate(MAPE_Inv_abs = abs(MAPE_Inv)) |>
  summarize(MAPE_accuracy = 1 - mean(MAPE_Inv_abs))

paste0("The average MAPE accuracy is ", accuracy[1, 1])


# Plot to show predicted sales vs actual sales figure for test set
forecast_analysis <- ggplot() +
  geom_line(data = results, aes(x = as.Date(date), y = sales, color = "Actual"), alpha = 0.35) +
  geom_line(data = results, aes(x = as.Date(date), y = forecast, color = "Forecast")) +
  scale_x_date(labels = date_format("%Y-%b"), breaks = date_breaks("2 months")) +
  xlab("Date") +
  ylab("Amount ($)") +
  ggtitle("Forecasted Sales vs Actual Sales") +
  scale_color_manual(values = c("Actual" = "#009E73", "Forecast" = "#CC79A7"), name = "Legend")

forecast_analysis
