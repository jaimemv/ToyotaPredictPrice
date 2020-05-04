library(dplyr)
library(keras)
library(ggplot2)

car.df <- read.csv("/Users/Jaime/RProjects/ToyotaCorolla.csv")
selected.var <- c(3, 4, 7, 9, 24, 25, 27, 29)
train.index <- sample(c(1:1436), 1100)
car.df$Price <- car.df$Price / 1000
train.df <- car.df[train.index, selected.var]
test.df <- car.df[-train.index, selected.var]
train_labels <- train.df$Price
test_labels <- test.df$Price
train.df$Price <- NULL
test.df$Price <- NULL
train.df <- data.matrix(train.df)
test.df <- data.matrix(test.df)


rm(selected.var, train.index)

mean <- apply(train.df, 2, mean)
std <- apply(train.df, 2, sd)
train.df <- scale(train.df, center = mean, scale = std)
test.df <- scale(test.df, center = mean, scale = std)

rm(mean, std, car.df)

build_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(train.df)[[2]]) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )
}

model <- build_model()
model %>% summary()


epochs <- 500
# 2.3) Fit the model and store training stats
history <- model %>% fit(
  train.df,
  train_labels,
  epochs = epochs,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list()
)

plot(history, optimizer = "rmsprop", smooth = FALSE) +
  coord_cartesian(ylim = c(0, 5))

c(loss, mae) %<-% (model %>% evaluate(test.df, test_labels, verbose = 0))
paste0("Mean absolute error on test set: $", sprintf("%.2f", mae * 1000))
