load('Rdata/processed_text.RData')
train_file <- read.csv("data/tfidf.csv",header = FALSE)

train_x = train_file[1:9998,]
test_x = train_file[9999:15998,]

idx = sample(nrow(train_x), floor(nrow(train_x) * 0.7))

Xtrain = as.matrix(train_x[idx,])
Xval = as.matrix(train_x[-idx,])
Ytra = as.matrix(Ytrain[idx])
Yval = as.matrix(Ytrain[-idx])
Xtest = as.matrix(test_x)

rm(train_x, test_x)

library(keras)
install_keras()
set.seed(3)

response <- function(Ytrain){
    Ytrain[Ytrain == ' documentary '] = 0
    Ytrain[Ytrain == ' drama '] = 1
    Ytrain[Ytrain == ' short '] = 2
    Ytrain[Ytrain == ' comedy '] = 3
    return(factor(Ytrain))
}

train_labels <- to_categorical(response(Ytra))
val_labels <- to_categorical(response(Yval))


rm(model)
units = c(512, 256, 128)
layers = c(2,4,6,8)

model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", input_shape = c(1*904)) %>%
    layer_dense(units = 4, activation = "softmax")

model

model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = c('accuracy')
)

history <- model %>% fit(
    Xtrain,
    train_labels,
    epochs = 10,
    batch = 64,
    validation_data = list(Xval, val_labels)
)

plot(history)

pred <- model %>% predict(Xtest)
pred1 = rep(NA, nrow(pred))
pred1[pred[,1] == apply(pred, 1, max)] = ' documentary '
pred1[pred[,2] == apply(pred, 1, max)] = ' drama '
pred1[pred[,3] == apply(pred, 1, max)] = ' short '
pred1[pred[,4] == apply(pred, 1, max)] = ' comedy '

out.df <- data.frame(id = testID, genre = pred1)
colnames(out.df) <- c('id', 'genre')
write.csv(out.df, file = "NN.csv", row.names = FALSE)
# 0.61550
