spam.data = read.table('/Users/stewart/Downloads/spam.data')
traintest = read.table('/Users/stewart/Downloads/spam.traintest')

test = spam.data[traintest== 1,]
train = spam.data[traintest==0,]
# install.packages('rpart')
require(rpart)
# help(rpart)
# names(spam.data)
model = rpart(V58~., data = train, method="class", cp=0, minsplit=0)
plot(model)
text(model)

train.error = sum((predict(model, train)[,1] == 0) != train$V58) / nrow(train)
test.error = sum((predict(model, test)[,1] == 0) != test$V58) / nrow(test)
print(paste0('Train Error Rate: ', train.error));
print(paste0('Test Error Rate: ', test.error));
## 2b

folds <- cut(sample(seq(1,nrow(train))),breaks=10,labels=FALSE)
ct.tests = c(0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1);
errors = c();
for (lambda in ct.tests) {
  val.errors = c();
  for (fold in 1:10) {
    model =  rpart(V58~., data = train[folds !=fold,], method='class', cp=lambda, minsplit=0);
    val.error = sum((predict(model, train[fold == fold,])[,1] == 0) != train[fold == fold,]$V58) / nrow(train[fold == fold,]);
    val.errors = c(val.errors, val.error);
  } 
  errors = c(errors, mean(val.errors))
  print(mean(val.errors))
}
plot(log(ct.tests, base=10), errors, ylab='CV Misclassification Error', main='CV(lambda)');
print(paste0('Optimal lambda', 0.0001))
model.optimal = rpart(V58~., data = train, method='class', cp=0.0001, minsplit=0)
plot(model.optimal)
train.error = sum((predict(model.optimal, train)[,1] == 0) != train$V58) / nrow(train)
test.error = sum((predict(model.optimal, test)[,1] == 0) != test$V58) / nrow(test) 
print(paste0('Train Error Rate: ', train.error));
print(paste0('Test Error Rate: ', test.error));

## 3
require(glmnet)

model = cv.glmnet(as.matrix(train[,1:57]), train$V58, nfolds=10, alpha=0, lambda.min.ratio=0)
print(paste0('Optimal Lambda: ', model$lambda.1se))

preds = predict(model, as.matrix(train[,1:57]), s="lambda.1se")

best.c <- 0
best.error <- 'inf'
for (pred in preds) {
  err = (sum((preds > pred) != train$V58));
  if (pred < best.error) {
    best.error = err;
    best.c = pred;
  } 
}
print(paste0('Optimal value for C: ', best.c))

train.error = best.error / nrow(train);
test.error =sum((predict(model, newx=as.matrix(test[,1:57]), s="lambda.1se") > best.c) != test$V58) / nrow(test)
print(paste0('Resubstitution Error on Training Data ', train.error))
print(paste0('Resubstitution Error on Testing Data ', test.error))

ls.model = lm('V58~.', data=train)
preds = predict(ls.model, data=train)
best.c <- 0
best.error <- 'inf'
for (pred in preds) {
  err = (sum((preds > pred) != train$V58));
  if (pred < best.error) {
    best.error = err;
    best.c = pred;
  } 
}
print(paste0('Optimal value for C for least squares model: ', best.c))
train.error = best.error / nrow(train);
test.error = sum((predict(ls.model, newdata=test) > best.c) != test$V58) / nrow(test)
print(paste0('Least Squares Resubstitution Error on Training Data ', train.error))
print(paste0('Least Squares Resubstitution Error on Testing Data ', test.error))

# 4 
# a)
# install.packages('ISLR')
require('ISLR')
data('College')
sample <- sample.int(n = nrow(College), size = floor(.8*nrow(College)), replace = F)
train <- College[sample, ]
test  <- College[-sample, ]

require('leaps')

folds <- cut(sample(seq(1,nrow(train))),breaks=10,labels=FALSE)
best.n.features = 0;
best.error = 'inf'
possible_features = c();
for (name in names(train)) {
  if (name != 'Outstate') {
    possible_features = c(possible_features, name);
  }
}

for (n_features in 1:(ncol(train) - 1)) {
  errors = c();
  for (fold in 1:10) {
    optimal_features = summary(
      regsubsets(Outstate~., data=train[folds != fold,], method='forward', nvmax=n_features, intercept = F)
    )$which[n_features,];
    model = lm(as.formula(paste("Outstate~", paste(possible_features[optimal_features], collapse="+"))), data=train[folds != fold,]);
    preds = predict(model, newdata=train[folds == fold,]) 
    mse = mean((preds - train[folds == fold, 'Outstate'])^2)
    errors= c(errors, mse);
  }
  print(mean(errors));
  if (mean(errors) < best.error) {
    best.error = mean(errors);
    best.n.features = n_features;
  }
}
print(paste0('Best n features using forward selection w/ 10 fold cv: ', best.n.features));
optimal_features = summary(
  regsubsets(Outstate~., data=train, method='forward', nvmax=best.n.features, intercept = F)
)$which[best.n.features,];
print('Optimal features: ')
print(possible_features[optimal_features])

# b
model = lm(as.formula(paste("Outstate~", paste(possible_features[optimal_features], collapse="+"))), data=train);

plot(predict(model, train), train$Outstate, xlab = 'Prediction', ylab='Truth', main='Predictions vs Truth')

# c
rmse = mean((predict(model, test) - test$Outstate)^2) ^.5
print(paste0('On the test data, the RMSE is ', rmse, ' so the predictions are usually within this far of the true on testing data.'))
