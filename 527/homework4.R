# 1

source('/Users/stewart/projects/stats/527/meatspec-train-test.R')

resubstitution_error = mean((lm(fat~., data = train)$fitted.values - train$fat)^2)
print(paste0('Average squared resubstitution error ', resubstitution_error))

gcv_estimate = mean(((lm(fat~., data = train)$fitted.values - train$fat) / (1 - (100 / nrow(train))))^2)
print(paste0('GCV Estimated average squared resubstitution error ', gcv_estimate))

X = as.matrix(train[,1:100])
require(MASS)
H = X %*% ginv(t(X) %*% X) %*% t(X)
cv_estimate = mean(((lm(fat~., data = train)$fitted.values - train$fat) / (1 - diag(H)))^2)
print(paste0('CV Estimated average squared resubstitution error ', cv_estimate))

test_error = mean((predict(lm(fat~., data = train), test[,1:100]) - test$fat) ^ 2)
print(paste0('Test Error ', test_error))

# 2 
require('leaps')
forward.select.features <- function(X, Y, nterm, selection_method="forward") {
  features = summary(
    regsubsets(x=X, y=Y, method=selection_method, nvmax=nterm, intercept = T)
  )$which[nterm,];
  return(names(X)[features[2:51]])
}
train_ = matrix(0, nrow=nrow(train), ncol=51)
test_ = matrix(0, nrow=nrow(test), ncol=51)
cols = ceiling(1:100 / 2 );
for (i in 1:50) {
  train_[, i] = rowMeans(train[, cols==i]);
  test_[, i] = rowMeans(test[, cols==i]);
}
train_[, 51] = train$fat
test_[, 51] = test$fat
train_ = as.data.frame(train_)
test_ = as.data.frame(test_)
names(train_)[51] = 'fat'
names(test_)[51] = 'fat'

folds <- cut(seq(1,nrow(train_)),breaks=5,labels=FALSE)

best_n_features=  0;
best_error = 'inf'
cv_errors = rep(0, 50);
for (n_features in 1:50) {
  errors = rep(0, 5)
  for (fold in unique(folds)) {
    fold_train = train_[folds != fold,];
    fold_test = train_[folds == fold,];
    features = forward.select.features(fold_train[,1:50], fold_train$fat, n_features)
    model <- lm(paste0('fat ~ ', paste(features, collapse='+')), fold_train)
    mse = mean((predict(model, fold_test) - fold_test$fat) ^2)
    errors[fold] = mse
  }
  cv_errors[n_features] = mean(errors)
  if (mean(errors) < best_error) { 
    best_error = mean(errors);
    best_n_features = n_features;
  }
}
 
plot(1:50, cv_errors) 
print(paste0('Estimate for optimal number of features: ', best_n_features));
features = forward.select.features(train_[,1:50], train_$fat, best_n_features)
mse_train = mean((lm(paste0('fat ~ ', paste(features, collapse='+')), train_)$fitted.values - train_$fat) ^2 )
print(paste0('Resubstitution Error for optimal number of features: ', mse_train))
mse_test = mean((predict(lm(paste0('fat ~ ', paste(features, collapse='+')), train_), test_) - test_$fat) ^2 )
print(paste0('Test Error for optimal number of features: ', mse_test))
