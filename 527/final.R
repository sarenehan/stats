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


### 6
# a)
K = 10;
knn.classifier <- function(X.train, y.train, X.test, k.try = 1, pi = rep(1/K,K), CV = F){
  classes = sort(unique(y.train));
  X.train = as.matrix(X.train);
  X.test = as.matrix(X.test);
  n.is = c();
  for (class in classes) {
    n.is = c(n.is, sum(y.train == class));
  }
  output = matrix(nrow=nrow(X.test), ncol=length(k.try));
  for (row.test in 1:(nrow(X.test))) {
    print(row.test);
    distances = rowSums(sweep(X.train, 2, as.vector(X.test[row.test,])) ^ 2)
    if (CV == T) {
      distances[row.test] = max(distances);
    }
    preds = y.train[sort(distances, index.return=TRUE)$ix];
    for (k.try.idx in 1:length(k.try)) {
      knns = preds[1:k.try[k.try.idx]];
      max.n.occurences = 0;
      best.prediction = -1;
      for (pred in unique(knns)) {
        pred.idx = match(pred, classes)
        n_occurences = sum(knns == pred) * pi[pred.idx] / n.is[pred.idx];
        if (n_occurences > max.n.occurences) {
          best.prediction = pred;
          max.n.occurences = n_occurences;
        }
      }
      output[row.test, k.try.idx] = best.prediction;
    }
  }
  return(output);
}

# b
library(datasets);
data(iris);
X.train = iris[,names(iris) != 'Species']
X.test = X.train
y.train = iris$Species
K = length(unique(y.train))
pi = c();
for (cls in sort(unique(y.train))) {
  pi = c(pi, sum(y.train == cls) / nrow(iris));
}
knn.nocv = knn.classifier(X.train, y.train, X.test, k.try = 5, pi = pi, CV = F);
knn.cv = knn.classifier(X.train, y.train, X.test, k.try = 1, pi = pi, CV = T);

print(paste0('Iris No CV misclassifications (out of 150 predictions): ', sum(knn.nocv != y.train)))
print(paste0('Iris CV misclassifications (out of 150 predictions): ', sum(knn.cv != y.train)))

# c
k.try = c(1, 3, 7, 11, 15, 21, 27, 35, 43);
train = read.table('/Users/stewart/Downloads/zip-train.dat');
X.train = train[,2:257]
X.test = X.train
y.train = train[,1]
pi = c();
for (cls in sort(unique(y.train))) {
  pi = c(pi, sum(y.train == cls) / nrow(train));
}
knn.out = knn.classifier(X.train, y.train, X.test, k.try = k.try, pi = pi, CV = T);

best.k = 0;
best.error = nrow(X.train);
for (k.idx in 1:length(k.try)) {
  preds = knn.out[,k.idx];
  n_errors = sum(preds != y.train);
  print(paste0('N errors for k = ', k.try[k.idx], ': ', n_errors));
  if (n_errors < best.error) {
    best.k = k.try[k.idx];
    best.error = n_errors;
  }
}
print(paste0('Optimal k: ', best.k));
#d 
test = read.table('/Users/stewart/Downloads/zip-test.dat')
test.preds = knn.classifier(X.train, y.train, test[,2:257], k.try=best.k, pi=pi, CV=F);

test.error = sum(test.preds != test[, 1]) / nrow(test);
print(paste0('Estimated Test Error Rate ', test.error));


###
sub = matrix(as.vector(X.train[1,]),nrow=nrow(X.train),ncol=length(v),byrow=TRUE)
library(dplyr)
diff = setdiff(as.matrix(X.train), sub)

