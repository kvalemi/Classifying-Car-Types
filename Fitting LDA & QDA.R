library(nnet)   # For logistic regression
library(car)    # For ANOVA after logistic regression with nnet
library(glmnet) # For logistic regression and LASSO
library(MASS)   # For discriminant analysis

setwd('./A10 - 18')
vehdata = read.csv('../vehicle.csv')
vehdata$class = factor(vehdata$class, labels=c('2D', '4D', 'BUS', 'VAN'))

# split the data
set.seed(46685326, kind="Mersenne-Twister")
perm <- sample(x=nrow(vehdata))
set1 <- vehdata[which(perm <= 3*nrow(vehdata)/4), ] 
set2 <- vehdata[which(perm > 3*nrow(vehdata)/4), ]

rownames(set1) = 1:dim(set1)[1]
rownames(set2) = 1:dim(set2)[1]

# 1

rescale <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- min(x2[,col])
    b <- max(x2[,col])
    x1[,col] <- (x1[,col]-a)/(b-a)
  }
  x1
}

### Create copies of our datasets and rescale
data.train.scale = set1
data.valid.scale = set2
data.train.scale[,-19] = rescale(data.train.scale[,-19], set1[,-19])
data.valid.scale[,-19] = rescale(data.valid.scale[,-19], set1[,-19])

Y.train = set1[,19]
Y.valid = set2[,19]

# Fit a logistic Regression
fit.log.nnet = multinom(class ~ ., data = data.train.scale, maxit = 200)

# Fit ANOVA test
Anova(fit.log.nnet) 

# Get the training misclassification rate
training.pred.log.nnet = predict(fit.log.nnet, data.train.scale)
(training.misclass.log.nnet = mean(training.pred.log.nnet != Y.train)) 


# test misclassification rate
test.pred.log.nnet = predict(fit.log.nnet, data.valid.scale)
(test.misclass.log.nnet = mean(test.pred.log.nnet != Y.valid)) 

# confusion matrix
table(Y.valid, test.pred.log.nnet, dnn = c("Observed", "Predicted"))


# 3 Fit a LASSO
X.train.scale = as.matrix(data.train.scale[,-19])
Y.train = data.train.scale[,19]

X.valid.scale = as.matrix(data.valid.scale[,-19])
Y.valid = data.valid.scale[,19]

fit.CV.lasso = cv.glmnet(X.train.scale, Y.train, family="multinomial")

lambda.min = fit.CV.lasso$lambda.min

coef(fit.CV.lasso, s = lambda.min)

# training error rate
tr.pred.lasso.min = predict(fit.CV.lasso, X.train.scale, s = lambda.min, type = "class")
(tr.miss.lasso.min = mean(Y.train != tr.pred.lasso.min))

# test error rate
pred.lasso.min = predict(fit.CV.lasso, X.valid.scale, s = lambda.min, type = "class")
(miss.lasso.min = mean(Y.valid != pred.lasso.min))



# 3

### Rescale x1 using the means and SDs of x2
scale.1 <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- mean(x2[,col])
    b <- sd(x2[,col])
    x1[,col] <- (x1[,col]-a)/b
  }
  x1
}

X.train.DA = scale.1(set1[,-19], set1[,-19])
X.valid.DA = scale.1(set2[,-19], set1[,-19])

Y.train = set1[,19]
Y.valid = set2[,19]

fit.lda = lda(X.train.DA, Y.train)


class.col <- ifelse(set1$class == '2D',y=53,n= ifelse(set1$class == '4D',y=68,n= ifelse(set1$class=='BUS',y=203,n=464)))

plot(fit.lda, col = class.col)

# training misclassification rate
pred.lda.test = predict(fit.lda, X.train.DA)$class
(miss.lda.train = mean(Y.train != pred.lda.test))

# test misclassifcation rate
pred.lda = predict(fit.lda, X.valid.DA)$class
(miss.lda = mean(Y.valid != pred.lda))

# QDA
fit.qda = qda(X.train.DA, Y.train)


# training misclassification rate
pred.lda.test = predict(fit.qda, X.train.DA)$class
(miss.lda.train = mean(Y.train != pred.lda.test))

# test misclassifcation rate
pred.lda = predict(fit.qda, X.valid.DA)$class
(miss.lda = mean(Y.valid != pred.lda))










