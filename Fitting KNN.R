library(FNN)

# 1
vehdata = read.csv('./vehicle.csv')
summary(vehdata)
vehdata$class = factor(vehdata$class, labels=c('2D', '4D', 'BUS', 'VAN'))
summary(vehdata)

expl_vars = subset(vehdata, select = -c(class))
colnames(expl_vars) = 1:18
corrs = round(cor(expl_vars, method = "pearson"), 2)

upper<-corrs
upper[upper.tri(corrs)]<-' '
upper <- as.vehdata.frame(upper)
upper


for(i in 1:ncol(upper)) {
  
  print(paste0("Variables with a Correlation Greater than 0.90 with variable ", i))
  print((upper[,i] > 0.90))
  cat("\n")
  cat("\n")
  
}



# 2
set.seed(46685326, kind="Mersenne-Twister")
perm <- sample(x=nrow(vehdata))
set1 <- vehdata[which(perm <= 3*nrow(vehdata)/4), ] # TRAINING SET
set2 <- vehdata[which(perm > 3*nrow(vehdata)/4), ]  # TEST SET

rownames(set1) = 1:dim(set1)[1]
rownames(set2) = 1:dim(set2)[1]


# 3

scale.1 <- function(x1,x2){
  for(col in 1:ncol(x1)){
    a <- mean(x2[,col])
    b <- sd(x2[,col])
    x1[,col] <- (x1[,col]-a)/b
  }
  x1
}

# Rescale the data

X.train.raw = set1[,-19]
X.valid.raw = set2[,-19]
Y.train = set1[,19]
Y.valid = set2[,19]

# Rescale
X.train = scale.1(X.train.raw, X.train.raw)
X.valid = scale.1(X.valid.raw, X.train.raw) # Watch the order

pred.knn = knn(X.train, X.valid, Y.train, k = 1)

table(pred.knn, Y.valid, dnn = c("Predicted", "Observed"))

### Next, let's get the misclassification rate
(misclass.knn = mean(pred.knn != Y.valid))


# 4

K.max = 40

mis.CV = rep(0, times = K.max)

for(i in 1:K.max){
  
  print(paste0(i, " of ", K.max))
  
  set.seed(9910314, kind="Mersenne-Twister")
  this.knn = knn.cv(X.train, Y.train, k=i)
  
  this.mis.CV = mean(this.knn != Y.train)
  mis.CV[i] = this.mis.CV
  
}

SE.mis.CV = sapply(mis.CV, function(r){
  sqrt(r*(1-r)/nrow(X.train))
})

plot(1:K.max, mis.CV, xlab = "K", 
     ylab = "Misclassification Rate",
     col = 'blue',
     main = 'Neighbors vs Validation Error Plus SE')


for(i in 1:K.max){
  
  lower = mis.CV[i] - SE.mis.CV[i]
  upper = mis.CV[i] + SE.mis.CV[i]
  
  lines(x = c(i, i), y = c(lower, upper), col = 'orange')
}


k.min = which.min(mis.CV)

thresh = mis.CV[k.min] + SE.mis.CV[k.min]
k.1se = max(which(mis.CV <= thresh))

knn.min = knn(X.train, X.valid, Y.train, k.min)
knn.1se = knn(X.train, X.valid, Y.train, k.1se)

(mis.min = round(mean(Y.valid != knn.min),2))
(mis.1se = round(mean(Y.valid != knn.1se),2))

write.table(mis.CV, file = './KNN_CV_Error_Rates.csv', row.names = FALSE, col.names = FALSE)








