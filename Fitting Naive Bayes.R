library(klaR)   # For naive Bayes

setwd('./A11 - 19')
vehdata = read.csv('../vehicle.csv')
vehdata$class = factor(vehdata$class, labels=c('2D', '4D', 'BUS', 'VAN'))

set.seed(46685326, kind="Mersenne-Twister")
perm <- sample(x=nrow(vehdata))
set1 <- vehdata[which(perm <= 3*nrow(vehdata)/4), ] 
set2 <- vehdata[which(perm > 3*nrow(vehdata)/4), ]

X.train = set1[,-19]
Y.train = set1[,19]

X.valid = set2[,-19]
Y.valid = set2[,19]

# 1

fit.NB = NaiveBayes(X.train, Y.train, usekernel = T)

par(mfrow = c(2,2)) # Set plotting to 2x3
plot(fit.NB)



# 2

# No PC + Kernel
fit.NB.kernel = NaiveBayes(X.train, Y.train, usekernel = T)

(mean(Y.train != predict(fit.NB.kernel, X.train)$class))
(mean(Y.valid != predict(fit.NB.kernel, X.valid)$class))

# No PC + Normal
fit.NB.normal = NaiveBayes(X.train, Y.train, usekernel = F)

(mean(Y.train != predict(fit.NB.normal, X.train)$class))
(mean(Y.valid != predict(fit.NB.normal, X.valid)$class))

# PC Analysis
fit.PCA = prcomp(X.train, scale. = T)
X.train.PC = fit.PCA$x
X.valid.PC = predict(fit.PCA, set2)

# PC + Kernel
fit.NB.PC.kernel = NaiveBayes(X.train.PC, Y.train, usekernel = T)

(mean(Y.train != predict(fit.NB.PC.kernel, X.train.PC)$class))
(mean(Y.valid != predict(fit.NB.PC.kernel, X.valid.PC)$class))

# PC + Normal
fit.NB.PC.normal = NaiveBayes(X.train.PC, Y.train, usekernel = F)

(mean(Y.train != predict(fit.NB.PC.normal, X.train.PC)$class))
(mean(Y.valid != predict(fit.NB.PC.normal, X.valid.PC)$class))





