library(rpart)        # For fitting classification trees
library(rpart.plot)   # For plotting classification trees
library(randomForest) # For random forests
library(gbm)          # For boosting

vehdata = read.csv('../vehicle.csv')
vehdata$class = factor(vehdata$class, labels=c('2D', '4D', 'BUS', 'VAN'))

set.seed(46685326, kind="Mersenne-Twister")
perm <- sample(x=nrow(vehdata))
set1 <- vehdata[which(perm <= 3*nrow(vehdata)/4), ] 
set2 <- vehdata[which(perm > 3*nrow(vehdata)/4), ]

# 1

# a
set.seed(8646824, kind="Mersenne-Twister")

fit.tree.full = rpart(class ~ ., 
                      data = set1, 
                      method = "class")

prp(fit.tree.full, type = 1, extra = 1, main = "Full Tree")

info.tree = fit.tree.full$cptable

print(info.tree)


# b
set.seed(8646824, kind="Mersenne-Twister")

fit.tree.full = rpart(class ~ ., 
                      data = set1, 
                      method = "class",
                      cp = 0)

prp(fit.tree.full, type = 1, extra = 1, main = "Full Tree")

info.tree = fit.tree.full$cptable

print(info.tree)


# c

# Find location of minimum error
minrow <- which.min(info.tree[,4])
# Take geometric mean of cp values at min error and one step up 
cplow.min <- info.tree[minrow,1]
cpup.min <- ifelse(minrow==1, yes=1, no=info.tree[minrow-1,1])
cp.min <- sqrt(cplow.min*cpup.min)

# Find smallest row where error is below +1SE
se.row <- min(which(info.tree[,4] < info.tree[minrow,4]+info.tree[minrow,5]))
# Take geometric mean of cp values at min error and one step up 
cplow.1se <- info.tree[se.row,1]
cpup.1se <- ifelse(se.row==1, yes=1, no=info.tree[se.row-1,1])
cp.1se <- sqrt(cplow.1se*cpup.1se)

# Creating a pruned tree using the CV-min rule.
fit.tree.min <- prune(fit.tree.full, cp=cp.min)
# Creating a pruned tree using the CV-1se rule.
fit.tree.1se <- prune(fit.tree.full, cp=cp.1se)

# d
Y.train = set1[,19]
Y.valid = set2[,19]


# default:
set.seed(8646824, kind="Mersenne-Twister")
fit.tree.full = rpart(class ~ ., 
                      data = set1, 
                      method = "class")

pred.tree.default = predict(fit.tree.full, set1, type = "class")
(mis.tree.min = mean(Y.train != pred.tree.default))

pred.tree.default = predict(fit.tree.full, set2, type = "class")
(mis.tree.min = mean(Y.valid != pred.tree.default))

# min-CV
pred.tree.default = predict(fit.tree.min, set1, type = "class")
(mis.tree.min = mean(Y.train != pred.tree.default))

pred.tree.default = predict(fit.tree.min, set2, type = "class")
(mis.tree.min = mean(Y.valid != pred.tree.default))


# 1SE
pred.tree.default = predict(fit.tree.1se, set1, type = "class")
(mis.tree.min = mean(Y.train != pred.tree.default))

pred.tree.default = predict(fit.tree.1se, set2, type = "class")
(mis.tree.min = mean(Y.valid != pred.tree.default))


# Most Optimal KNN(5):  0.25

# Logistic Regression:  0.13
# LR LASSO:             0.14

# LDA:                  0.17
# QDA:                  0.13

# No PC, Kernel         0.36
# No PC, Normal         0.58
# PC, Kernel            0.20
# PC, Normal            0.22

# Full/Default          0.31
# Min-CV Tree           0.32
# 1SE Tree              0.34

# Default RF            0.28
# Optimal RF            0.26

# 2
# Starting with default, no real reason
default.rf <- randomForest(data = set1, 
                           class ~ ., 
                           importance=TRUE, 
                           keep.forest=TRUE)

round(importance(default.rf),3) # Print out importance measures
x11(h=7,w=15)
varImpPlot(default.rf) # Plot of importance measures; more interesting with more variables


pred.rf.default = predict(default.rf, set1, type = "class")
(mean(set1[,19] != pred.rf.default))

pred.rf.default = predict(default.rf, set2, type = "class")
(mean(set2[,19] != pred.rf.default))

# 3

### Split the data into training and validation sets
data.train.rf = set1
data.valid.rf = set2
Y.train.rf = data.train.rf$class
Y.valid.rf = data.valid.rf$class

### Set tuning parameters
all.mtrys = c(2,4,6,10,18)
all.nodesizes = c(1,3,5,7,10)
all.pars.rf = expand.grid(mtry = all.mtrys, nodesize = all.nodesizes)
n.pars = nrow(all.pars.rf)

M = 5 # Number of times to repeat RF fitting. I.e. Number of OOB errors

### Container to store OOB errors. This will be easier to read if we name
### the columns.
# all.OOB.rf = array(0, dim = c(M, n.pars))
names.pars = apply(all.pars.rf, 1, paste0, collapse = "-")
colnames(all.OOB.rf) = names.pars
all.OOB.rf = array(0, dim = c(M, n.pars))

for(i in 1:n.pars){
  ### Progress update
  print(paste0(i, " of ", n.pars))
  
  ### Get tuning parameters for this iteration
  this.mtry = all.pars.rf[i, "mtry"]
  this.nodesize = all.pars.rf[i, "nodesize"]
  
  for(j in 1:M){
    ### Fit RF, then get and store OOB errors
    this.fit.rf = randomForest(class ~ ., 
                               ntree = 500,
                               data = data.train.rf,
                               mtry = this.mtry, 
                               nodesize = this.nodesize)
    
    pred.this.rf = predict(this.fit.rf)
    this.err.rf = mean(Y.train.rf != pred.this.rf)
    
    all.OOB.rf[j, i] = this.err.rf
  }
}

names.pars = apply(all.pars.rf, 1, paste0, collapse = "-")
colnames(all.OOB.rf) = names.pars
print(all.OOB.rf)

# Relative boxplot
rel.OOB.rf = apply(all.OOB.rf, 1, function(W) W/min(W))
boxplot(t(rel.OOB.rf), 
        las=2,
        main = "Relative OOB Boxplot") 



# most optimal RF parameter set:
all.mtrys = c(4)
all.nodesizes = c(3)
all.pars.rf = expand.grid(mtry = all.mtrys, nodesize = all.nodesizes)
n.pars = nrow(all.pars.rf)

M = 5 # Number of times to repeat RF fitting. I.e. Number of OOB errors

### Container to store OOB errors. This will be easier to read if we name
### the columns.
# all.OOB.rf = array(0, dim = c(M, n.pars))
names.pars = apply(all.pars.rf, 1, paste0, collapse = "-")
colnames(all.OOB.rf) = names.pars
all.OOB.rf = array(0, dim = c(M, n.pars))

for(i in 1:n.pars){
  ### Progress update
  print(paste0(i, " of ", n.pars))
  
  ### Get tuning parameters for this iteration
  this.mtry = all.pars.rf[i, "mtry"]
  this.nodesize = all.pars.rf[i, "nodesize"]
  
  for(j in 1:M){
    ### Fit RF, then get and store OOB errors
    this.fit.rf = randomForest(class ~ .,
                               data = data.train.rf,
                               mtry = this.mtry, 
                               nodesize = this.nodesize)
    
    pred.this.rf = predict(this.fit.rf)
    this.err.rf = mean(Y.train.rf != pred.this.rf)
    
    all.OOB.rf[j, i] = this.err.rf
  }
}

pred.rf.optimal = predict(this.fit.rf, set2, type = "class")
(mean(set2[,19] != pred.rf.optimal))












