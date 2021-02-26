# read data in
nyt1<-read.csv(“nyt1.csv")
# eliminate zeros
nyt1<-nyt1[which(nyt1$Impressions>0 & nyt1$Clicks>0 & nyt1$Age>0),]
## or could just have this: nyt1<-nyt1[which(nyt1$Impressions>0 & nyt1$Age>0),]
nnyt1<-dim(nyt1)[1]
#90% to train
sampling.rate=0.9
#remainder to test
num.test.set.labels=nnyt1*(1.-sampling.rate)
#construct a random set of training indices (training)
training <-sample(1:nnyt1,sampling.rate*nnyt1, replace=FALSE)
#build the training set (train)
train<-subset(nyt1[training,],select=c(Age,Impressions))
#construct the remaining test indices (testing)
testing<-setdiff(1:nnyt1,training)
#define the test set
test<-subset(nyt1[testing,],select=c(Age,Impressions))
#construct labels for another variable (Gender) in the training set
cg<-nyt1$Gender[training]
#construct true labels the other variable in the test set
true.labels<-nyt1$Gender[testing]
#run the classifier, can change k
classif<-knn(train,test,cg,k=5)
#view the classifier
classif
#looks at attriburtes
attributes(.Last.value)

# more in later classes
