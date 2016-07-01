# Sentiment analysis for Hubspot reviews

library(tm)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(graph)
library(Rgraphviz)
library(FactoMineR)
library(topicmodels)
library(stringr)
library(RTextTools)
library(e1071)

## Loading data

reviews <- read.csv('Data/hubspot_reviews3.csv', stringsAsFactors = FALSE)

## Step 1. Exploring and preparing the data

reviews$Date <- as.Date(reviews$Date, "%m/%d/%Y")
str(reviews$Rating)
table(reviews$Rating)

### Example of the data
reviews[1,3]

reviewers.score = c(1,0,1,0,2,3,12,36,98,306)
reviewers.df = data.frame(x = 1:10, y = reviewers.score)

hist(reviews$Rating, breaks = 10, col = "#2171b5", main =
"Distribution of reviews' rating scores\n assigned by reviewers", 
xlab = "Rating score of the review", ylab ="Number of reviews")

#qplot(reviewers.score, main = "Distribution of reviews' rating 
#      scores assigned by reviewers",
#      xlab = "Rating score of the review", ylab ="Number of reviews",
#      xlim = range(1:10),  ylim = range(1:310),
 #     fill=I("#2171b5"), binwidth = 0.5,
 #     col=I("darkgrey"))+theme_few()


### Data preparation -- cleaning and standardizing text data

myCorpus <- Corpus(VectorSource(reviews$Review)) # build a corpus

myCorpus <- myCorpus %>% 
  tm_map(content_transformer(tolower))%>% # convert to lower case
  tm_map(removePunctuation)%>% # remove punctuation
  tm_map(removeNumbers)%>% # remove numbers
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(stemDocument, lazy=TRUE)%>%
  tm_map(stripWhitespace, lazy=TRUE) # remove extra whitespaces

### Term-document matrix with tf-idf weighting

tdm <- TermDocumentMatrix(myCorpus, control = list(weighting =
              function(x) weightTfIdf(x, normalize = TRUE)))
tdm

tdm_tf <- TermDocumentMatrix(myCorpus)

### Keywords

lowfreq = 400
keywords <- findFreqTerms(tdm_tf, lowfreq)
term.freq <- rowSums(as.matrix(tdm_tf))
term.freq <- subset(term.freq, term.freq >= lowfreq)
df <- data.frame(term = names(term.freq), freq = term.freq)
df <- plyr::arrange(df, freq)

reorder_size <- function(x) {
  factor(x, levels = x)}

g <- ggplot(df, aes(x = reorder_size(df$term), y = freq)) + 
  geom_bar(stat = "identity", fill = "#2171b5") + xlab("Terms") +
  ylab("Count") + coord_flip() + theme_few() + 
  ggtitle("Top frequent words in the set of reviews")
g

### Associations
vtxcnt <- rowSums(cor(as.matrix(t(tdm_tf[keywords,])))>.225)-1
mycols<-c("#f7fbff","#deebf7","#c6dbef",
          "#9ecae1","#4292c6","#6baed6","#6baed6","#4292c6",
          "#2171b5", "#084594")
vc <- mycols[vtxcnt+1]
names(vc) <- names(vtxcnt)


corThreshold = 0.225
p <- plot(tdm_tf, term = keywords, corThreshold, weighting = T,
          nodeAttrs=list(fillcolor=vc))
p

### clusters of documents: Hierarchical Clustering on Principle Components (HCPC)

dtm_tf <- DocumentTermMatrix(myCorpus)# Document Term Matrix of reviews with term frequancy weight
dtm_tf_matr <- as.matrix(dtm_tf[,keywords[1:10]])


test_sample_review <- sample(1:459, 45)
res.pca <- PCA(dtm_tf_matr)  
res.hcpc <- HCPC(res.pca, kk = 30)
res.hcpc$data.clust

res.hcpc$data.clust <- res.hcpc$data.clust %>% tbl_dt() %>% group_by(clust) 
res.hcpc$data.clust %>% summarise(a = n())


#distMatrix <- dist(scale(dtm_tf_matr))
#fit <- hclust(distMatrix, method = "ward.D2")
#plot(fit, labels = NULL)
#rect.hclust(fit, k = 3, border = c("blue",
 #                             "darkgrey", "darkblue"))

### Topic modelling

lda <- LDA(dtm_tf, method = "Gibbs", k = 3) # is used to estimate and fit a latent dirichlet allocation model with the VEM algorithm (default) or Gibbs sampling; k - number of topics to be found.
term <- terms(lda, 8) # first 8 terms of every topic
term <- apply(term, MARGIN = 2, paste, collapse = ", ")
term

topic <- topics(lda, 1)
topics <- data.frame(date=reviews$Date, topic = res.hcpc$data.clust$clust)
levels(topics$topic) = c("Topic 1","Topic 2","Topic 3")

topics$topic <- factor(topics$topic,levels=c(1,2,3),
                      labels=c("Topic1","Topic2","Topic3"))

qplot(date, ..count..*100, data=topics, geom = "density", alpha=I(.5),
      fill=topic, main = "How topics of reviews changed over time",
      xlab="Date", ylab="Count of reviews corresponding to each topic")+theme_few()

## Sentiment Analysis, lexicon-based approach

### Loanding Hu and Liu vocabulary
hu.liu.pos <- scan('data/positive-words.txt',
                   what = 'character', comment.char = ';')

hu.liu.neg <- scan('data/negative-words.txt',
                   what = 'character', comment.char = ';')

### Score function

hu.liu.pos <- scan('data/positive-words.txt',
                   what = 'character', comment.char = ';')

hu.liu.neg <- scan('data/negative-words.txt',
                   what = 'character', comment.char = ';')

# Not normalized sentiment score

score.sentiment = function(sentences, pos.words, neg.words){
  scores = plyr::laply(sentences, function(sentence, pos.words, neg.words) {
    word.list = str_split(sentence, '\\s+') # split into words
    words = unlist(word.list)
    # compare our words to the dictionaries of positive & negative terms
    pos.matches = !is.na(match(words, pos.words))# match() returns the position of the matched term or NA
    neg.matches = !is.na(match(words, neg.words))
    score = sum(pos.matches) - sum(neg.matches)
  }, pos.words, neg.words)
  return(scores)
}

rating <- score.sentiment(reviews$Review, hu.liu.pos, hu.liu.neg)
rating_max = max(rating)
rating_min = min(rating)
for (i in 1:459){
  rating[i]=(rating[i]-rating_min)*9/(rating_max-rating_min)+1
}
summary(rating)

sse = 0 # sum of square errors
for (i in 1:459){
  sse = sse + (rating[i]- reviews$Rating[i])^2
}
see

# Normalized sentiment score by review length

score.sentiment_nor = function(sentences, pos.words, neg.words){
  scores = plyr::laply(sentences, function(sentence, pos.words, neg.words) {
    word.list = str_split(sentence, '\\s+') # split into words
    words = unlist(word.list)
    l = length(words)
    # compare our words to the dictionaries of positive & negative terms
    pos.matches = !is.na(match(words, pos.words))# match() returns the position of the matched term or NA
    neg.matches = !is.na(match(words, neg.words))
    score = sum(pos.matches) - sum(neg.matches)
    score = score/l
  }, pos.words, neg.words)
  return(scores)
}


rating_nor <- score.sentiment_nor(reviews$Review, hu.liu.pos, hu.liu.neg)
rating_nor_max = max(rating_nor)
rating_nor_min = min(rating_nor)
for (i in 1:459){
  rating_nor[i]=(rating_nor[i]-rating_nor_min)*9/(rating_nor_max-rating_nor_min)+1
}
summary(rating_nor)

sse_nor = 0 # sum of square errors
for (i in 1:459){
  sse_nor = sse_nor + (rating_nor[i]- reviews$Rating[i])^2
}
sse_nor

# Plotting of sentiments

### NORMAL lexicon-based approach (10 numerical levels)

qplot(rating_nor_round, main = "Distribution of reviews' sentiment classes\n by lexicon-based approach (normal formulation)",
      xlab = "Class of the review", ylab ="Number of reviews",
      xlim = c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"),  binwidth=0.5, 
      fill=I("#2171b5"), 
      col=I("darkgrey"))+theme_few()

### Basic lexicon-based approach (3 levels: negative, neutral, positive)

rating_basic <- vector()
for (i in 1:459){
  if (rating_nor[i] <= 3){rating_basic[i] = "negative"}
  else if (rating_nor[i] <= 6) {rating_basic[i] = "neutral"}
  else {rating_basic[i] = "positive"}
}

### BINARY lexicon-based approach (2 levels: positive and negative)

rating_binary <- vector()
for (i in 1:459){
  if (rating_nor_round[i] < 5){rating_binary[i] = "negative"}
  else {rating_binary[i] = "positive"}
}

n=0
for (i in 1:459){
if (rating_binary[i] == "positive"){n = n+1}
}


height_bin = c(rating_nor_round, rating_basic, rating_binary)
facet_level =c(rep ("NORMAL formulation", 459),
               rep ("BASIC formulation", 459),
               rep ("BINARY formulation", 459))
hist_df = data.frame (height_bin, facet_level)

g <- ggplot(hist_df, aes(height_bin, colour = I("#2171b5")))
g+ theme_few() + geom_bar()+ 
  facet_grid(. ~ facet_level, scales = "free") +
  ylab("Number of reviews") + xlab("Class of the review") +
  ggtitle("Distribution of reviews' sentiment classes\n by lexicon-based approach")
          

## Accuracy of the approach
# Let us evaluate the accuracy of each implementation of the approach.

# NORMAL formulation

table(reviews$Rating, rating_nor_round)
RTextTools::recall_accuracy(reviews$Rating, rating_nor_round)

# BASIC formulation

reviews$Rating_basic <- vector()
for (i in 1:459){
  if (reviews$Rating[i] <= 3){reviews$Rating_basic[i] = "negative"}
  else if (reviews$Rating[i] <= 6) {reviews$Rating_basic[i] = "neutral"}
  else {reviews$Rating_basic[i] = "positive"}
}

table(reviews$Rating_basic, rating_basic)
RTextTools::recall_accuracy(reviews$Rating_basic, rating_basic)


# BINARY formulation

reviews$Rating_binary <- vector()
for (i in 1:459){
  if (reviews$Rating[i] < 5){reviews$Rating_binary[i] = "negative"}
  else {reviews$Rating_binary[i] = "positive"}
}

table(reviews$Rating_binary, rating_binary)
RTextTools::recall_accuracy(reviews$Rating_binary, rating_binary)

#----------------------------------------------------------------

# Naive Bayes classifyer 

library(RTextTools)
library(e1071)

# NORMAL implementation
## Data preparation -- creating training and test datasets

set.seed(123)

train_sample <- sample(459, 345)
reviews$Rating <- as.factor(reviews$Rating)

train_labels <- reviews$Rating[train_sample]
test_labels <- reviews$Rating[-train_sample]

prop.table(table(train_labels))
prop.table(table(test_labels))

freq_words <- findFreqTerms(dtm_tf, 5)
dtm_train<- dtm_tf[train_sample, freq_words]
dtm_test <- dtm_tf[-train_sample, freq_words]

convert_counts <- function(x) {
  x <- ifelse(x > 0, 'Yes', 'No')}

train <- apply(dtm_train, MARGIN = 2, convert_counts)
test <- apply(dtm_test, MARGIN = 2, convert_counts)

hub_classifier <- naiveBayes(train, train_labels) #Training the model
test_pred <- predict(hub_classifier, test)    
table(test_pred, test_labels)
recall_accuracy(test_pred, test_labels)

# BASIC implementation

set.seed(234)

train_sample <- sample(459, 345)
reviews$Rating_basic <- as.factor(reviews$Rating_basic)

train_labels <- reviews$Rating_basic[train_sample]
test_labels <- reviews$Rating_basic[-train_sample]

freq_words <- findFreqTerms(dtm_tf, 5)
dtm_train<- dtm_tf[train_sample, freq_words]
dtm_test <- dtm_tf[-train_sample, freq_words]

convert_counts <- function(x) {
  x <- ifelse(x > 0, 'Yes', 'No')}

train <- apply(dtm_train, MARGIN = 2, convert_counts)
test <- apply(dtm_test, MARGIN = 2, convert_counts)

hub_classifier <- naiveBayes(train, train_labels) # train a model
test_pred <- predict(hub_classifier, test)    

table(test_pred, test_labels)
recall_accuracy(test_pred, test_labels)

# BINARY implementation

set.seed(345)

train_sample <- sample(459, 345)
reviews$Rating_binary <- as.factor(reviews$Rating_binary)

train_labels <- reviews$Rating_binary[train_sample]
test_labels <- reviews$Rating_binary[-train_sample]

freq_words <- findFreqTerms(dtm_tf, 6)
dtm_train<- dtm_tf[train_sample, freq_words]
dtm_test <- dtm_tf[-train_sample, freq_words]

train <- apply(dtm_train, MARGIN = 2, convert_counts)
test <- apply(dtm_test, MARGIN = 2, convert_counts)

classifier <- naiveBayes(train, train_labels) # train a model
test_pred <- predict(classifier, test)    

table(test_pred, test_labels)
recall_accuracy(test_pred, test_labels)

#-----------------------------------------------------------------
# RTextTools
## SVM", "MAXENT", "RF", "TREE", "NNET" models

### NORMAL implementation

# build the data to specify response variable, training set, testing set.

set.seed(1)

train_sample <- sample(459, 345)
test_sample <- 1:459
test_sample <- test_sample[-train_sample]

container = create_container(dtm_tf, reviews$Rating,
                             trainSize=train_sample, testSize=test_sample,virgin=FALSE)

#train the model with multiple machine learning algorithms:
models = train_models(container, algorithms=c("SVM", "MAXENT", "RF", "TREE", "NNET"))

#classify the testing set using the trained models.
results = classify_models(container, models)


# accuracy table
test_labels <- reviews$Rating[-train_sample]
table(test_labels, results[,"SVM_LABEL"]) 
table(test_labels, results[,"MAXENTROPY_LABEL"])
table(test_labels, results[,"FORESTS_LABEL"]) 
table(test_labels, results[,"TREE_LABEL"]) 
table(test_labels, results[,"NNETWORK_LABEL"])

# recall accuracy
recall_accuracy(test_labels, results[,"SVM_LABEL"]) # 0.7192982
recall_accuracy(test_labels, results[,"MAXENTROPY_LABEL"]) #0.5877193
recall_accuracy(test_labels, results[,"FORESTS_LABEL"]) #0.7017544
recall_accuracy(test_labels, results[,"TREE_LABEL"]) #0.5964912
recall_accuracy(test_labels, results[,"NNETWORK_LABEL"]) #0.6315789



### BASIC implementation

# build the data to specify response variable, training set, testing set.

set.seed(1)

train_sample <- sample(459, 345)
test_sample <- 1:459
test_sample <- test_sample[-train_sample]

for (i in 1:459){
  if (reviews$Rating[i] <= 3){reviews$Rating_basic[i] = "negative"}
  else if (reviews$Rating[i] <= 6) {reviews$Rating_basic[i] = "neutral"}
  else {reviews$Rating_basic[i] = "positive"}
}

container = create_container(dtm_tf, reviews$Rating_basic,
                             trainSize=train_sample, testSize=test_sample,virgin=FALSE)

#train the model with multiple machine learning algorithms:
models = train_models(container, algorithms=c("SVM", "MAXENT", "RF", "TREE", "NNET"))

#classify the testing set using the trained models.
results = classify_models(container, models)


# accuracy table
test_labels <- reviews$Rating_basic[-train_sample]
table(test_labels, results[,"SVM_LABEL"])

# recall accuracy
recall_accuracy(test_labels, results[,"SVM_LABEL"]) #0.9824561
recall_accuracy(test_labels, results[,"MAXENTROPY_LABEL"]) #0.9736842
recall_accuracy(test_labels, results[,"FORESTS_LABEL"]) #0.9824561
recall_accuracy(test_labels, results[,"TREE_LABEL"]) #0.9824561
recall_accuracy(test_labels, results[,"NNETWORK_LABEL"]) #0.9824561


### BINARY implementation

# build the data to specify response variable, training set, testing set.

set.seed(3)

train_sample <- sample(459, 345)
test_sample <- 1:459
test_sample <- test_sample[-train_sample]

reviews$Rating_binary <- vector()
for (i in 1:459){
  if (reviews$Rating[i] < 5){reviews$Rating_binary[i] = "negative"}
  else {reviews$Rating_binary[i] = "positive"}
}

container = create_container(dtm_tf, reviews$Rating_binary,
                             trainSize=train_sample, testSize=test_sample,virgin=FALSE)

#train the model with multiple machine learning algorithms:
models = train_models(container, algorithms=c("SVM", "MAXENT", "RF", "TREE", "NNET"))

#classify the testing set using the trained models.
results = classify_models(container, models)


# accuracy table
test_labels <- reviews$Rating_binary[-train_sample]
table(test_labels, results[,"SVM_LABEL"])

# recall accuracy
recall_accuracy(test_labels, results[,"SVM_LABEL"])# 1
recall_accuracy(test_labels, results[,"MAXENTROPY_LABEL"])# 1
recall_accuracy(test_labels, results[,"FORESTS_LABEL"])# 1
recall_accuracy(test_labels, results[,"TREE_LABEL"])# 1
recall_accuracy(test_labels, results[,"NNETWORK_LABEL"])# 1

#results
accuracy_comp <-  c(c(0.0065, 0.19,0.69), c(0.63, 0.99,1),
                  c(0.72, 0.98, 1),  c(0.59, 0.97, 1) , c(0.7, 0.98, 1),
                  c(0.6, 0.98, 1), c(0.63, 0.98, 1))
method_comp <- c(rep("Lexicon-based",3),
                 rep("Naive Bayes", 3), rep('SVM', 3), 
                 rep("MAXENT", 3), rep("RF", 3), rep("TREE", 3), rep("NNET",3))
formul_comp <- rep(c("NORMAL", "BASIC", "BINARY"), 7)
sup_comp <- c(rep("Unsupervised", 3), rep("Supervised", 18))
sup_comp <- factor(sup_comp, levels = c("Unsupervised", "Supervised"),
                   labels = c("Unsupervised", "Supervised"))
x <- c(rep(1,3),
       rep(2, 3), rep(3, 3), 
       rep(4, 3), rep(5, 3), rep(6, 3), rep(7,3))
x <- factor(x, labels = c("Lexicon-based", "Naive Bayes", 'SVM', "Maximum entropy", 
                          "Random forest", "Regression tree", "Neural networks"))

comp_df = data.frame (x, accuracy_comp, method_comp, formul_comp, sup_comp)

g <- ggplot(comp_df, aes(x = x, y = accuracy_comp, colour = I("#2171b5")))
g+ theme_bw() + geom_point()+ 
  facet_grid( formul_comp ~ sup_comp, scales = "free") +
  ylab("Accuracy") + xlab("Method") + ylim(0:1)+
  ggtitle("The accuracy of the implemented algorithms")+
  theme(axis.text.x = element_text(angle=45, hjust = 1, 
                size = 14, colour = "#084594"),
        text = element_text(size = 14))

