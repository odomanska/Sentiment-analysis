# Sentiment analysis for Hubspot reviews

library(tm)
library(dplyr)
library(ggplot2)
library(graph)
library(Rgraphviz)
library(FactoMineR)
library(topicmodels)

## Loading data

reviews <- read.csv('Data/hubspot_reviews3.csv', stringsAsFactors = FALSE)

## Step 1. Exploring and preparing the data

reviews$Date <- as.Date(reviews$Date, "%m/%d/%Y")
str(reviews$Rating)
table(reviews$Rating)

### Example of the data
reviews[1,3]

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

### Keywords

lowfreq = 3.25
keywords <- findFreqTerms(tdm, lowfreq)
term.freq <- rowSums(as.matrix(tdm))
term.freq <- subset(term.freq, term.freq >= lowfreq)
df <- data.frame(term = names(term.freq), freq = term.freq)
df <- plyr::arrange(df, freq)

reorder_size <- function(x) {
  factor(x, levels = x)}

g <- ggplot(df, aes(x = reorder_size(df$term), y = freq)) + geom_bar(stat = "identity") + xlab("Terms") +
  ylab("Count") + coord_flip()
g

### Associations
corThreshold = 0.1
p <- plot(tdm, term = keywords, corThreshold, weighting = T)
p

### clusters of documents: Hierarchical Clustering on Principle Components (HCPC)

dtm_tf <- DocumentTermMatrix(myCorpus)# Document Term Matrix of reviews with term frequancy weight
dtm_tf_matr <- as.matrix(dtm_tf)

res.pca <- PCA(dtm_tf_matr)  
res.hcpc <- HCPC(res.pca)

### Topic modelling

lda <- LDA(dtm_tf, method = "Gibbs", k = 3) # is used to estimate and fit a latent dirichlet allocation model with the VEM algorithm (default) or Gibbs sampling; k - number of topics to be found.
term <- terms(lda, 8) # first 8 terms of every topic
term <- apply(term, MARGIN = 2, paste, collapse = ", ")
term

topic <- topics(lda, 1)
topics <- data.frame(date=reviews$Date, topic)
topics$topic <- factor(topics$topic,levels=c(1,2,3),
                      labels=c("Topic1","Topic2","Topic3"))
qplot(date, ..count.., data=topics, geom = "density", alpha=I(.5),
      fill=topic, main = "How topics of reviews changed over time",
      xlab="Date", ylab="Count")

# Sentiment analysis Hu and Liu vocabulary
hu.liu.pos <- scan('data/positive-words.txt',
                   what = 'character', comment.char = ';')

hu.liu.neg <- scan('data/negative-words.txt',
                   what = 'character', comment.char = ';')

score.sentiment = function(sentences, pos.words, neg.words){
  require(plyr); require(stringr)
  scores = laply(sentences, function(sentence, pos.words, neg.words) {
    word.list = str_split(sentence, '\\s+') # split into words
    words = unlist(word.list)
    l = length(words)
    # compare our words to the dictionaries of positive & negative terms
    pos.matches = !is.na(match(words, pos.words))# match() returns the position of the matched term or NA
    neg.matches = !is.na(match(words, neg.words))
    score = sum(pos.matches) - sum(neg.matches)
    score = score/l*100}, 
    pos.words, neg.words)
  return(scores)
  }

rating_hu_liu <- score.sentiment(reviews$Review, hu.liu.pos, hu.liu.neg)
hist(rating_hu_liu, main = "Sentiments of reviews using lexicon-based approach",
     xlab = "Difference between the number of positive and negative words in the reviews, 
     normalized based on the its length", ylab ="Number of reviews",
     col = "lightblue", xlim = range(-2:13), cex=2)

summary(rating_hu_liu)
rating_norm <- vector()
for (i in 1:459){
  if (rating_hu_liu[i] < 0){rating_norm[i] = "negative"}
  else if (rating_hu_liu[i] < 1) {rating_norm[i] = "neutral"}
  else {rating_norm[i] = "positive"}
}

rating <- vector()
for (i in 1:459){
  if (reviews$Rating[i] <4){rating[i] = "negative"}
  else if (reviews$Rating[i] < 8) {rating[i] = "neutral"}
  else {rating[i] = "positive"}
}

plus = 0
minus = 0
for (i in 1:459){
  if (rating[i] == rating_norm[i]){plus = plus+1}
  else minus = minus +1
}

accuracy = plus/length(rating)

# Naive Bayes classifyer 
# Data preparation -- creating training and test datasets

library(RTextTools)
library(e1071)
hub <- reviews[ , -c(1,4)]
hub$Rating <- as.factor(hub$Rating)

hub_corpus <- VCorpus(VectorSource(hub$Review))
hub_corpus_clean <- tm_map(hub_corpus, content_transformer(tolower))
hub_corpus_clean <- tm_map(hub_corpus_clean, removeNumbers)
hub_corpus_clean <- tm_map(hub_corpus_clean, removeWords, stopwords())
hub_corpus_clean <- tm_map(hub_corpus_clean, removePunctuation)
hub_corpus_clean <- tm_map(hub_corpus_clean, stemDocument, lazy=TRUE)
hub_corpus_clean <- tm_map(hub_corpus_clean, stripWhitespace,lazy=TRUE)

rating <- vector()
for (i in 1:459){
  if (is.na(hub$Rating[i])){rating[i] = 0}
  else if (hub$Rating[i] %in% 1:3){rating[i] = -1}
  else if (hub$Rating[i] %in% 4:6){rating[i] = 0}
  else if (hub$Rating[i] %in% 7:10) {rating[i] = 1}
}
rating <- as.factor(rating)

#as.character(hub_corpus[[1]]),50
#as.character(hub_corpus_clean[[1]]),50

hub_dtm <- DocumentTermMatrix(hub_corpus_clean)

set.seed(123)

train_sample <- sample(459, 345)
test_sample <- 1:459
test_sample <- test_sample[-train_sample]

hub_dtm_train <- hub_dtm[train_sample, ]
hub_dtm_test <- hub_dtm[- train_sample, ]

hub_train_labels <- rating[train_sample]
hub_test_labels <- rating[-train_sample]

prop.table(table(hub_train_labels))
prop.table(table(hub_test_labels))


# Data preparation -- creating indicator features for frequent words

hub_freq_words <- findFreqTerms(hub_dtm_train, 5)
str(hub_freq_words)

hub_dtm_freq_train<- hub_dtm_train[ , hub_freq_words]
hub_dtm_freq_test <- hub_dtm_test[ , hub_freq_words]

convert_counts <- function(x) {
  x <- ifelse(x > 0, 'Yes', 'No')
}

hub_train <- apply(hub_dtm_freq_train, MARGIN = 2, convert_counts)
hub_test <- apply(hub_dtm_freq_test, MARGIN = 2, convert_counts)

### Step 2. Training a model on the data

hub_classifier <- naiveBayes(hub_train, hub_train_labels)

### Step 3. Evaluating model performance

hub_test_pred <- predict(hub_classifier, hub_test)

table(hub_test_pred, hub_test_labels)
recall_accuracy(hub_test_pred, hub_test_labels)

### Step 4. Improving model performance

hub_classifier2 <- naiveBayes(hub_train, hub_train_labels, laplace = 1)
hub_test_pred2 <- predict(hub_classifier2, hub_test)

CrossTable(hub_test_pred2, hub_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))


### SVM model

# build the data to specify response variable, training set, testing set.
container = create_container(dtm_tf, rating,
                             trainSize=train_sample, testSize=test_sample,virgin=FALSE)

#train the model with multiple machine learning algorithms:
models = train_models(container, algorithms=c("SVM", "MAXENT", "RF", "TREE", "NNET"))

#classify the testing set using the trained models.
results = classify_models(container, models)


# accuracy table
table(hub_test_labels, results[,"SVM_LABEL"])

# recall accuracy
recall_accuracy(hub_test_labels, results[,"SVM_LABEL"])
recall_accuracy(hub_test_labels, results[,"MAXENTROPY_LABEL"])
recall_accuracy(hub_test_labels, results[,"FORESTS_LABEL"])
recall_accuracy(hub_test_labels, results[,"TREE_LABEL"])
recall_accuracy(hub_test_labels, results[,"NNETWORK_LABEL"])


# model summary
analytics = create_analytics(container, results)
summary(analytics)

#results
score = c(0.85, 0.99, 0.99,0.99,0.99,0.99,0.99)
model = c("lexicon-based", "Naive Bayes", "SVM", "MAXENT", "RF", "TREE", "NNET")
barplot(height = score, names.arg = model)
