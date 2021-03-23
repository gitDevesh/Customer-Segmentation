# Clearing the environment
rm(list = ls())

#### Installing Packages ####
# install.packages(c('dplyr', 'ggplot2', 'corrplot', 'plotly', 'caret', 'ggfortify',
                   # 'factoextra', 'normalr','psych'))

#### Installing important libraries ####
x <- c('dplyr', 'ggplot2', 'corrplot', 'plotly', 'caret', 'ggfortify',
       'factoextra', 'normalr','psych')
lapply(x, require, character.only = TRUE)

rm(x)

# Set Working Directory
setwd('D:\\Data Science\\Edwisor\\Projects\\Customer Segmentation\\R')

# PDF for the plots
#pdf(file = 'Rplot%03d.pdf')

#### Imorting Dataset ####
credit_card <- read.csv('credit-card-data.csv', header = TRUE)
View(credit_card)

# General information about data
dim(credit_card)
summary(credit_card)
str(credit_card)

# As we can see all except CUST ID are integer or numeric, let us drop it
credit_card <- credit_card[-1]

#### EDA ####
# Missing Values Analysis
missing_values <- data.frame(sapply(credit_card, function(x){sum(is.na(x))}))
missing_values$columns <- row.names(missing_values)
names(missing_values)[1] <- 'Missing Percentage'
missing_values$`Missing Percentage` <- missing_values$`Missing Percentage`/nrow(credit_card)*100
missing_values <- missing_values[order(-missing_values$`Missing Percentage`),]
row.names(missing_values) <- NULL
missing_values <- missing_values[c(2,1)]
View(missing_values)

# Imputing missing values(median)
credit_card$MINIMUM_PAYMENTS[is.na(credit_card$MINIMUM_PAYMENTS)] <- median(credit_card$MINIMUM_PAYMENTS, na.rm = TRUE)
credit_card$CREDIT_LIMIT[is.na(credit_card$CREDIT_LIMIT)] <- median(credit_card$CREDIT_LIMIT, na.rm = TRUE)
sum(is.na(credit_card))

# outliers Analysis
par(mfrow = c(1,1))

for (i in 1:17){
  boxplot(credit_card[,i], main = colnames(credit_card[i]))
}

# We have outliers in almost all the variables but we are not going to remove or impute them because these values are 
# important for the data as each customer has different spending which naturally makes the values to vary by a lot.
# Also a lot of variables have 0 value due to which log transformation can not be applied on the dataframe.

# Correlation Matrix
corrplot(cor(credit_card), method = 'number', type = 'lower', tl.cex = 0.5, tl.srt = 45)

#### Creating KPI's ####
credit_KPI <- credit_card
credit_KPI$monthly_avg_purchase <- credit_KPI$PURCHASES/credit_KPI$TENURE
credit_KPI$monthly_avg_cash_adv <- credit_KPI$CASH_ADVANCE/credit_KPI$TENURE
credit_KPI$limit_usage <- credit_KPI$BALANCE/credit_KPI$CREDIT_LIMIT
credit_KPI$min_pay <- credit_KPI$PAYMENTS/credit_KPI$MINIMUM_PAYMENTS

# for (i in 1:21) {
#   credit_KPI <- round(credit_KPI[i], digits = 3)
# }

#### Visualizing Data ####
par(mfrow = c(2,1))

for (i in 1:17) {
  hist(credit_card[,i], main = colnames(credit_card)[i], col = 'magenta')
}

#### Data Scaling(Standardization) ####
scaled_data <- scale(credit_card)
#scaled_data <- normalise(credit_card)

# Correlation Matrix
par(mfrow = c(1,1))
corr1 <- corrplot(cor(scaled_data), method = 'number', type = 'lower', tl.cex = 0.5, tl.srt = 45)

#### Dimensionality reduction, using Factanal for Factor Analysis ####
# Number of factors required for Factor Analysis(USING PCA)
scaled_data.pca <- prcomp(scaled_data)
summary(scaled_data.pca)
# y <- scaled_data.pca$x[,1:8]

# As we can see from the result more than 75% variance in the data can be explained
# by 6 factors. So we will choose 6 factors for factor analysis

# Factor Analysis
credit_fa <- fa(corr1, nfactors = 6, rotate = 'varimax', fm = 'ml')
credit_fa
credit_fa$loadings

# Based on the result from Factor Analysis we can reduce the number of variables
# from the dataset to only those who explain most of the variance
credit_new <- subset(scaled_data, select = c('BALANCE','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES',
                                             'CASH_ADVANCE','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY',
                                             'PURCHASES_INSTALLMENTS_FREQUENCY','CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX',
                                             'PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS'))

#### KMeans ####
# Checking the number of clusters needed
fviz_nbclust(credit_new, method = 'wss', kmeans)

# Creating a clustering model(KMeans)
set.seed(1)
model <- kmeans(credit_new, centers = 5, iter.max = 50)
model

# Filtering on the basis of clusters
credit_clusters <- data.frame(credit_KPI, model$cluster)

for (i in 1:5) {
  View(filter(credit_clusters, (credit_clusters$model.cluster == i)))
}

#### Visualizing the clusters ####
autoplot(model, data = scaled_data, frame = TRUE)

#### KPI'S visualization with different clusters ####
# Barplot of Clusters and monthly_avg_purchase
ggplot(credit_clusters, aes(x = model.cluster,y= monthly_avg_purchase)) +
  geom_bar(stat='identity') + theme() + xlab('Cluster') + ylab('Monthly_avg_purchase') +
  ggtitle('Cluster - Monthly avg purchase')

# Barplot of Clusters and monthly_avg_cash_adv
ggplot(credit_clusters, aes(x = model.cluster, y = monthly_avg_cash_adv)) +
  geom_bar(stat = 'identity') + theme() + xlab('Cluster') + ylab('Monthy_avg_cash_advance')+
  ggtitle('Cluster - Monthly_avg_cash_adv')

# Barplot of Clusters and min_pay
ggplot(credit_clusters, aes(x = model.cluster, y = min_pay)) +
  geom_bar(stat = 'identity') + theme() + xlab('Cluster') + ylab('Min_pay') +
  ggtitle('Cluster - Min_pay')

# Scatterplot of Purchases and Balance
ggplot(credit_clusters, aes_string(x = credit_clusters$PURCHASES, y = credit_clusters$BALANCE))+
  geom_point(aes_string(colour = credit_clusters$model.cluster)) + theme() + 
  xlab('Purchases') + ylab('BALANCE') + ggtitle('Purchase-Balance')

# Scatterplot of Purchases and Credit Limit
ggplot(credit_clusters, aes_string(x = credit_clusters$PURCHASES, y = credit_clusters$CREDIT_LIMIT))+
  geom_point(aes_string(colour = credit_clusters$model.cluster)) + theme() + 
  xlab('Purchases') + ylab('Cedit Limit') + ggtitle('Purchase-Credit Limit')

# Scatterpplot of ONEOFF_PURCHASES and INSTALLMENTS_PURCHASES
ggplot(credit_clusters, aes_string(x = credit_clusters$ONEOFF_PURCHASES, y = credit_clusters$INSTALLMENTS_PURCHASES))+
  geom_point(aes_string(colour = credit_clusters$model.cluster)) + theme() + 
  xlab('ONEOFF_PURCHASES') + ylab('INSTALLMENTS_PURCHASES') + ggtitle('ONEOFF_PURCHASES-INSTALLMENTS_PURCHASES')

# Scatterpplot of Balance and Credit Limit
ggplot(credit_clusters, aes_string(x = credit_clusters$BALANCE, y = credit_clusters$CREDIT_LIMIT))+
  geom_point(aes_string(colour = credit_clusters$model.cluster)) + theme() + 
  xlab('Balance') + ylab('CREDIT_LIMIT') + ggtitle('BALANCE-CREDIT_LIMIT')
