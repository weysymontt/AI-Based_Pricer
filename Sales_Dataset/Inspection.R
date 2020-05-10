setwd("C:/Master/TFM/Sales_Dataset")
df <- read.csv(file = 'C:/Master/TFM/Sales_Dataset/sales_train.csv', header = TRUE)
df$date # from 01/01/2013 to 31/12/2014, two complete years
max(df$item_price)