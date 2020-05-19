# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:10:10 2020

@author: Wladyslaw Eysymontt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn import preprocessing
import statistics
from scipy.stats import boxcox
from scipy.stats import shapiro


train_df = pd.read_csv("../Files/Completely_processed_data/shaped_input_train_5_30_100.csv")
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df.to_csv("../Files/Completely_processed_shuffled_data/shuffled_shaped_input_train_5_30_100.csv", index=False)

test_df = pd.read_csv("../Files/Completely_processed_data/shaped_input_test_5_30_100.csv")
test_df = test_df.sample(frac=1).reset_index(drop=True)
test_df.to_csv("../Files/Completely_processed_shuffled_data/shuffled_shaped_input_test_5_30_100.csv", index=False)


def getCorrelationsFilteredByShopsAndItems(dataframe,shops,items,parameter):
    counter = 0
    total = [0,0,0,0,0]
    for shop in shops:
        for item in items:
            counter += 1
            
            filter_by_shop = dataframe['shop_id']==shop
            filtered_df = dataframe[filter_by_shop]
            filter_by_item = filtered_df['item_id']==item
            filtered_df = filtered_df[filter_by_item]
            
            if parameter == 'relative_shop_size':
                corr_matrix = np.corrcoef(filtered_df["sales_t"], filtered_df["relative_shop_size"])
                where_are_NaNs = np.isnan(corr_matrix)
                corr_matrix[where_are_NaNs] = 0
                correlation = corr_matrix[0][1]
                total[0] += correlation
            
            else:
                for i in range(0,5):
                    if i == 0:
                        if parameter == 'price':
                            corr_matrix = np.corrcoef(filtered_df["sales_t"], filtered_df["price_t"])
                            where_are_NaNs = np.isnan(corr_matrix)
                            corr_matrix[where_are_NaNs] = 0
                            correlation = corr_matrix[0][1]
                            total[0] += correlation
                    else:
                        corr_matrix = np.corrcoef(filtered_df["sales_t"], filtered_df[("{}_t-{}").format(parameter,i)])
                        where_are_NaNs = np.isnan(corr_matrix)
                        corr_matrix[where_are_NaNs] = 0
                        correlation = corr_matrix[0][1]
                        total[i] += correlation
            
    mean_correlation = [0,0,0,0,0]
    for i in range(0,5):
        mean_correlation[i] = total[i] / counter
    
    return mean_correlation


def getCorrelationsFilteredByItems(dataframe,items,parameter):
    counter = 0
    total = [0,0,0,0,0]
    for item in items:
        counter += 1
        
        filter_by_item = dataframe['item_id']==item
        filtered_df = dataframe[filter_by_item]
        
        if parameter == 'relative_shop_size':
            corr_matrix = np.corrcoef(filtered_df["sales_t"], filtered_df["relative_shop_size"])
            where_are_NaNs = np.isnan(corr_matrix)
            corr_matrix[where_are_NaNs] = 0
            correlation = corr_matrix[0][1]
            total[0] = total[0] + correlation
        
        else:
            for i in range(0,5):
                if i == 0:
                    if parameter == 'price':
                        corr_matrix = np.corrcoef(filtered_df["sales_t"], filtered_df["price_t"])
                        where_are_NaNs = np.isnan(corr_matrix)
                        corr_matrix[where_are_NaNs] = 0
                        correlation = corr_matrix[0][1]
                        total[0] = total[0] + correlation
                else:
                    corr_matrix = np.corrcoef(filtered_df["sales_t"], filtered_df[("{}_t-{}").format(parameter,i)])
                    where_are_NaNs = np.isnan(corr_matrix)
                    corr_matrix[where_are_NaNs] = 0
                    correlation = corr_matrix[0][1]
                    total[i] = total[i] + correlation
        
    mean_correlation = [0,0,0,0,0]
    for i in range(0,5):
        mean_correlation[i] = total[i] / counter
    
    return mean_correlation


def getCorrelationsNotFiltered(dataframe,parameter):
    counter = 0
    total = [0,0,0,0,0]
    for item in items:
        counter += 1
        
        if parameter == 'relative_shop_size':
            corr_matrix = np.corrcoef(dataframe["sales_t"], dataframe["relative_shop_size"])
            where_are_NaNs = np.isnan(corr_matrix)
            corr_matrix[where_are_NaNs] = 0
            correlation = corr_matrix[0][1]
            total[0] = total[0] + correlation
        
        else:
            for i in range(0,5):
                if i == 0:
                    if parameter == 'price':
                        corr_matrix = np.corrcoef(dataframe["sales_t"], dataframe["price_t"])
                        where_are_NaNs = np.isnan(corr_matrix)
                        corr_matrix[where_are_NaNs] = 0
                        correlation = corr_matrix[0][1]
                        total[0] = total[0] + correlation
                else:
                    corr_matrix = np.corrcoef(dataframe["sales_t"], dataframe[("{}_t-{}").format(parameter,i)])
                    where_are_NaNs = np.isnan(corr_matrix)
                    corr_matrix[where_are_NaNs] = 0
                    correlation = corr_matrix[0][1]
                    total[i] = total[i] + correlation
        
    mean_correlation = [0,0,0,0,0]
    for i in range(0,5):
        mean_correlation[i] = total[i] / counter
    
    return mean_correlation


def standarize(input_array,mean,standard_deviation):
    standarized_array = []
    for number_of_sales in input_array:
        standarized = (number_of_sales-mean)/standard_deviation
        standarized_array.append(standarized)
    return standarized_array


def standarizeArray(input_array,mean,standard_deviation):
    standarized_array = []
    for number_of_sales in input_array:
        standarized = (number_of_sales-mean)/standard_deviation
        standarized_array.append([standarized])
    return standarized_array


def destandarize(input_array,mean,standard_deviation):
    destandarized_array = []
    for standarized in input_array:
        sales = (standarized*standard_deviation)+mean
        destandarized_array.append(sales)
    return destandarized_array


def applyBoxCox(column):
    filtered = train_df[column].values
    filtered_1D = []
    for row in filtered:
        filtered_1D.append(row)
    transformed = boxcox(filtered_1D,0)
    return transformed



no_negative_df = train_df[(train_df > 0).all(1)]
train_df = no_negative_df


plt.hist(train_df["sales_t"], bins=60, range = (0,60))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')

plt.hist(train_df["price_t"], bins=100, range = (0,6000))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')

plt.hist(train_df["relative_shop_size"], bins=100, range = (0,10))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')

plt.hist(train_df["salesRelationalCoefficient_t-1"], bins=60, range = (-3,14))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')

plt.hist(train_df["priceRelationalCoefficient_t-1"], bins=60, range = (0,14))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')


shops = train_df["shop_id"].unique()
items = train_df["item_id"].unique()


price_correlation = getCorrelationsFilteredByShopsAndItems(train_df,shops,items,'price')
sales_correlation = getCorrelationsFilteredByShopsAndItems(train_df,shops,items,'sales')

priceRelationalCoefficient_correlation = getCorrelationsNotFiltered(train_df,'priceRelationalCoefficient')
salesRelationalCoefficient_correlation = getCorrelationsNotFiltered(train_df,'salesRelationalCoefficient')
relative_shop_size_correlation = getCorrelationsNotFiltered(train_df,'relative_shop_size')


to_log = ['sales_t']
df_log = train_df[to_log].applymap(lambda x: np.log(x+1))
plt.hist(df_log["sales_t"], bins=100, range = (0,4))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
stat = shapiro(df_log)
print('Statistics={}'.format(stat[0]))

transformed = applyBoxCox('price_t')
plt.hist(transformed, bins=100, range = (0,10))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
stat = shapiro(transformed)
print('Statistics={}'.format(stat[0]))

transformed = applyBoxCox('relative_shop_size')
plt.hist(transformed, bins=100, range = (0,2))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
# Normalize

transformed = applyBoxCox('salesRelationalCoefficient_t-1')
plt.hist(transformed, bins=100, range = (-2,3))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
stat = shapiro(transformed)
print('Statistics={}'.format(stat[0]))

transformed = applyBoxCox('priceRelationalCoefficient_t-1')
plt.hist(transformed, bins=100, range = (-6,6))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
stat = shapiro(transformed)
print('Statistics={}'.format(stat[0]))



allSales = []
for variable in range(2,len(train_df.columns)):
    column_name = train_df.columns[variable]
    if 'sales_' in column_name:
        for value in train_df.iloc[:,variable:(variable+1)].values:
            allSales.append(value[0])

global_sales_mean = sum(allSales) / len(allSales)
global_sales_stdev = statistics.stdev(allSales)

scaled_columns = []
for variable in range(2,len(train_df.columns)):
    column_name = train_df.columns[variable]
    if 'sales_' in column_name:
        to_log = [column_name]
        df_log = train_df[to_log].applymap(lambda x: np.log(x+1))
        scaled_variable = np.asarray(standarizeArray(df_log.values,global_sales_mean,global_sales_stdev))
    elif 'relative_shop_size' in column_name:
        scaled_variable = preprocessing.normalize(train_df.iloc[:,variable:(variable+1)])
    else:
        transformed = applyBoxCox(column_name)
        scaled_variable = preprocessing.scale(transformed)
        break
    temp_array = []
    for value in scaled_variable:
        try:
            temp_array.append(value[0])
        except:
            temp_array.append(value)
    scaled_columns.append(temp_array)


sum(scaled_variable)/len(scaled_variable)
plt.hist(scaled_variable, bins=100, range = (0,1))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')