# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:08:58 2020

@author: Wladyslaw Eysymontt
"""

import pandas as pd
import numpy as np
from numpy import hstack
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn import preprocessing
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import statistics
import tkinter as tk
import matplotlib
from tkinter import DISABLED
from scipy.stats import boxcox


def dataframes(timeWindow, shops, items, model):
    train_df = pd.read_csv(("../Files/Completely_processed_data/shaped_input_train_{}_{}_{}.csv").format(timeWindow,shops,items))
    train_df = train_df.dropna()
    test_df = pd.read_csv(("../Files/Completely_processed_data/shaped_input_test_{}_{}_{}.csv").format(timeWindow,shops,items))
    test_df = test_df.dropna()
    
    no_negative_df = train_df[(train_df >= 0).all(1)]
    no_cero_df = no_negative_df.replace(0,0.0001)
    train_df = no_cero_df
    
    unique_shops = train_df["shop_id"].unique()
    unique_items = train_df["item_id"].unique()
    filter_by_shop = test_df[test_df['shop_id'].isin(unique_shops)]
    filter_by_item = filter_by_shop[filter_by_shop['item_id'].isin(unique_items)]
    test_df = filter_by_item
    no_negative_df = test_df[(test_df >= 0).all(1)]
    no_cero_df = no_negative_df.replace(0,0.0001)
    test_df = no_cero_df
    
    deseasonalized_df = pd.read_csv(("../Files/Deseasonalized/deseasonalizedSales_{}_{}.csv").format(shops,items))
    prices_df = pd.read_csv(("../Files/Product_prices/product_prices_{}_{}.csv").format(shops,items))
    prices_df = prices_df.groupby('item_id').mean().reset_index()
    
    model = load_model(('../Files/Trained_ANNs/{}.h5').format(model))
    
    sales_mean = sum(test_df.iloc[:,2].values)/len(test_df.iloc[:,2].values)
    sales_stdev = statistics.stdev(test_df.iloc[:,2].values)
    return (test_df,deseasonalized_df,prices_df,model,sales_mean,sales_stdev)


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


def getMeanPrediction(one_row_dataframe):
    prices = []
    counter = -1
    for observation in one_row_dataframe:
        counter += 1
        if 'sales_' in observation and observation != 'sales_t':
            prices.append([one_row_dataframe.iloc[[0],counter].values])
    sumOfPrices = 0
    for price in prices:
        sumOfPrices += price[0]
    mean = sumOfPrices / len(prices)
    return mean


def applyBoxCox(dataframe,column):
    filtered = dataframe[column].values
    filtered_1D = []
    for row in filtered:
        filtered_1D.append(row)
    transformed = boxcox(filtered_1D,0)
    return transformed


def getAnnPrediction(model, one_row_dataframe, scaled_columns, global_sales_mean, global_sales_stdev, product_price='default'):
    row_index = str(one_row_dataframe.index).split('[')[1].split(']')[0]
    timeWindow = int((len(scaled_columns)-3)/4)
    X_test = []
    for row in range(0,len(scaled_columns[0])):
        if row == int(row_index):
            price_t = []
            if product_price != 'default':
                mean = sum(scaled_columns[1])/len(scaled_columns[1])
                stdev = statistics.stdev(scaled_columns[1])
            for period in range(0,timeWindow):
                if product_price == 'default':
                    price_t.append(scaled_columns[1][row])
                else:
                    price_t.append(standarize([product_price],mean,stdev))
            price_t = np.asarray(price_t)
            price_t = price_t.reshape((len(price_t), 1))
            relative_shop_size = []
            for period in range(0,timeWindow):
                relative_shop_size.append(scaled_columns[2][row])
            relative_shop_size = np.asarray(relative_shop_size)
            relative_shop_size = relative_shop_size.reshape((len(relative_shop_size), 1))
                
            allSeries = [price_t,relative_shop_size]
            
            for serie in range(3,7):
                values = []
                counter = 0
                startCounter = False
                for variable in range(3,(len(scaled_columns))):
                    if startCounter == True:
                        counter += 1
                    if counter % timeWindow == 0 and variable == (serie + counter):
                        startCounter = True
                        values.append(scaled_columns[variable][row])
                values = np.asarray(values)
                values = values.reshape((len(values), 1))
                allSeries.append(values)
            
            observation = hstack(allSeries)
            X_test.append(observation)
    
    X_test = np.asarray(X_test)
    
    ann_predictions_test = []
    predicted = (model.predict(X_test)).tolist()
    for prediction in predicted:
        ann_predictions_test.append(prediction[0])
    ann_prediction = destandarize(ann_predictions_test,global_sales_mean,global_sales_stdev)
    ann_prediction = ann_prediction[0]
    if ann_prediction < 0:
        ann_prediction = 0
    return ann_prediction


def getVariables(test_df, sales_mean, sales_stdev):
    allSales = []
    for variable in range(2,len(test_df.columns)):
        column_name = test_df.columns[variable]
        if 'sales_' in column_name and column_name != 'sales_t':
            for value in test_df.iloc[:,variable:(variable+1)].values:
                allSales.append(value[0])
    
    global_sales_mean = sum(allSales) / len(allSales)
    global_sales_stdev = statistics.stdev(allSales)
    
    scaled_columns = []
    for variable in range(2,len(test_df.columns)):
        column_name = test_df.columns[variable]
        if 'sales_' in column_name:
            scaled_variable = np.asarray(standarizeArray(test_df.iloc[:,variable].values,global_sales_mean,global_sales_stdev))
        elif 'relative_shop_size' in column_name:
            scaled_variable = preprocessing.normalize(test_df.iloc[:,variable:(variable+1)])
        else:
            transformed = applyBoxCox(test_df,column_name)
            scaled_variable = preprocessing.scale(transformed)
        temp_array = []
        for value in scaled_variable:
            try:
                temp_array.append(value[0][0])
            except:
                temp_array.append(value)
        scaled_columns.append(temp_array)
    
    
    timeWindow = int((len(scaled_columns)-3)/4)
    X_test = []
    for row in range(0,len(scaled_columns[0])):
        price_t = []
        for period in range(0,timeWindow):
            price_t.append(scaled_columns[1][row])
        price_t = np.asarray(price_t)
        price_t = price_t.reshape((len(price_t), 1))
        relative_shop_size = []
        for period in range(0,timeWindow):
            relative_shop_size.append(scaled_columns[2][row])
        relative_shop_size = np.asarray(relative_shop_size)
        relative_shop_size = relative_shop_size.reshape((len(relative_shop_size), 1))
            
        allSeries = [price_t,relative_shop_size]
        
        for serie in range(3,7):
            values = []
            counter = 0
            startCounter = False
            for variable in range(3,(len(scaled_columns))):
                if startCounter == True:
                    counter += 1
                if counter % timeWindow == 0 and variable == (serie + counter):
                    startCounter = True
                    values.append(scaled_columns[variable][row])
            values = np.asarray(values)
            values = values.reshape((len(values), 1))
            allSeries.append(values)
        
        observation = hstack(allSeries)
        X_test.append(observation)
    
    X_test = np.asarray(X_test)
    Y_test = standarize(test_df['sales_t'],global_sales_mean,global_sales_stdev)
    Y_test_only_standarization = standarize(test_df['sales_t'],sales_mean,sales_stdev)
    Y_destandarized = destandarize(Y_test_only_standarization,sales_mean,sales_stdev)

    return (X_test, Y_test, Y_destandarized, global_sales_mean, global_sales_stdev, scaled_columns)


######################################################################

def calculatePerformance(self):
    self.Label3.config(fg=self.text_color)
    self.Label4.config(fg=self.text_color)
    self.Label7.config(fg=self.text_color)
    self.Label8.config(fg=self.text_color)
    self.Label12.config(fg=self.text_color)
    self.Label13.config(fg=self.text_color)
    self.Label14.config(fg=self.text_color)
    self.Label15.config(fg=self.text_color)
    self.Label16.config(fg=self.text_color)
    
    errors_detected = False
    
    if self.model.get() == '':
        errors_detected = True
        self.Label3.config(fg=self.incorrect_color)
    
    if self.test_set.get() == '':
        errors_detected = True
        self.Label4.config(fg=self.incorrect_color)
        
        
    if errors_detected == False:
        percentage_counter = 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        
        timeWindow = self.test_set.get().split('_')[0]
        shops = self.test_set.get().split('_')[1]
        items = self.test_set.get().split('_')[2]
        percentage_counter += 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        
        data_to_use = dataframes(timeWindow, shops, items, self.model.get())
        percentage_counter += 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        test_df = data_to_use[0]
        deseasonalized_df = data_to_use[1]
        model = data_to_use[3]
        sales_mean = data_to_use[4]
        sales_stdev = data_to_use[5]
        percentage_counter += 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        
        
        variables_to_use = getVariables(test_df, sales_mean, sales_stdev)
        percentage_counter += 2
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        X_test = variables_to_use[0]
        Y_test = variables_to_use[1]
        Y_destandarized = variables_to_use[2]
        global_sales_mean = variables_to_use[3]
        global_sales_stdev = variables_to_use[4]
        percentage_counter += 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        
        shops = test_df['shop_id'].unique()
        items = test_df['item_id'].unique()
        percentage_counter += 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        
        steps = len(shops) * len(items)
        steps_in_percentage = steps / 90
        if steps_in_percentage < 1:
            percentages_in_step = 1 / steps_in_percentage
        percentage_counter += 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        
        
        # Mean price as prediction
        mean_predictions_test = []
        for observation in X_test:
            prices = []
            for variable in observation:
                prices.append(variable[2])
            sumOfPrices = 0
            for price in prices:
                sumOfPrices += price
            mean = sumOfPrices / len(prices)
            mean_predictions_test.append(mean)
        mean_predictions_destandarized = destandarize(mean_predictions_test,global_sales_mean,global_sales_stdev)
        
        
        # ANN predictions
        ann_predictions_test = []
        predicted = (model.predict(X_test)).tolist()
        for prediction in predicted:
            ann_predictions_test.append(prediction[0])
        
        ann_predictions_destandarized_positive = []
        ann_predictions_destandarized = destandarize(ann_predictions_test,global_sales_mean,global_sales_stdev)
        for prediction in ann_predictions_destandarized:
            if prediction < 0:
                ann_predictions_destandarized_positive.append(0)
            else:
                ann_predictions_destandarized_positive.append(prediction)
        
        
        error_of_mean_test = mean_squared_error(Y_test, mean_predictions_test).round(4)
        error_of_mean_destandarized = mean_squared_error(Y_destandarized, mean_predictions_destandarized).round(4)
        error_of_ann_test = mean_squared_error(Y_test, ann_predictions_test).round(4)
        error_of_ann_destandarized = mean_squared_error(Y_destandarized, ann_predictions_destandarized_positive).round(4)
        efficiency_relation = error_of_mean_destandarized / error_of_ann_destandarized   
        
             
        failed_to_calculate = 0
        correct_values = []
        mean_predictions = []
        counter = 0
        percentage_counter += 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        for shop in shops:
            for item in items:
                counter += 1
                if steps_in_percentage >= 1:
                    if counter >= steps_in_percentage:
                        percentage_counter += 1
                        if percentage_counter <= 100:
                            self.progress['value'] = percentage_counter
                            self.wind.update_idletasks()
                        else:
                            self.progress['value'] = 100
                            self.wind.update_idletasks()
                        counter = counter-steps_in_percentage
                else:
                    percentage_counter += percentages_in_step
                    if percentage_counter <= 100:
                        self.progress['value'] = percentages_in_step
                        self.wind.update_idletasks()
                    else:
                        self.progress['value'] = 100
                        self.wind.update_idletasks()
                
                filter_df = test_df['shop_id']==shop
                filtered = test_df[filter_df]
                filter_df = filtered['item_id']==item
                filtered = filtered[filter_df]
                
                month_number = max(deseasonalized_df['date_block_num'].unique())
                filter_df_si = deseasonalized_df['shop_id']==shop
                filtered_si = deseasonalized_df[filter_df_si]
                filter_df_si = filtered_si['item_id']==item
                filtered_si = filtered_si[filter_df_si]
                filter_df_si = filtered_si['date_block_num']==month_number
                filtered_si = filtered_si[filter_df_si]
                seasonal_index = filtered_si.iloc[0,-1]
                
                try:
                    real_value = int((((filtered['sales_t'].values)[0]) * seasonal_index).round())
                    mean_prediction = int(((getMeanPrediction(filtered)[0]) * seasonal_index).round())
                    correct_values.append(real_value)
                    mean_predictions.append(mean_prediction)
                except:
                    failed_to_calculate += 1
        
        self.progress['value'] = 100
        self.wind.update_idletasks()
        
        error_of_mean = mean_squared_error(correct_values, mean_predictions).round(4)
        error_of_ann = error_of_mean / efficiency_relation.round(4)
        
        
        modelPerformance = ('''MSE of standarized mean predictions: {}
MSE of standarized ANN predictions: {}

MSE of deseasonalized mean predictions: {}
MSE of deseasonalized ANN predictions: {}

MSE of seasonalized mean predictions: {}
MSE of seasonalized ANN predictions: {}''').format(error_of_mean_test,error_of_ann_test,error_of_mean_destandarized,error_of_ann_destandarized,error_of_mean,error_of_ann)

        self.performance.config(state='normal')
        self.performance.delete('1.0', tk.END)
        self.performance.insert(tk.END, modelPerformance)
        self.performance.config(state=DISABLED)
        
        self.progress['value'] = 0
        self.wind.update_idletasks()


######################################################################

def calculatePredictions(self, background_color):
    self.Label3.config(fg=self.text_color)
    self.Label4.config(fg=self.text_color)
    self.Label7.config(fg=self.text_color)
    self.Label8.config(fg=self.text_color)
    self.Label12.config(fg=self.text_color)
    self.Label13.config(fg=self.text_color)
    self.Label14.config(fg=self.text_color)
    self.Label15.config(fg=self.text_color)
    self.Label16.config(fg=self.text_color)
    
    errors_detected = False
    
    if self.model.get() == '':
        errors_detected = True
        self.Label3.config(fg=self.incorrect_color)
    
    if self.test_set.get() == '':
        errors_detected = True
        self.Label4.config(fg=self.incorrect_color)
    
    if self.shop.get() == '':
        errors_detected = True
        self.Label7.config(fg=self.incorrect_color)
    
    if self.item.get() == '':
        errors_detected = True
        self.Label8.config(fg=self.incorrect_color)
    
    try:
        if float(self.precision.get()).is_integer() != True or float(self.precision.get()) < 1:
            errors_detected = True
            self.Label12.config(fg=self.incorrect_color)
    except:
        errors_detected = True
        self.Label12.config(fg=self.incorrect_color)
    
    try:
        if float(self.max_price_multiplicator.get()) < 1:
            errors_detected = True
            self.Label13.config(fg=self.incorrect_color)
    except:
        errors_detected = True
        self.Label13.config(fg=self.incorrect_color)
    
    try:
        if float(self.delta_multiplicator.get()) < 0:
            errors_detected = True
            self.Label14.config(fg=self.incorrect_color)
    except:
        errors_detected = True
        self.Label14.config(fg=self.incorrect_color)
    
    try:
        if self.selectedBtn.get() == "2":
            if float(self.item_cost.get()) < 0 or float(self.item_cost.get()) > 1:
                errors_detected = True
                self.Label15.config(fg=self.incorrect_color)
        else:
            if float(self.item_cost.get()) < 0:
                errors_detected = True
                self.Label15.config(fg=self.incorrect_color)
    except:
        errors_detected = True
        self.Label15.config(fg=self.incorrect_color)
    
    try:
        if float(self.fixed_costs.get()) < 0:
            errors_detected = True
            self.Label16.config(fg=self.incorrect_color)
    except:
        errors_detected = True
        self.Label16.config(fg=self.incorrect_color)
    
    
    if errors_detected == False:
        percentage_counter = 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        
        timeWindow = self.test_set.get().split('_')[0]
        shops = self.test_set.get().split('_')[1]
        items = self.test_set.get().split('_')[2]
        percentage_counter += 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        
        data_to_use = dataframes(timeWindow, shops, items, self.model.get())
        percentage_counter += 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        test_df = data_to_use[0]
        prices_df = data_to_use[2]
        model = data_to_use[3]
        sales_mean = data_to_use[4]
        sales_stdev = data_to_use[5]
        percentage_counter += 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        
        
        variables_to_use = getVariables(test_df, sales_mean, sales_stdev)
        percentage_counter += 2
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        global_sales_mean = variables_to_use[3]
        global_sales_stdev = variables_to_use[4]
        scaled_columns = variables_to_use[5]
        percentage_counter += 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        
        shops = test_df['shop_id'].unique()
        items = test_df['item_id'].unique()
        percentage_counter += 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        
        item_id = int(self.item.get())
        shop_id = int(self.shop.get())
        partitions = int(self.precision.get())
        max_price_multiplicator = float(self.max_price_multiplicator.get())
        penalization_coefficient = float(self.delta_multiplicator.get())
        filter_df = prices_df['item_id']==item_id
        filtered = prices_df[filter_df]
        item_mean_price = filtered.iloc[0,2]
        item_cost = float(self.item_cost.get())
        if self.selectedBtn.get() == "2":
            item_cost = item_cost * item_mean_price
        fixed_costs = float(self.fixed_costs.get())
        percentage_counter += 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        
        filter_df = test_df['shop_id']==shop_id
        filtered = test_df[filter_df]
        filter_df = filtered['item_id']==item_id
        filtered = filtered[filter_df]
        
        steps_in_percentage = partitions / 90
        if steps_in_percentage < 1:
            percentages_in_step = 1 / steps_in_percentage
        
        prices = []
        pure_ann_predictions = []
        counter = 0
        percentage_counter += 1
        self.progress['value'] = percentage_counter
        self.wind.update_idletasks()
        for possible_price in np.linspace(0,(item_mean_price*max_price_multiplicator),partitions):
            counter += 1
            if steps_in_percentage >= 1:
                if counter >= steps_in_percentage:
                    percentage_counter += 1
                    if percentage_counter <= 100:
                        self.progress['value'] = percentage_counter
                        self.wind.update_idletasks()
                    else:
                        self.progress['value'] = 100
                        self.wind.update_idletasks()
                    counter = counter-steps_in_percentage
            else:
                percentage_counter += percentages_in_step
                if percentage_counter <= 100:
                    self.progress['value'] = percentages_in_step
                    self.wind.update_idletasks()
                else:
                    self.progress['value'] = 100
                    self.wind.update_idletasks()
                        
            prices.append(possible_price)
            sales_prediction_ann = getAnnPrediction(model,filtered,scaled_columns,global_sales_mean,global_sales_stdev,possible_price)
            pure_ann_predictions.append(sales_prediction_ann)
        
        self.progress['value'] = 100
        self.wind.update_idletasks()
        
        counter = -1
        
        max_index = int(partitions/max_price_multiplicator)
        to_maximum = pure_ann_predictions[0:(max_index+1)]
        sales_backwards = []
        new_sales_backwards = []
        counter = -1
        
        for index in range((len(to_maximum)-1),0,-1):
            sales_backwards.append(to_maximum[index])
            counter += 1
            if counter == 0:
                new_sales_backwards.append(to_maximum[index])
            else:
                new_sales = new_sales_backwards[-1] + abs((sales_backwards[-2] - to_maximum[index]) * penalization_coefficient)
                new_sales_backwards.append(new_sales)
                
        new_sales_to_max = []
        for index in range((len(new_sales_backwards)-1),0,-1):
            new_sales_to_max.append(new_sales_backwards[index])
        
        sales_predictions_penalized = []
        accumulated_delta = 0
        counter = -1
        for possible_price in np.linspace(0,(item_mean_price*max_price_multiplicator),partitions):
            counter += 1
            if counter >= (max_index-1):
                if counter > 0:
                    if possible_price > item_mean_price:
                        delta = abs(pure_ann_predictions[counter-1] - pure_ann_predictions[counter]) * penalization_coefficient
                    else:
                        delta = 0
                else:
                    delta = 0
                accumulated_delta += delta
                sales_prediction = pure_ann_predictions[counter] - accumulated_delta
                if sales_prediction < 0:
                    sales_prediction = 0
                sales_predictions_penalized.append(sales_prediction)
            else:
                sales_predictions_penalized.append(new_sales_to_max[counter])
        
        
        pure_ann_benefits = []
        counter = -1
        for price in prices:
            counter += 1
            profit = (price*pure_ann_predictions[counter])-(item_cost*pure_ann_predictions[counter])-fixed_costs
            pure_ann_benefits.append(profit)
        
        pure_ann_maxBenefits = max(pure_ann_benefits)
        pure_ann_maxBenefits_index = 0
        counter = -1
        for value in pure_ann_benefits:
            counter += 1
            if value == pure_ann_maxBenefits:
                pure_ann_maxBenefits_index = counter
        
        pure_ann_maxBenefits = pure_ann_maxBenefits.round(2)
        pure_ann_optimal_price = (prices[pure_ann_maxBenefits_index]).round(2)
        
        
        benefits = []
        counter = -1
        for price in prices:
            counter += 1
            profit = (price*sales_predictions_penalized[counter])-(item_cost*sales_predictions_penalized[counter])-fixed_costs
            benefits.append(profit)
        
        maxBenefits = max(benefits)
        maxBenefits_index = 0
        counter = -1
        for value in benefits:
            counter += 1
            if value == maxBenefits:
                maxBenefits_index = counter
        
        maxBenefits = maxBenefits.round(2)
        optimal_price = (prices[maxBenefits_index]).round(2)
        
        
        item_mean_price = item_mean_price.round(2)
        sales_with_mean_price = (pure_ann_predictions[int(partitions/max_price_multiplicator)]).round(2)
        self.plots_data = (prices,pure_ann_predictions,sales_predictions_penalized,pure_ann_benefits,benefits,item_mean_price,sales_with_mean_price,pure_ann_optimal_price,pure_ann_maxBenefits,optimal_price,maxBenefits)
        
        font = {'family' : 'Verdana',
        'weight' : 'normal',
        'size'   : 10}
        matplotlib.rc('font', **font)
        
        self.plot1.clear()
        self.plot1.plot(prices,benefits)
        self.plot1.set_title('Benefits / Price (optimized algorithm)', size=12)
        self.plot1.set_ylabel('Benefits', fontsize = 12) # Y label
        self.plot1.set_xlabel('Price', fontsize = 12) # X label
        self.canvas.draw()
        self.text1.config(state='normal')
        self.text1.delete('1.0', tk.END)
        self.text1.insert(tk.END, (('Optimal price: {}').format(optimal_price)))
        self.text1.config(state=DISABLED)
        self.text2.config(state='normal')
        self.text2.delete('1.0', tk.END)
        self.text2.insert(tk.END, (('Corr. benefits: {}').format(maxBenefits)))
        self.text2.config(state=DISABLED)
        
        self.plot2.clear()
        self.plot2.plot(prices,sales_predictions_penalized)
        self.plot2.set_title('Sales / Price (optimized algorithm)', size=12)
        self.plot2.set_ylabel('Sales', fontsize = 12) # Y label
        self.plot2.set_xlabel('Price', fontsize = 12) # X label
        self.canvas2.draw()
        self.text3.config(state='normal')
        self.text3.delete('1.0', tk.END)
        self.text3.insert(tk.END, (('Mean price: {}').format(item_mean_price)))
        self.text3.config(state=DISABLED)
        self.text4.config(state='normal')
        self.text4.delete('1.0', tk.END)
        self.text4.insert(tk.END, (('Corr. sales: {}').format(sales_with_mean_price)))
        self.text4.config(state=DISABLED)
        
        self.progress['value'] = 0
        self.wind.update_idletasks()