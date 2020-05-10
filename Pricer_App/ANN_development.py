# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 16:22:56 2020

@author: Wladyslaw Eysymontt
"""

import pandas as pd
import numpy as np
from numpy import hstack
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from sklearn.metrics import mean_squared_error
import statistics
from keras.callbacks import ModelCheckpoint


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


train_df = pd.read_csv("../Files/Completely_processed_data/shaped_input_train_5_30_100.csv")
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df.to_csv("../Files/Completely_processed_shuffled_data/shuffled_shaped_input_train_5_30_100.csv", index=False)

test_df = pd.read_csv("../Files/Completely_processed_data/shaped_input_test_5_30_100.csv")
test_df = test_df.sample(frac=1).reset_index(drop=True)
test_df.to_csv("../Files/Completely_processed_shuffled_data/shuffled_shaped_input_test_5_30_100.csv", index=False)

sales_mean = sum(train_df.iloc[:,2].values)/len(train_df.iloc[:,2].values)
sales_stdev = statistics.stdev(train_df.iloc[:,2].values)


plt.hist(train_df["sales_t"], bins=60, range = (0,60))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')

plt.hist(train_df["price_t"], bins=100, range = (0,6000))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')

plt.hist(train_df["relative_shop_size"], bins=40, range = (0,10))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')

plt.hist(train_df["salesRelationalCoefficient_t-1"], bins=60, range = (0,14))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')

plt.hist(train_df["priceRelationalCoefficient_t-1"], bins=60, range = (0,14))
plt.gca().set(title='Frequency Histogram', ylabel='Frequency')


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
        scaled_variable = np.asarray(standarizeArray(train_df.iloc[:,variable].values,global_sales_mean,global_sales_stdev))
    else:
        scaled_variable = preprocessing.scale(train_df.iloc[:,variable:(variable+1)].values)
    temp_array = []
    for value in scaled_variable:
        temp_array.append(value[0])
    scaled_columns.append(temp_array)


timeWindow = int((len(scaled_columns)-3)/4)
X_train = []
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
    X_train.append(observation)

X_train = np.asarray(X_train)
Y_train = standarize(train_df['sales_t'],global_sales_mean,global_sales_stdev)


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
    else:
        scaled_variable = preprocessing.scale(test_df.iloc[:,variable:(variable+1)].values)
    temp_array = []
    for value in scaled_variable:
        temp_array.append(value[0])
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
Y_destandarized = test_df.iloc[:,2]


###################################################################

# The LSTM architecture
model = Sequential()
model.add(LSTM(units=180, activation='sigmoid', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, return_sequences=True, input_shape=(timeWindow,6)))
model.add(LSTM(units=300, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, return_sequences=True))
model.add(LSTM(units=300, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, return_sequences=True))
model.add(LSTM(units=180, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None))
model.add(Dense(units=1))
model.add(Activation('linear'))

#opt = optimizers.Adam(beta_1=0.975, beta_2=0.9999)
model.compile(optimizer='adam',loss='mean_squared_error')
model.summary()

# Fitting to the training set
callbacks_val_loss = [ModelCheckpoint(filepath='../Files/Trained_ANNs/best_temporal_model.h5', monitor='val_loss', save_best_only=True)]
epochs = 10
log = model.fit(X_train, Y_train, callbacks=callbacks_val_loss, epochs=epochs, batch_size=128, validation_data=(X_test,Y_test))

###################################################################


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
ann_predictions_destandarized_positive[-1]

# Mean squared errors
error_of_mean_test = mean_squared_error(Y_test, mean_predictions_test)
error_of_mean_destandarized = mean_squared_error(Y_destandarized, mean_predictions_destandarized)
error_of_ann_test = mean_squared_error(Y_test, ann_predictions_test)
error_of_ann_destandarized = mean_squared_error(Y_destandarized, ann_predictions_destandarized_positive)
print('MSE of standarized mean predictions: ' + str(error_of_mean_test))
print('MSE of standarized ANN predictions: ' + str(error_of_ann_test))
print()
print('MSE of deseasonalized mean predictions: ' + str(error_of_mean_destandarized))
print('MSE of deseasonalized ANN predictions: ' + str(error_of_ann_destandarized))


# Plot learning process
results=pd.DataFrame(log.history)
results.plot(figsize=(8, 5), color = ['#fc4f30','#008fd5'])
plt.hlines(error_of_mean_test,xmin=0,xmax=epochs,label='mean_perdictions_loss',colors='green')
plt.grid(True)
plt.xlabel ("Epochs")
plt.ylabel ("Mean Squared Error")
plt.legend(['loss','val_loss','mean_predictions_loss'])
plt.xticks(np.arange(0, epochs+10, 10))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()


# Save model
model.save('../Files/Trained_ANNs/model_standarized_LOSS.h5')