import pandas as pd
import numpy as np
from numpy import hstack
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
#from keras.layers import Dropout
import keras.optimizers as optimizers
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error
import statistics
from keras.callbacks import ModelCheckpoint
from scipy.stats import boxcox


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

def applyBoxCox(dataframe,column):
    filtered = dataframe[column].values
    filtered_1D = []
    for row in filtered:
        filtered_1D.append(row)
    transformed = boxcox(filtered_1D,0)
    return transformed


train_df = pd.read_csv("../Files/Completely_processed_data/shaped_input_train_5_30_100.csv")

no_negative_df = train_df[(train_df >= 0).all(1)]
no_cero_df = no_negative_df.replace(0,0.0001)
train_df = no_cero_df
train_df = train_df.sample(frac=1).reset_index(drop=True)

splitter = int(len(train_df) * 0.2)
validation_df = train_df.iloc[:splitter,:]
train_df = train_df.iloc[splitter:,:]

sales_mean = sum(train_df.iloc[:,2].values)/len(train_df.iloc[:,2].values)
sales_stdev = statistics.stdev(train_df.iloc[:,2].values)


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
    elif 'relative_shop_size' in column_name:
        scaled_variable = preprocessing.normalize(train_df.iloc[:,variable:(variable+1)])
    else:
        transformed = applyBoxCox(train_df,column_name)
        scaled_variable = preprocessing.scale(transformed)
    temp_array = []
    for value in scaled_variable:
        try:
            temp_array.append(value[0][0])
        except:
            temp_array.append(value)
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
for variable in range(2,len(validation_df.columns)):
    column_name = validation_df.columns[variable]
    if 'sales_' in column_name and column_name != 'sales_t':
        for value in validation_df.iloc[:,variable:(variable+1)].values:
            allSales.append(value[0])

global_sales_mean = sum(allSales) / len(allSales)
global_sales_stdev = statistics.stdev(allSales)

scaled_columns = []
for variable in range(2,len(validation_df.columns)):
    column_name = validation_df.columns[variable]
    if 'sales_' in column_name:
        to_log = [column_name]
        df_log = validation_df[to_log].applymap(lambda x: np.log(x+1))
        scaled_variable = np.asarray(standarizeArray(df_log.values,global_sales_mean,global_sales_stdev))
    elif 'relative_shop_size' in column_name:
        scaled_variable = preprocessing.normalize(validation_df.iloc[:,variable:(variable+1)])
    else:
        transformed = applyBoxCox(validation_df,column_name)
        scaled_variable = preprocessing.scale(transformed)
    temp_array = []
    for value in scaled_variable:
        try:
            temp_array.append(value[0][0])
        except:
            temp_array.append(value)
    scaled_columns.append(temp_array)


timeWindow = int((len(scaled_columns)-3)/4)
X_validation = []
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
    X_validation.append(observation)

X_validation = np.asarray(X_validation)
Y_validation = standarize(validation_df['sales_t'],global_sales_mean,global_sales_stdev)
Y_destandarized = validation_df.iloc[:,2]


############################################################

# The LSTM architecture
model = Sequential()
model.add(LSTM(units=120, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='orthogonal', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=regularizers.l2(0.00005), recurrent_regularizer=regularizers.l2(0.00005), bias_regularizer=None, activity_regularizer=None, return_sequences=True, input_shape=(timeWindow,6)))
#model.add(Dropout(0.2))
model.add(LSTM(units=200, activation='tanh', use_bias=True, kernel_initializer='orthogonal', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=regularizers.l2(0.00005), recurrent_regularizer=regularizers.l2(0.00005), bias_regularizer=None, activity_regularizer=None, return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(units=120, activation='tanh', use_bias=True, kernel_initializer='orthogonal', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=regularizers.l2(0.00005), recurrent_regularizer=regularizers.l2(0.00005), bias_regularizer=None, activity_regularizer=None))
#model.add(Dropout(0.2))
model.add(Dense(units=1))
model.add(Activation('linear'))

opt = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.975, clipnorm=1, clipvalue=0.5)
#opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1, clipvalue=0.5)
model.compile(optimizer=opt,loss='mean_squared_error')
model.summary()

# Fitting to the training set
callbacks_val_loss = [ModelCheckpoint(filepath='../Files/Trained_ANNs/best_temporal_model.h5', monitor='val_loss', save_best_only=True)]
epochs = 40
log = model.fit(X_train, Y_train, callbacks=callbacks_val_loss, epochs=epochs, batch_size=128, validation_data=(X_validation,Y_validation))

############################################################


# Mean price as prediction
mean_predictions_validation = []
for observation in X_validation:
    prices = []
    for variable in observation:
        prices.append(variable[2])
    sumOfPrices = 0
    for price in prices:
        sumOfPrices += price
    mean = sumOfPrices / len(prices)
    mean_predictions_validation.append(mean)
mean_predictions_destandarized = destandarize(mean_predictions_validation,global_sales_mean,global_sales_stdev)


# ANN predictions
ann_predictions_validation = []
predicted = (model.predict(X_validation)).tolist()
for prediction in predicted:
    ann_predictions_validation.append(prediction[0])

ann_predictions_destandarized_positive = []
ann_predictions_destandarized = destandarize(ann_predictions_validation,global_sales_mean,global_sales_stdev)
for prediction in ann_predictions_destandarized:
    if prediction < 0:
        ann_predictions_destandarized_positive.append(0)
    else:
        ann_predictions_destandarized_positive.append(prediction)
ann_predictions_destandarized_positive[-1]


# Mean squared errors
error_of_mean_validation = mean_squared_error(Y_validation, mean_predictions_validation)
error_of_mean_destandarized = mean_squared_error(Y_destandarized, mean_predictions_destandarized)
error_of_ann_validation = mean_squared_error(Y_validation, ann_predictions_validation)
error_of_ann_destandarized = mean_squared_error(Y_destandarized, ann_predictions_destandarized_positive)
print('MSE of standarized mean predictions: ' + str(error_of_mean_validation))
print('MSE of standarized ANN predictions: ' + str(error_of_ann_validation))
print()
print('MSE of deseasonalized mean predictions: ' + str(error_of_mean_destandarized))
print('MSE of deseasonalized ANN predictions: ' + str(error_of_ann_destandarized))


# Plot learning process
results=pd.DataFrame(log.history)
results.plot(figsize=(8, 5), color = ['#fc4f30','#008fd5'])
plt.hlines(error_of_mean_validation,xmin=0,xmax=epochs,label='mean_perdictions_loss',colors='green')
plt.grid(True)
plt.xlabel ("Epochs")
plt.ylabel ("Mean Squared Error")
plt.legend(['validation_loss','train_loss','mean_predictions_loss'])
plt.xticks(np.arange(0, epochs+10, 10))
plt.yticks(np.arange(0, 1.2, 0.1))
plt.gca().set_ylim(0, 1.1)
plt.show()


model.save('../Files/Trained_ANNs/validation_standarizedLog_MSE.h5')