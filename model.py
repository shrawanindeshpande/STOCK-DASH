#import dependencies
import quandl
import pandas as pd
import pandas_datareader as data
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import plot, sca
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential
#get stock data
start='2012-1-1'
end='2022-1-3'
df=data.DataReader('AAPL','yahoo',start,end)
df=df.reset_index()
#print(df)
df=df.drop(['Date','Adj Close'],axis=1)
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
# plot.figure(figsize=(12,6))
# plot.plot(df.Close)
# plot.plot(ma100,'r')
#print(ma61)

#spliting data into Training and Testing
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
# print(data_training.shape)
# print(data_testing.shape)
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)
#print(data_training_array)

x_train=[]
y_train=[]
for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
#print(x_train)
x_train,y_train=np.array(x_train),np.array(y_train)

#ml model

model=Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='relu',return_sequences=True))
model.add(Dropout(0.4))


model.add(LSTM(units=120,activation='relu',return_sequences=True))
model.add(Dropout(0.5))

model.add(Dense(units=1))

#print(model.summary())
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=50)
model.save('keras_model.h5')

#print(data_testing.head())

past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)


input_data=scaler.fit_transform(final_df)
print(input_data.shape)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test,y_test=np.array(x_test),np.array(y_test)
print(x_test.shape)
print(y_test.shape)

#making prediction

y_predicted=model.predict(x_test)
#print(y_predicted)
#print(y_test)
scaler.scale_
# print(scaler.scale_)
scale_factor=1/0.00682769
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor
# print(scale_factor)
print(y_predicted)
# print(y_test)