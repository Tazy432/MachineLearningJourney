

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
#%matplotlib inline
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

from datetime import datetime

tech_list=['AAPL','GOOG','MSFT','AMZN']
end=datetime.now()
start=datetime(end.year-1,end.month,end.day)

for stock in tech_list:
    globals()[stock]=yf.download(stock,start,end)

company_list=[AAPL,GOOG,MSFT,AMZN]
company_name=["APPLE","GOOGLE","MICROSOFT","AMAZON"]
for company,com_name in zip(company_list,company_name):
    company["company_name"]=com_name
df=pd.concat(company_list,axis=0)
df.tail(10)
plt.figure(figsize=(15,10))
plt.subplots_adjust(top=1.25,bottom=1.2)

# plot the closing prices of the companies
for i,company in enumerate(company_list,1):
    plt.subplot(2,2,i)
    company['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Closing price of {tech_list[i-1]}")
plt.tight_layout()

# plot the volume of sales for each stock traded each day
plt.figure(figsize=(15,10))
plt.subplots_adjust(top=1.25,bottom=1.2)
for i,company in enumerate(company_list,1):
    plt.subplot(2,2,i)
    company['Volume'].plot()
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"Sales Volume for {tech_list[i-1]}")
plt.tight_layout()

# computing the moving average of stocks
ma_day=[10,20,50]
for ma in ma_day:
    for company in company_list:
        column_name=f"MA for {ma} days"
        company[column_name]=company['Adj Close'].rolling(ma).mean()
fig,axes=plt.subplots(nrows=2,ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)
AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(ax=axes[0,0])
axes[0,0].set_title('APPLE')

GOOG[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(ax=axes[0,1])
axes[0,1].set_title('GOOGLE')

MSFT[['Adj Close', 'MA for 10 days', 'MA for 20 days','MA for 50 days']].plot(ax=axes[1,0])
axes[1,0].set_title('MICROSOFT')

AMZN[['Adj Close', 'MA for 10 days','MA for 20 days','MA for 50 days']].plot(ax=axes[1,1])
axes[1,1].set_title('AMAZON')

plt.tight_layout()

# daily return of the stock on average

for company in company_list:
    company['Daily return']=company['Adj Close'].pct_change()
fig,axes=plt.subplots(nrows=2,ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

AAPL['Daily return'].plot(ax=axes[0,0],legend=True,linestyle='--',marker='o')
axes[0,0].set_title('APPLE')

GOOG['Daily return'].plot(ax=axes[0,1],legend=True,linestyle='--',marker='o')
axes[0,1].set_title('GOOGLE')

MSFT['Daily return'].plot(ax=axes[1,0],legend=True,linestyle='--',marker='o')
axes[1,0].set_title('MICROSOFT')

AMZN['Daily return'].plot(ax=axes[1,1],legend=True,linestyle='--',marker='o')
axes[1,1].set_title('AMAZON')

fig.tight_layout()

plt.figure(figsize=(12,9))
for i,company in enumerate(company_list,1):
    plt.subplot(2,2,i)
    company['Daily return'].hist(bins=50)
    plt.xlabel('Daily return')
    plt.ylabel('Counts')
    plt.title(f'{company_name[i-1]}')
plt.tight_layout()

# analyze the return of all the stocks in the list
closing_df=pdr.get_data_yahoo(tech_list,start=start,end=end)['Adj Close']
tech_rets=closing_df.pct_change()
tech_rets.head()

# analyze the correlation between all the stocks in the list
sns.pairplot(tech_rets,kind='reg')

# correlation plot to get numerical values for the correlation between the stocks daily return values
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.heatmap(tech_rets.corr(),annot=True,cmap='summer')
plt.title('Correlation of stock return')

plt.subplot(2,2,2)
sns.heatmap(closing_df.corr(),annot=True,cmap='summer')
plt.title('Correlation of stock closing price')

# compunting how much value we put at risk when investing in a particular stock
rets=tech_rets.dropna()
area=np.pi*20
plt.figure(figsize=(10,8))
plt.scatter(rets.mean(),rets.std(),s=area)
plt.xlabel('Expected return')
plt.ylabel('Risk')
for label,x,y in zip(rets.columns,rets.mean(),rets.std()):
    plt.annotate(label,xy=(x,y),xytext=(50,50),textcoords='offset points',ha='right',va='bottom',arrowprops=dict(arrowstyle='-',color='blue',connectionstyle='arc3,rad=-0.3'))

# predict the closing price of apple
df=pdr.get_data_yahoo('AAPL',start='2012-01-01',end=datetime.now())
df
plt.figure(figsize=(16,6))
plt.title('Close price history')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close price $',fontsize=18)
data=df.filter(['Close'])
dataset=data.values
training_data_len=int(np.ceil(len(dataset)*0.95))

#we now scale the data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

# create the training dataset , and the scaled training data set
train_data=scaled_data[0:int(training_data_len), :]
x_train=[]
y_train=[]
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense,LSTM

# we now bulid the lstm model
model=Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(64,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=1,epochs=1)

#create the testing data set
test_data=scaled_data[training_data_len-60: , :]
x_test=[]
y_test=dataset[training_data_len:, :]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)
rmse=np.sqrt(np.mean(((predictions-y_test)**2)))

#ploting the data
train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price $',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()





if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
