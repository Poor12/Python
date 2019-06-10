import numpy as np
import pandas as pd
data_strings=np.array(['03-04-2005 11:35 PM',
                       '23-05-2010 12:01 AM',
                       '04-09-2009 09:09 PM'])
time=[pd.to_datetime(date,format='%d-%m-%Y %I:%M %p') for date in data_strings]
print(time)
time2=time[0].tz_localize('Europe/London')
print(time2)

dataframe=pd.DataFrame()
dataframe['date']=pd.date_range('1/1/2001',periods=100000,freq='H')
print(dataframe[(dataframe['date']>'2002-1-1 01:00:00')&(dataframe['date']<'2002-1-1 04:00:00')])

dataframe=pd.DataFrame()
dataframe['date']=pd.date_range('1/1/2001',periods=150,freq='W')
dataframe['year']=dataframe['date'].dt.year
dataframe['month']=dataframe['date'].dt.month
dataframe['day']=dataframe['date'].dt.day
dataframe['hour']=dataframe['date'].dt.hour
dataframe['minute']=dataframe['date'].dt.minute
print(dataframe.head(3))

#时间差
dataframe=pd.DataFrame()
dataframe['Arrived']=[pd.Timestamp('01-01-2017'),pd.Timestamp('01-04-2017')]
dataframe['left']=[pd.Timestamp('01-01-2017'),pd.Timestamp('01-06-2017')]
print(dataframe['left']-dataframe['Arrived'])

dates=pd.Series(pd.date_range("2/2/2019",periods=3,freq="M"))
print(dates)
print(dates.dt.weekday_name)

#lagged feature
dataframe=pd.DataFrame()
dataframe["dates"]=pd.date_range("1/1/2001",periods=5,freq="D")
dataframe["stock_price"]=[1.1,2.2,3.3,4.4,5.5]
dataframe["previous_price"]=dataframe["stock_price"].shift(1)
print(dataframe)

#处理丢失的时间数据
time_index=pd.date_range("01/01/2010",periods=5,freq="M")
dataframe=pd.DataFrame(index=time_index)
dataframe['Sales']=[1.0,2.0,np.nan,np.nan,5.0]
print(dataframe.interpolate())
print(dataframe.ffill())