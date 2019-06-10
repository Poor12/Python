import pandas as pd
import numpy as np
#load csv
#dataframe=pd.read_csv("test.csv")
#print(dataframe.head(2))

#load excel
#dataframe=pd.read_excel("test.xlsx,sheetname=0,header=1)

#load json
#dataframe=pd.read_json("test.json",orient='columns')
#print(dataframe.head(2))

#load database
#from sqlalchemy import create_engine
#database_connection=create_engine('sqlite:///sample.db')
#dataframe=pd.read_sql_query('select * from data',database_connection)

url='https://tinyurl.com/titanic-csv'
dataframe=pd.read_csv(url)
print(dataframe.head(5))
print(dataframe.iloc[0:5]) #前五行
dataframe=dataframe.set_index(dataframe['Name'])
print(dataframe.loc['Allen, Miss Elisabeth Walton'])#以index来定位
print(dataframe.shape)
print(dataframe.describe())
#apply a function over all elements in a column
print(dataframe)
def uppercase(x):
    return x.upper()
print(dataframe['Name'].apply(uppercase)[0:2])
#条件选择行
print(dataframe[(dataframe['SexCode']==1)&(dataframe['Age']>=40)].head(2))
#值的替换
print(dataframe['Sex'].replace(['female','male'],['woman','man']).head(2))
#列的rename
dataframe.rename(columns={'PClass':'Passenger Class'})
#统计
print("Max：",dataframe['Age'].max())
print("Mean:",dataframe['Age'].mean())
print("Count",dataframe['Age'].count())
#找到不重复的值
print(dataframe['Sex'].unique())
print(dataframe['Sex'].value_counts())
#找到丢失的数据
print(dataframe[dataframe['Age'].isnull()].head(2))
dataframe['Age']=dataframe['Age'].replace(np.nan,'male')
print(dataframe[dataframe['Age'].isnull()].head(2))
#删除列
dataframe.drop(['Age','Sex'],axis=1)
#删除行
dataframe[dataframe['Sex']!='male']
#删除重复行
dataframe.drop_duplicates() #完全匹配的元组
dataframe.drop_duplicates(subset=['Sex'])
dataframe.drop_duplicates(subset=['Sex'],keep='last')#保留最后的元组
#group rows by value
print(dataframe.groupby('Sex').mean())
print(dataframe.groupby('Sex').apply(lambda x:x.mean()))
#group rows by time
time_index=pd.date_range('06/06/2017',periods=100000,freq='30s')
dataframe=pd.DataFrame(index=time_index)
dataframe['Sale_Amount']=np.random.randint(1,10,100000)
print(dataframe.resample('W').sum())
print(dataframe.resample('M',label='left').count())

#连结dataframe
data_a={'id':['1','2','3'],'first':['Alex','Amy','Allen']}
dataframe_a=pd.DataFrame(data_a,columns=['id','first',])
data_b={'id':['4','5','6'],'first':['Billy','Brian','Bran']}
dataframe_b=pd.DataFrame(data_b,columns=['id','first'])
dataframe_ab=pd.concat([dataframe_a,dataframe_b],axis=1)
print(dataframe_ab)
data_c={'id':['1','2','3'],'second':['Billy','Brian','Bran']}
dataframe_c=pd.DataFrame(data_c,columns=['id','second'])
dataframe_ba=pd.merge(dataframe_a,dataframe_c,on='id',how='outer')
print(dataframe_ba)


#create dataframe
dataframe=pd.DataFrame()
dataframe['Name']=['Jacky Jackson','Steven Stevenson']
dataframe['Age']=[38,25]
dataframe['Driver']=[1,0]
print(dataframe['Age'])
#create row
#append的时候，不是对直接对原始数据进行操作，而是在内存中建立副本
new_person=pd.Series({'Name':'Molly Mooney','Age':40,'Driver':1})
dataframe=dataframe.append(new_person,ignore_index=1)
print(dataframe)

