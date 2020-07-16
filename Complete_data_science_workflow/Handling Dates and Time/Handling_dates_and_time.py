# -*- coding: utf-8 -*-

"""
Created on mon Jul 16 7:29:11 2020

@author: Pritam
"""

import pandas as pd
import numpy as np
from pytz import all_timezones

#Converting strings to dates

date_strings = np.array(['03-04-2005 11:35 PM',
                         '23-05-2010 12:01 AM',
                         '04-09-2009 09:09 PM'])

#print([pd.to_datetime(date, format = '%d-%m-%Y %I:%M %p') for date in date_strings])
#print([pd.to_datetime(date, format = '%d-%m-%Y %I:%M %p',errors = 'coerce') for date in date_strings])

#Handling time zones

#print(pd.Timestamp('2017-05-01 06:00:00', tz = 'Europe/London'))

date = pd.Timestamp('2017-05-01 06:00:00')

date_in_london = date.tz_localize('Europe/London')

#print(date_in_london)

#print(date_in_london.tz_convert('Africa/abidjan'))

dates = pd.Series(pd.date_range('2/2/2002',periods = 3,freq = 'M'))

#print(dates.tz_localize('Africa/abidjan'))

#print(all_timezones[:2])

#Selecting Dates and time

dataframe = pd.DataFrame()

dataframe['date'] = pd.date_range('1/1/2001',periods = 100000,freq = 'H')

#print(dataframe[(dataframe['date'] > '2002-1-1 01:00:00') & (dataframe['date'] > '2002-1-1 04:00:00')])

#print(dataframe.set_index(dataframe['date']))

#print(dataframe.loc['2002-1-1 01:00:00':'2002-1-1 04:00:00'])

#Breaking up Date into Multiple features

dataframe = pd.DataFrame()

dataframe['date'] = pd.date_range('1/1/2001', periods=150, freq='W')

dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute

#print(dataframe.head(3))

dataframe = pd.DataFrame()

dataframe['Arrived'] = [pd.Timestamp('01-02-2017'),pd.Timestamp('01-04-2017')]
dataframe['Left'] = [pd.Timestamp('01-02-2017'),pd.Timestamp('01-06-2017')]

#print(dataframe['Left'] - dataframe['Arrived'])
#print(pd.Series(delta.days for delta in(dataframe['Left'] - dataframe['Arrived'])))

#Encoding days of the week

dates = pd.Series(pd.date_range("02/02/2002",periods = 3, freq  = 'M' )) 

#print(dates.dt.weekday_name)

#Creating Lagged feature

dataframe = pd.DataFrame()

dataframe['dates'] = pd.date_range('1/1/2001',periods = 5,freq = 'D')
dataframe['stock_price'] = [1.1,2.2,3.3,4.4,5.5]

dataframe['Previous_day_stockprice'] = dataframe['stock_price'].shift(1)

#print(dataframe)

#Using Rolling Time Windows

time_index = pd.date_range('01/01/2010',periods = 5, freq = 'M' )
dataframe = pd.DataFrame(index= time_index)

dataframe['stock_price'] = [1,2,3,4,5]

#print(dataframe.rolling(window = 2).mean())

#Handling missing data in time series

time_index = pd.date_range("01/01/2010", periods=5, freq="M")

dataframe = pd.DataFrame(index=time_index)

dataframe["Sales"] = [1.0,2.0,np.nan,np.nan,5.0]

print(dataframe.interpolate())
print(dataframe.ffill())
print(dataframe.bfill())


