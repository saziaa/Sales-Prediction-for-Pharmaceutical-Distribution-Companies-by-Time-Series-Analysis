#!/usr/bin/env python
# coding: utf-8

# In[433]:


import warnings

import numpy as np
from numpy import array
import pandas as pd
from pandas import concat
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import ParameterGrid


# In[1]:


get_ipython().system('pip install pandas-profiling')


# In[405]:


import pandas as pd


# In[406]:


from pandas_profiling import ProfileReport


# In[407]:


import pandas as pd
# Import the data
df_monthly = pd.read_csv("C:\\Users\\User\\Desktop\\Ryerson\\CIND 820\\Dataset\\salesmonthly.csv")
df_monthly['datum'] = pd.to_datetime(df_monthly['datum'])
df_monthly.rename(columns = {'datum':'Date'}, inplace = True)
# Set the date as index 
df_monthly = df_monthly.set_index('Date')
df_monthly.head()


# In[164]:


profile = ProfileReport(df_monthly,title='Pandas Profiling Report',explorative=True)


# In[165]:


profile.to_widgets()


# In[11]:


profile.to_file(output_file='pharma_report.html')


# In[408]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
df_monthly.plot(figsize=(14,6))


# # Check for common time series pattern

# ### 1.Visualize Data

# ### a.Checking Trend

# In[409]:


import warnings
import matplotlib.pyplot as plt
y = df_monthly['M01AB']
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label='Monthly')
ax.plot(y.resample('Y').mean(),marker='o', markersize=8, linestyle='-', label='Yearly Mean Resample')
ax.set_ylabel('M01AB')
ax.legend();


# In[410]:


import pandas as pd
# Import the data
df_monthly = pd.read_csv("C:\\Users\\User\\Desktop\\Ryerson\\CIND 820\\Dataset\\salesmonthly.csv")
df_monthly['datum'] = pd.to_datetime(df_monthly['datum'])
df_monthly.rename(columns = {'datum':'Date'}, inplace = True)
# Set the date as index 
df_monthly = df_monthly.set_index('Date')
df_monthly.head()

plt.style.use("fivethirtyeight")
df_monthly.plot(subplots=True, figsize=(12, 15))


# ### b.Cheking Seasonality

# In[411]:


# Import Data
df_monthly = pd.read_csv("C:\\Users\\User\\Desktop\\Ryerson\\CIND 820\\Dataset\\salesmonthly.csv", parse_dates=['datum'], index_col='datum')
df_monthly.reset_index(inplace=True)
df_monthly.rename(columns = {'datum':'Date'}, inplace = True)

# Prepare data
df_monthly['year'] = [d.year for d in df_monthly.Date]
df_monthly['month'] = [d.strftime('%b') for d in df_monthly.Date]
years = df_monthly['year'].unique()


# In[412]:


import seaborn as sns

# Splitting the plot into (1,2) subplots
# and initializing them using fig and ax
# variables
fig, ax = plt.subplots(nrows=1, ncols=2,
                       figsize=(15, 6))
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['year'],
            df_monthly["M01AB"], ax=ax[0])
 
# Defining the title and axes names
ax[0].set_title('Year-wise Box Plot for M01AB',
                fontsize=20, loc='center')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('M01AB sales')
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['month'],
            df_monthly["M01AB"], ax=ax[1])
 
# Defining the title and axes names
ax[1].set_title('Month-wise Box Plot for M01AB',
                fontsize=20, loc='center')
ax[1].set_xlabel('Month')
ax[1].set_ylabel('M01AB sales')
 
# rotate the ticks and right align them
fig.autofmt_xdate()


# In[323]:


import seaborn as sns

# Splitting the plot into (1,2) subplots
# and initializing them using fig and ax
# variables
fig, ax = plt.subplots(nrows=1, ncols=2,
                       figsize=(15, 6))
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['year'],
            df_monthly["M01AE"], ax=ax[0])
 
# Defining the title and axes names
ax[0].set_title('Year-wise Box Plot for M01AE',
                fontsize=20, loc='center')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('M01AE sales')
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['month'],
            df_monthly["M01AE"], ax=ax[1])
 
# Defining the title and axes names
ax[1].set_title('Month-wise Box Plot for M01AE',
                fontsize=20, loc='center')
ax[1].set_xlabel('Month')
ax[1].set_ylabel('M01AE sales')
 
# rotate the ticks and right align them
fig.autofmt_xdate()


# In[324]:


import seaborn as sns

# Splitting the plot into (1,2) subplots
# and initializing them using fig and ax
# variables
fig, ax = plt.subplots(nrows=1, ncols=2,
                       figsize=(15, 6))
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['year'],
            df_monthly["N02BA"], ax=ax[0])
 
# Defining the title and axes names
ax[0].set_title('Year-wise Box Plot for N02BA',
                fontsize=20, loc='center')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('N02BA sales')
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['month'],
            df_monthly["N02BA"], ax=ax[1])
 
# Defining the title and axes names
ax[1].set_title('Month-wise Box Plot for N02BA',
                fontsize=20, loc='center')
ax[1].set_xlabel('Month')
ax[1].set_ylabel('N02BA sales')
 
# rotate the ticks and right align them
fig.autofmt_xdate()


# In[325]:


import seaborn as sns

# Splitting the plot into (1,2) subplots
# and initializing them using fig and ax
# variables
fig, ax = plt.subplots(nrows=1, ncols=2,
                       figsize=(15, 6))
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['year'],
            df_monthly["N02BE"], ax=ax[0])
 
# Defining the title and axes names
ax[0].set_title('Year-wise Box Plot for N02BE',
                fontsize=20, loc='center')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('N02BE sales')
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['month'],
            df_monthly["N02BE"], ax=ax[1])
 
# Defining the title and axes names
ax[1].set_title('Month-wise Box Plot for N02BE',
                fontsize=20, loc='center')
ax[1].set_xlabel('Month')
ax[1].set_ylabel('N02BE sales')
 
# rotate the ticks and right align them
fig.autofmt_xdate()


# In[326]:


import seaborn as sns

# Splitting the plot into (1,2) subplots
# and initializing them using fig and ax
# variables
fig, ax = plt.subplots(nrows=1, ncols=2,
                       figsize=(15, 6))
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['year'],
            df_monthly["N05B"], ax=ax[0])
 
# Defining the title and axes names
ax[0].set_title('Year-wise Box Plot for N05B',
                fontsize=20, loc='center')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('N05B sales')
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['month'],
            df_monthly["N05B"], ax=ax[1])
 
# Defining the title and axes names
ax[1].set_title('Month-wise Box Plot for N05B',
                fontsize=20, loc='center')
ax[1].set_xlabel('Month')
ax[1].set_ylabel('N05B sales')
 
# rotate the ticks and right align them
fig.autofmt_xdate()


# In[327]:


import seaborn as sns

# Splitting the plot into (1,2) subplots
# and initializing them using fig and ax
# variables
fig, ax = plt.subplots(nrows=1, ncols=2,
                       figsize=(15, 6))
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['year'],
            df_monthly["N05C"], ax=ax[0])
 
# Defining the title and axes names
ax[0].set_title('Year-wise Box Plot for N05C',
                fontsize=20, loc='center')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('N05C sales')
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['month'],
            df_monthly["N05C"], ax=ax[1])
 
# Defining the title and axes names
ax[1].set_title('Month-wise Box Plot for N05C',
                fontsize=20, loc='center')
ax[1].set_xlabel('Month')
ax[1].set_ylabel('N025C sales')
 
# rotate the ticks and right align them
fig.autofmt_xdate()


# In[328]:


import seaborn as sns

# Splitting the plot into (1,2) subplots
# and initializing them using fig and ax
# variables
fig, ax = plt.subplots(nrows=1, ncols=2,
                       figsize=(15, 6))
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['year'],
            df_monthly["R03"], ax=ax[0])
 
# Defining the title and axes names
ax[0].set_title('Year-wise Box Plot for R03',
                fontsize=20, loc='center')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('R03 sales')
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['month'],
            df_monthly["R03"], ax=ax[1])
 
# Defining the title and axes names
ax[1].set_title('Month-wise Box Plot for R03',
                fontsize=20, loc='center')
ax[1].set_xlabel('Month')
ax[1].set_ylabel('R03 sales')
 
# rotate the ticks and right align them
fig.autofmt_xdate()


# In[329]:


import seaborn as sns

# Splitting the plot into (1,2) subplots
# and initializing them using fig and ax
# variables
fig, ax = plt.subplots(nrows=1, ncols=2,
                       figsize=(15, 6))
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['year'],
            df_monthly["R06"], ax=ax[0])
 
# Defining the title and axes names
ax[0].set_title('Year-wise Box Plot for R06',
                fontsize=20, loc='center')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('R06 sales')
 
# Using Seaborn Library for Box Plot
sns.boxplot(df_monthly['month'],
            df_monthly["R06"], ax=ax[1])
 
# Defining the title and axes names
ax[1].set_title('Month-wise Box Plot for R06',
                fontsize=20, loc='center')
ax[1].set_xlabel('Month')
ax[1].set_ylabel('R06 sales')
 
# rotate the ticks and right align them
fig.autofmt_xdate()


# ### 2.Decompose data

# In[413]:


import pandas as pd
# Import the data
df_monthly = pd.read_csv("C:\\Users\\User\\Desktop\\Ryerson\\CIND 820\\Dataset\\salesmonthly.csv")
df_monthly['datum'] = pd.to_datetime(df_monthly['datum'])
df_monthly.rename(columns = {'datum':'Date'}, inplace = True)
# Set the date as index 
df_monthly = df_monthly.set_index('Date')


# In[414]:


import statsmodels.api as sm

# graphs to show seasonal_decompose
def seasonal_decompose (y):
    decomposition = sm.tsa.seasonal_decompose(y, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()


# In[415]:


seasonal_decompose(y)


# In[204]:


y1 = df_monthly['M01AE']
# graphs to show seasonal_decompose
def seasonal_decompose (y1):
    decomposition = sm.tsa.seasonal_decompose(y1, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()


# In[205]:


seasonal_decompose(y1)


# In[206]:


y2 = df_monthly['N02BA']
def seasonal_decompose (y2):
    decomposition = sm.tsa.seasonal_decompose(y2, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()


# In[200]:


seasonal_decompose(y2)


# In[207]:


y3 = df_monthly['N02BE']
def seasonal_decompose (y3):
    decomposition = sm.tsa.seasonal_decompose(y3, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()


# In[208]:


seasonal_decompose(y3)


# In[209]:


y4 = df_monthly['N05B']
def seasonal_decompose (y4):
    decomposition = sm.tsa.seasonal_decompose(y4, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()


# In[210]:


seasonal_decompose(y4)


# In[211]:


y5 = df_monthly['N05C']
def seasonal_decompose (y5):
    decomposition = sm.tsa.seasonal_decompose(y5, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()


# In[212]:


seasonal_decompose(y5)


# In[213]:


y6 = df_monthly['R03']
def seasonal_decompose (y6):
    decomposition = sm.tsa.seasonal_decompose(y6, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()


# In[214]:


seasonal_decompose(y6)


# In[215]:


y7 = df_monthly['R06']
def seasonal_decompose (y7):
    decomposition = sm.tsa.seasonal_decompose(y7, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()


# In[216]:


seasonal_decompose(y7)


# # Check for Stationarity

# ### 1.Visualization

# ### Plot rolling statistic for testing stationarity

# In[416]:


import math
df_monthly = pd.read_csv("C:\\Users\\User\\Desktop\\Ryerson\\CIND 820\\Dataset\\salesmonthly.csv")
cols_plot = ['M01AB','M01AE','N02BA','N02BE', 'N05B','N05C','R03','R06']

rolling_mean = df_monthly[cols_plot].rolling(window=12, center=True).mean()
rolling_std = df_monthly[cols_plot].rolling(window=12, center=True).std()
subplotindex=0
numrows=4
numcols=2
fig, ax = plt.subplots(numrows, numcols, figsize=(18, 12))
plt.subplots_adjust(wspace=0.1, hspace=0.3)

for x in cols_plot:
    rowindex=math.floor(subplotindex/numcols)
    colindex=subplotindex-(rowindex*numcols)
    ax[rowindex,colindex].plot(df_monthly.loc[:,x], linewidth=0.5, label='Original Monthly sales')
    ax[rowindex,colindex].plot(rolling_mean.loc[:,x], label='Rolling Mean')
    
    ax[rowindex,colindex].plot(rolling_std.loc[:,x], color='0.5', linewidth=3, label='Rolling Std')
    ax[rowindex,colindex].set_ylabel('Sales')
    ax[rowindex,colindex].legend()
    ax[rowindex,colindex].set_title('Trends in '+x+' drugs sales');   
    subplotindex=subplotindex+1
    
plt.show()


# In[417]:


df_daily = pd.read_csv("C:\\Users\\User\\Desktop\\Ryerson\\CIND 820\\Dataset\\salesdaily.csv")
cols_plot = ['M01AB','M01AE','N02BA','N02BE', 'N05B','N05C','R03','R06']

rolling_mean = df_daily[cols_plot].rolling(window=30, center=True).mean()
rolling_std = df_daily[cols_plot].rolling(window=30, center=True).std()
subplotindex=0
numrows=4
numcols=2
fig, ax = plt.subplots(numrows, numcols, figsize=(18, 12))
plt.subplots_adjust(wspace=0.1, hspace=0.3)

for x in cols_plot:
    rowindex=math.floor(subplotindex/numcols)
    colindex=subplotindex-(rowindex*numcols)
    ax[rowindex,colindex].plot(df_daily.loc[:,x], linewidth=0.5, label='Original daily sales')
    ax[rowindex,colindex].plot(rolling_mean.loc[:,x], label='Rolling Mean')
    
    ax[rowindex,colindex].plot(rolling_std.loc[:,x], color='0.5', linewidth=3, label='Rolling Std')
    ax[rowindex,colindex].set_ylabel('Sales')
    ax[rowindex,colindex].legend()
    ax[rowindex,colindex].set_title('Trends in '+x+' drugs sales');   
    subplotindex=subplotindex+1
    
plt.show()


# ### 2.Augmented Dickey-Fuller Test

# In[418]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


# In[421]:


df_monthly = pd.read_csv("C:\\Users\\User\\Desktop\\Ryerson\\CIND 820\\Dataset\\salesmonthly.csv")

from statsmodels.tsa.stattools import adfuller

# Augmented Dickey-Fuller Test

def adf_test(timeseries):
    print("All Column results of Dickey-Fuller Test: ")
    for value in timeseries:
        dftest = adfuller(df_monthly[value], regression='ct',autolag="AIC")
        dfoutput = pd.Series(
            dftest[0:4],
            index=[
                "Test Statistic",
                "p-value",
                "#Lags Used",
                "Number of Observations Used",
            ],
        )
        for key, value in dftest[4].items():
            dfoutput["Critical Value (%s)" % key] = value
        print(dfoutput)
        print("")
        print("*********")
        print("")


# In[422]:


timeseries = ['M01AB','M01AE','N02BA','N02BE', 'N05B','N05C','R03','R06']
adf_test(timeseries)


# ### 3.KPSS

# In[423]:


from statsmodels.tsa.stattools import kpss

# KPSS Test

def kpss_test(timeseries):
    print("All column results of KPSS Test:")
    for value in timeseries:
        kpsstest = kpss(df_monthly[value], regression="ct", nlags="auto")
        kpss_output = pd.Series(
            kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
        )
        for key, value in kpsstest[3].items():
            kpss_output["Critical Value (%s)" % key] = value
        print(kpss_output)
        print("")
        print("*********")
        print("")
    


# In[424]:


timeseries = ['M01AB','M01AE','N02BA','N02BE', 'N05B','N05C','R03','R06']
kpss_test(timeseries)


# # Make the Data Stationary

# ## 1.Detrending

# In[425]:


### plot for Rolling Statistic for testing Stationarity
def test_stationarity(timeseries, title):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean() 
    rolstd = pd.Series(timeseries).rolling(window=12).std()
    
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(timeseries, label= title)
    ax.plot(rolmean, label='rolling mean');
    ax.plot(rolstd, label='rolling std (x10)');
    ax.legend()
pd.options.display.float_format = '{:.8f}'.format


# In[426]:


def ADF_test(timeseries, dataDesc):
    print(' > Is the {} stationary ?'.format(dataDesc))
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    print('Test statistic = {:.3f}'.format(dftest[0]))
    print('P-value = {:.3f}'.format(dftest[1]))
    print('Critical values :')
    for k, v in dftest[4].items():
        print('\t{}: {} - The data is {} stationary with {}% confidence'.format(k, v, 'not' if v<dftest[0] else '', 100-int(k[:-1])))


# In[427]:


def kpss_test(timeseries):
    print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="ct", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    print(kpss_output)


# In[428]:


df_monthly = pd.read_csv("C:\\Users\\User\\Desktop\\Ryerson\\CIND 820\\Dataset\\salesmonthly.csv")
y = df_monthly['M01AB']
# Detrending(to remove the underlying trend in the time series)
y_detrend =  (y - y.rolling(window=12).mean())/y.rolling(window=12).std()

test_stationarity(y_detrend,'de-trended data')
ADF_test(y_detrend,'de-trended data')


# In[375]:


y1 = df_monthly['M01AE']
# Detrending(to remove the underlying trend in the time series)
y1_detrend =  (y1 - y1.rolling(window=12).mean())/y1.rolling(window=12).std()

test_stationarity(y1_detrend,'de-trended data')
ADF_test(y1_detrend,'de-trended data')


# In[376]:


y2 = df_monthly['N02BA']
# Detrending(to remove the underlying trend in the time series)
y2_detrend =  (y2 - y2.rolling(window=12).mean())/y2.rolling(window=12).std()

test_stationarity(y2_detrend,'de-trended data')
ADF_test(y2_detrend,'de-trended data')


# In[377]:


y3 = df_monthly['N02BE']
# Detrending(to remove the underlying trend in the time series)
y3_detrend =  (y3 - y3.rolling(window=12).mean())/y3.rolling(window=12).std()

test_stationarity(y3_detrend,'de-trended data')
ADF_test(y3_detrend,'de-trended data')


# In[378]:


y4 = df_monthly['N05B']
# Detrending(to remove the underlying trend in the time series)
y4_detrend =  (y4 - y4.rolling(window=12).mean())/y4.rolling(window=12).std()

test_stationarity(y4_detrend,'de-trended data')
ADF_test(y4_detrend,'de-trended data')


# In[379]:


y5 = df_monthly['N05C']
# Detrending(to remove the underlying trend in the time series)
y5_detrend =  (y5 - y5.rolling(window=12).mean())/y5.rolling(window=12).std()

test_stationarity(y5_detrend,'de-trended data')
ADF_test(y5_detrend,'de-trended data')


# In[380]:


y6 = df_monthly['R03']
# Detrending(to remove the underlying trend in the time series)
y6_detrend =  (y6 - y6.rolling(window=12).mean())/y6.rolling(window=12).std()

test_stationarity(y6_detrend,'de-trended data')
ADF_test(y6_detrend,'de-trended data')


# In[381]:


y7 = df_monthly['R06']
# Detrending(to remove the underlying trend in the time series)
y7_detrend =  (y7 - y7.rolling(window=12).mean())/y7.rolling(window=12).std()

test_stationarity(y7_detrend,'de-trended data')
ADF_test(y7_detrend,'de-trended data')


# ## 2.Differencing

# In[386]:


# Differencing(to remove the underlying seasonal or cyclical patterns in the time series)
y_12lag =  y - y.shift(12)

test_stationarity(y_12lag,'12 lag differenced data')
ADF_test(y_12lag,'12 lag differenced data')


# In[387]:


# Differencing(to remove the underlying seasonal or cyclical patterns in the time series)
y1_12lag =  y1 - y1.shift(12)

test_stationarity(y1_12lag,'12 lag differenced data')
ADF_test(y1_12lag,'12 lag differenced data')


# In[388]:


# Differencing(to remove the underlying seasonal or cyclical patterns in the time series)
y2_12lag =  y2 - y2.shift(12)

test_stationarity(y2_12lag,'12 lag differenced data')
ADF_test(y2_12lag,'12 lag differenced data')


# In[389]:


# Differencing(to remove the underlying seasonal or cyclical patterns in the time series)
y3_12lag =  y3 - y3.shift(12)

test_stationarity(y3_12lag,'12 lag differenced data')
ADF_test(y3_12lag,'12 lag differenced data')


# In[391]:


# Differencing(to remove the underlying seasonal or cyclical patterns in the time series)
y4_12lag =  y4 - y4.shift(12)

test_stationarity(y4_12lag,'12 lag differenced data')
ADF_test(y4_12lag,'12 lag differenced data')


# In[392]:


# Differencing(to remove the underlying seasonal or cyclical patterns in the time series)
y5_12lag =  y5 - y5.shift(12)

test_stationarity(y5_12lag,'12 lag differenced data')
ADF_test(y5_12lag,'12 lag differenced data')


# In[393]:


# Differencing(to remove the underlying seasonal or cyclical patterns in the time series)
y6_12lag =  y6 - y6.shift(12)

test_stationarity(y6_12lag,'12 lag differenced data')
ADF_test(y6_12lag,'12 lag differenced data')


# In[394]:


# Differencing(to remove the underlying seasonal or cyclical patterns in the time series)
y7_12lag =  y7 - y7.shift(12)

test_stationarity(y7_12lag,'12 lag differenced data')
ADF_test(y7_12lag,'12 lag differenced data')


# ## 3.Combining Detrending and Differencing

# In[429]:


# Detrending + Differencing

y_12lag_detrend =  y_detrend - y_detrend.shift(12)

test_stationarity(y_12lag_detrend,'12 lag differenced de-trended data')
ADF_test(y_12lag_detrend,'12 lag differenced de-trended data')


# In[396]:


# Detrending + Differencing

y1_12lag_detrend =  y1_detrend - y1_detrend.shift(12)

test_stationarity(y1_12lag_detrend,'12 lag differenced de-trended data')
ADF_test(y1_12lag_detrend,'12 lag differenced de-trended data')


# In[397]:


# Detrending + Differencing

y2_12lag_detrend =  y2_detrend - y2_detrend.shift(12)

test_stationarity(y2_12lag_detrend,'12 lag differenced de-trended data')
ADF_test(y2_12lag_detrend,'12 lag differenced de-trended data')


# In[398]:


# Detrending + Differencing

y3_12lag_detrend =  y3_detrend - y3_detrend.shift(12)

test_stationarity(y3_12lag_detrend,'12 lag differenced de-trended data')
ADF_test(y3_12lag_detrend,'12 lag differenced de-trended data')


# In[399]:


# Detrending + Differencing

y4_12lag_detrend =  y4_detrend - y4_detrend.shift(12)

test_stationarity(y4_12lag_detrend,'12 lag differenced de-trended data')
ADF_test(y4_12lag_detrend,'12 lag differenced de-trended data')


# In[400]:


# Detrending + Differencing

y5_12lag_detrend =  y5_detrend - y5_detrend.shift(12)

test_stationarity(y5_12lag_detrend,'12 lag differenced de-trended data')
ADF_test(y5_12lag_detrend,'12 lag differenced de-trended data')


# In[401]:


# Detrending + Differencing

y6_12lag_detrend =  y6_detrend - y6_detrend.shift(12)

test_stationarity(y6_12lag_detrend,'12 lag differenced de-trended data')
ADF_test(y6_12lag_detrend,'12 lag differenced de-trended data')


# In[402]:


# Detrending + Differencing

y7_12lag_detrend =  y7_detrend - y7_detrend.shift(12)

test_stationarity(y7_12lag_detrend,'12 lag differenced de-trended data')
ADF_test(y7_12lag_detrend,'12 lag differenced de-trended data')


# # Autocorrelation analysis

# In[452]:


df_monthly = pd.read_csv("C:\\Users\\User\\Desktop\\Ryerson\\CIND 820\\Dataset\\salesmonthly.csv")
df_monthly['datum'] = pd.to_datetime(df_monthly['datum'])
df_monthly.rename(columns = {'datum':'Date'}, inplace = True)
# Set the date as index 
df_monthly = df_monthly.set_index('Date')

from statsmodels.graphics.tsaplots import plot_acf

subplotindex=0
numrows=4
numcols=2
fig, ax = plt.subplots(numrows, numcols, figsize=(18,12))
plt.subplots_adjust(wspace=0.1, hspace=0.3)
with plt.rc_context():
    plt.rc("figure", figsize=(18,12))
    for x in ['M01AB','M01AE','N02BA','N02BE', 'N05B','N05C','R03','R06']:
        rowindex=math.floor(subplotindex/numcols)
        colindex=subplotindex-(rowindex*numcols)
        plot_acf(df_monthly[x],lags=50,title=x, ax=ax[rowindex,colindex])
        subplotindex=subplotindex+1


# # Partial autocorrelation analysis
# 

# In[459]:


df_monthly = pd.read_csv("C:\\Users\\User\\Desktop\\Ryerson\\CIND 820\\Dataset\\salesmonthly.csv")
df_monthly['datum'] = pd.to_datetime(df_monthly['datum'])
df_monthly.rename(columns = {'datum':'Date'}, inplace = True)
# Set the date as index 
df_monthly = df_monthly.set_index('Date')

from statsmodels.graphics.tsaplots import plot_pacf

subplotindex=0
numrows=4
numcols=2
fig, ax = plt.subplots(numrows, numcols, figsize=(18,12))
plt.subplots_adjust(wspace=0.1, hspace=0.3)
with plt.rc_context():
    plt.rc("figure", figsize=(14,6))
    for x in ['M01AB','M01AE','N02BA','N02BE', 'N05B','N05C','R03','R06']:
        rowindex=math.floor(subplotindex/numcols)
        colindex=subplotindex-(rowindex*numcols)
        plot_pacf(df_monthly[x], lags=30, title=x, ax=ax[rowindex,colindex])
        subplotindex=subplotindex+1


# # Data Normalization

# In[470]:


df_monthly = pd.read_csv("C:\\Users\\User\\Desktop\\Ryerson\\CIND 820\\Dataset\\salesmonthly.csv")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
cols_to_norm = ['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']
df_monthly[cols_to_norm] = scaler.fit_transform(df_monthly[cols_to_norm])
df_monthly['datum'] = pd.to_datetime(df_monthly['datum'])
df_monthly.rename(columns = {'datum':'Date'}, inplace = True)
# Set the date as index 
df_monthly = df_monthly.set_index('Date')
df_monthly.head()


# # Building Time Series Prediction Model

# ## 1. ARIMA model

# ### 1.1 Choosing parameters for ARIMA model

# In[469]:


# load dataset
df_monthly = pd.read_csv("C:\\Users\\User\\Desktop\\Ryerson\\CIND 820\\Dataset\\salesmonthly.csv")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
cols_to_norm = ['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']
df_monthly[cols_to_norm] = scaler.fit_transform(df_monthly[cols_to_norm])
df_monthly['datum'] = pd.to_datetime(df_monthly['datum'])


# In[473]:


import warnings
warnings.filterwarnings("ignore")

# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)

evaluate_models(df_monthly.values, p_values, d_values, q_values)


# In[ ]:


# grid search ARIMA parameters for time series
import warnings
from math import sqrt
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
 
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = sqrt(mean_squared_error(test, predictions))
	return rmse
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
 
# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)


# In[487]:


# grid search ARIMA parameters for time series
import warnings
from math import sqrt
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
 
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    
    #make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    #calculating error
    error = mean_squared_error(test, predictions)
    return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(f, dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                except:
                    continue
    print(f+' - Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

p_values = range(0, 6)
d_values = range(0, 2)
q_values = range(0, 6)

warnings.filterwarnings("ignore")

df_monthly = pd.read_csv("C:\\Users\\User\\Desktop\\Ryerson\\CIND 820\\Dataset\\salesmonthly.csv")
df_monthly['datum'] = pd.to_datetime(df_monthly['datum'])
df_monthly.rename(columns = {'datum':'Date'}, inplace = True)
# Set the date as index 
df_monthly = df_monthly.set_index('Date')

for f in ['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']:
    evaluate_models(f, df_monthly[f].values, p_values, d_values, q_values)


# In[477]:


import statsmodels.api as sm
df_monthly = pd.read_csv("C:\\Users\\User\\Desktop\\Ryerson\\CIND 820\\Dataset\\salesmonthly.csv")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
cols_to_norm = ['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']
df_monthly[cols_to_norm] = scaler.fit_transform(df_monthly[cols_to_norm])
warnings.filterwarnings("ignore")
for x in ['M01AB','M01AE','N02BA','N02BE', 'N05B','N05C','R03','R06']:
    resDiff = sm.tsa.arma_order_select_ic(df[x], max_ar=5, max_ma=5, ic='aic', trend='c')
    print('ARMA(p,q,'+x+') =',resDiff['aic_min_order'],'is the best.')


# In[488]:


def evaluate_arima_model(X, arima_order):
    train_size = int(len(X) - 50)
    train, test = X[0:train_size], X[train_size:]
    model = ARIMA(train, order=arima_order)
    model_fit = model.fit()
    forecast = model_fit.predict(1,len(test))
    error = mean_squared_error(test, forecast)
    return error

def evaluate_models(f, dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                except:
                    continue
    print(f+' - Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

p_values = range(0, 6)
d_values = range(0, 2)
q_values = range(0, 6)

warnings.filterwarnings("ignore")

for f in ['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']:
    evaluate_models(f, df[f].values, p_values, d_values, q_values)


# In[ ]:





# # Building Time Series Prediction Model

# ### SARIMA model

# In[104]:


import itertools

def sarima_grid_search(y,seasonal_period):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2],seasonal_period) for x in list(itertools.product(p, d, q))]
    
    mini = float('+inf')
    
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()
                
                if results.aic < mini:
                    mini = results.aic
                    param_mini = param
                    param_seasonal_mini = param_seasonal

#                print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    print('The set of parameters with the minimum AIC is: SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini, mini))


# In[ ]:





# In[ ]:




