# -*- coding: utf-8 -*-
"""
@author: rnicatusmac
"""
"""
Import required libraries
"""
import pyhdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dateutil.parser import parse 
import matplotlib as mpl
import pmdarima as pm
from data_load_utils import Settings
%matplotlib qt 
"""
Setup visualization figure
"""
plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})
"""
Connect to HANA DB
"""

conn_object = Settings.load_config("confige2edata.ini")

connection = pyhdb.connect(
    host=conn_object["url"],
    port=conn_object["port"],
    user=conn_object["user"],
    password=conn_object["pwd"]
)
conn_object={}
cursor = connection.cursor()

"""
                        PART 1 - Python Open Source ML Model

"""

"""
Retrieve DB Data
"""
cursor.execute('select "DATE","AMOUNT" from "ZATM"."ZATM_TRANS_CONS_ALL_BY_DAY" where YEAR(date) <> 1998')
cursor_data = cursor.fetchall()
"""
Store data in Pandas Dataframe and do necessary adjustments to feed custom ML Model
"""
df = pd.DataFrame(cursor_data, columns=['date', 'amount'])
df['date'] = pd.to_datetime(df['date'])
df.index = df['date'] 
df.drop(['date'], axis=1, inplace = True)
df=df.resample('M').sum()
"""
Visualize data before using it for ML 
"""
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Amount', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()
plot_df(df, x=df.index, y=df.amount, title='Monthly Amounts')  


fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)
# Usual Differencing
axes[0].plot(df[:], label='Original Series')
axes[0].plot(df[:].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)


# Seasinal Dei
axes[1].plot(df[:], label='Original Series')
axes[1].plot(df[:].diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('ATM Monthly Transactions', fontsize=16)
plt.show()

"""
Train ML Model
"""
smodel = pm.auto_arima(df, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None,D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)
"""
ARIMA = Auto-Regressive Integrated Moving Average
An ARIMA model can be understood by outlining each of its components as follows:
	• Autoregression (AR) refers to a model that uses the dependent relationship between 
      an observation and some number of lagged observations

	• Integrated (I) The use of differencing of raw observations (e.g. subtracting an 
      observation from an observation at the previous time step) in order to make the 
      time series stationary.
	
	• Moving average (MA) incorporates the dependency between an observation and a 
      residual error from a moving average model applied to lagged observations.

Each component functions as a parameter with a standard notation. 
For ARIMA models, a standard notation would be ARIMA with p, d, and q, where integer values 
substitute for the parameters to indicate the type of ARIMA model used. 
The parameters can be defined as:
	• p: the number of lag observations in the model; also known as the lag order.
	• d: the number of times that the raw observations are differenced; 
         also known as the degree of differencing.
    • q: the size of the moving average window; also known as the order of the 
      moving average.

"""
smodel.summary()
"""
SARIMAX = X addition to the method name means that the implementation also supports 
          exogenous variables.
          These are parallel time series variates that are not modeled directly via
          AR, I or MA processes, but are made available as a weighted input to the model.
"""

"""
Predict 
"""
n_periods = 24
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
"""
prepare a dataframe with a column that contains the TimeStamps
"""
index_of_fc = pd.date_range(df.index[-1], periods = n_periods, freq='MS')

"""
Store the predicted values in a Series dataframe
"""
fitted_series = pd.Series(fitted, index=index_of_fc)
"""
Prepare dataframe with lower and upper precited confidence intervals
"""
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df)
plt.plot(fitted_series, color='green')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("SARIMA - Forecast of ATM Transactions Amount - Monthly - 2 years in advance")
plt.show()

"""
SAP HANA Implementation using PAL AutoARIMA
"""

"""
Import required SAP Libraries 
"""
from data_load_utils import Settings
from hana_ml import dataframe
from hana_ml.algorithms.pal.tsa.auto_arima import AutoARIMA
import logging
"""
Set up and start logging
"""
logger = logging.getLogger()
handler = logging.FileHandler('HANAML_SQLtrace_1301.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.info("Start logging")

"""
BEFORE GOING FURTHER, MAKE SURE THE DATA IS PREPARED IN THE DATABASE

"ZATM"."ZATM_TRANS_C" - All transactional data uploaded from CSV

PROCEDURE "ZATM"."Z_ML_TIMESERIES_FORECAST::Z_CREATE_MOVE_DEBIT_TRANS_CONS_BY_DAY" ( ) 
 Take the data from ZATM_TRANS_C and move it to "ZATM"."ZATM_TRANS_CONS_ALL_BY_DAY"
 where amounts are summed by day
 
PROCEDURE "ZATM"."Z_ML_TIMESERIES_FORECAST::Z_CREATE_MOVE_DEBIT_TRANS_CONS_BY_MONTH" ( )
 Take the data from "ZATM"."ZATM_TRANS_CONS_ALL_BY_DAY" and move it to "ZATM"."ZATM_TRANS_CONS_ALL_BY_MONTH"
 where amounts are summed by month
 
PROCEDURE "ZATM"."Z_ML_TIMESERIES_FORECAST::Z_CREATE_TBL_WITH_DAY_COUNT" ( ) 
 Take the data from "ZATM"."ZATM_TRANS_CONS_ALL_BY_MONTH" and move it to "ZATM"."ZATM_TRANS_CONS_ALL_BY_MONTH_ID"
 we are using this procedure to replace the DATE column with ID column. Sap model does not support date column.
 It uses an ID column with values like 1,2,3,4,5 (in our case 1=January 1993, 2=February 1993,...13 = January 1994)

"""

"""
Connect to the database
"""
conn_object = Settings.load_config("confige2edata.ini")
conn = dataframe.ConnectionContext(conn_object["url"], conn_object["port"], conn_object["user"], conn_object["pwd"])
conn_object={}

"""
Prepare Hana data by calling the requierd procedures
"""
"""
Instantiate the DataFrame to point to required DB Table
"""
df_v = conn.table("ZATM_TRANS_CONS_ALL_BY_MONTH_ID", schema="ZATM")
df_v.describe().select_statement
"""
Check a couple of values
"""
df_v.head(6).collect()



"""
Instantiate SAP PAL Model

initial_p - Order p of user-defined initial model. Valid only when search_strategy is 1.
 p is the number of lagged observations.
 We use 1 because we want one previous period in time to be used for actual period.
 Eg: We can see that the summer values have the same trend in all the years.
 
initial_q - Order q of user-defined initial model. Valid only when search_strategy is 1.
 q is the dependency between an observation and a residual error from a moving average model
   applied to lagged observations
 Best is to use a similar value as p
 
max_p - The maximum value of AR order p.
max_q - The maximum value of MA order q
seasonal_period- Value of the seasonal period
initial_seasonal_p - Order seasonal_p of user-defined initial model.
seasonal_d - Order of seasonal-differencing
search_strategy - The search strategy for optimal ARMA model.
 We can choose between exhaustive and stepwise. We choosed stepwise since exhaustive method is a
 highly computational load
allow_linear - Controls whether to check linear model ARMA(0,0)(0,0)m
thread_ratio - Controls the proportion of available threads to use. The ratio of available threads
    0: single thread.
    0~1: percentage.
    Others: heuristically determined
"""
sap_autoarima = AutoARIMA(conn, initial_p=1, initial_q=1, max_p=3, max_q=3, 
                      seasonal_period=12, initial_seasonal_p=0,  seasonal_d=1,
                      search_strategy=1, allow_linear=1, thread_ratio=1.0, 
                      output_fitted=True)
"""
Train SAP PAL Model
"""
sap_autoarima.fit(df_v, endog='AMOUNT')

"""
Collect statistics
"""
sap_autoarima.model_.collect().head(14)
#sap_autoarima.fitted_.collect().set_index('ID').head(17)
"""
Predict future values
"""
result= sap_autoarima.predict(forecast_method='innovations_algorithm', forecast_length=24)
"""
Retrieve predicted values
"""
saparima_res = result.collect()

"""
STORE PREDICTED VALUES IN HANA DB
"""

sap_store_result = result.save(where = ('ZATM', 'ZARIMA_PRED_VALUES_1301'), 
                                              table_type = 'COLUMN', 
                                              force = True)
####
"""
#-----plot sap forecasted values
#1-convert forecasted values to array
"""
fitted_sap= saparima_res["FORECAST"].values
"""
#prepare date-range for predicted interval (1998 and 1999)
"""
index_of_fc_sap = pd.date_range(start ='1-1-1998', end ='12-01-1999', freq ='MS') 
"""
#prepare date-range for period used for prediction- because our SAP dataframe has ID's 
#and we need dateTime values for plotting
"""
df_years_for_pred = pd.date_range(start ='1-1-1993', end ='12-01-1997', freq ='MS')
""" 
#recreate the SAP data used for training but with dates instead of ID
"""
train_data=df_v.collect()
df_sap = pd.DataFrame(train_data["AMOUNT"].values,index=df_years_for_pred)
"""
#store the forecasted values as pandas series object
"""
fitted_series_sap = pd.Series(fitted_sap, index=index_of_fc_sap)
"""
#Set up lower and upper confidence intervals
"""
lower_series_sap = pd.Series(confint[:, 0], index=index_of_fc_sap)
upper_series_sap = pd.Series(confint[:, 1], index=index_of_fc_sap)
"""
# Plot
"""
plt.plot(df_sap)
plt.plot(fitted_series_sap, color='red')
plt.fill_between(lower_series_sap.index, 
                 lower_series_sap, 
                 upper_series_sap, 
                 color='k', alpha=.15)

plt.title("SARIMA - SAP Forecast of ATM Transactions Amount - Monthly - 2 years in advance")
plt.show()

"""
Format fitted
%19.2f
"""
