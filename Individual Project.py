# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:52:45 2022

@author: James Clark
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import skew
from scipy.stats import kurtosis

# Step 1: Determine the Trading Direction and Position Size

bitcoin = pd.read_csv('BTC_USD_2014-11-03_2021-12-31-CoinDesk-2.csv')
ff3factor = pd.read_csv('F-F_Research_Data_Factors_daily.csv')
monthly_factor = pd.read_csv('F-F_Research_Data_Factors (2).csv')

# Set a datetime index for bitcoin
bitcoin['Datetime'] = pd.to_datetime(bitcoin['Date'])
bitcoin = bitcoin.set_index(['Datetime'])
del bitcoin['Date']
del bitcoin['Currency']
del bitcoin['24h Open (USD)']
del bitcoin['24h High (USD)']
del bitcoin['24h Low (USD)']

bitcoin.columns = ['Price']

# Convert the date format and set a datetime index for daily factors
ff3factor['Datetime'] = pd.to_datetime(ff3factor['Date'], format = '%Y%m%d')
ff3factor = ff3factor.set_index(['Datetime'])
del ff3factor['Date']

ff3factor = ff3factor.div(100)

# Set a datetime index for the monthly factors
monthly_factor['Datetime'] = pd.to_datetime(monthly_factor['Date'])
monthly_factor = monthly_factor.set_index(['Datetime'])
del monthly_factor['Date']

monthly_factor = monthly_factor.div(100)

# Bitcoin daily returns
bitcoin['Daily Rets'] = bitcoin['Price'].pct_change()

# Add risk free rate to bitcoin dataframe
bitcoin = bitcoin.merge(ff3factor['RF'].loc['2014-11-03':], how = 'left', 
                                  left_index = True, right_index = True)

# Fill na values with the previous non-na value
bitcoin.fillna(method = 'ffill', inplace = True)

# Calculate bitcoin excess returns
bitcoin['BTC Excess'] = bitcoin['Daily Rets'] - bitcoin['RF']

# Compute monthly cumulative returns
btc_cum_rets = bitcoin['BTC Excess'].resample('M').agg(
    lambda r: (r + 1).prod()- 1)

# Compute rolling 12-month cumulative returns
rolling_12 =  pd.DataFrame(btc_cum_rets.rolling(window = 12).apply(
    lambda r: (r + 1).prod()- 1))
rolling_12.columns = ['Roll 12-Month Cumret']

# Compute the ex-ante volatility measures based on daily excess returns
delta = 0.983606557

ex_ante_vol = []
for i in range(len(bitcoin)):
    exret = np.array(bitcoin['BTC Excess'].iloc[0:i+1])
    counter1 = np.array(range(0,i + 1))[::-1]
    counter2 = (1 - delta) * delta**counter1
    exret_mn = np.dot(counter2[1:], exret[1:])
    sqdev = (exret - exret_mn)**2
    exret_sqdev = np.dot(counter2[1:], sqdev[1:])
    fin_stdev = (365 * exret_sqdev)**0.5
    ex_ante_vol.append(fin_stdev)

# Dataframe containing the daily ex-ante volatilities    
daily_vol = pd.DataFrame(ex_ante_vol, index = bitcoin.index)
daily_vol.columns = ['Daily Vol']

# Dataframe containing ex-ante volatilities for last day of each month
month_end_vol = daily_vol.groupby(pd.Grouper(freq = 'M')).tail(1)
month_end_vol.columns = ['Month-End Vol']

# Combining monthly_vol with rolling_12
monthly_df = rolling_12.merge(month_end_vol,how = 'left', 
                                  left_index = True, right_index = True)

# Determine the trigger and direction of the trade
monthly_df['Position'] = np.where(monthly_df['Roll 12-Month Cumret'] > 0, 1, -1)

# Determine the size of the position so that annualized ex-ante volatility = 40%
vol_target = 0.4

for i in range(len(monthly_df)):
    monthly_df['Pos Size'] = vol_target/(monthly_df['Month-End Vol'])

# Drop dates before the start of the 12-month cumulative returns
monthly_df = monthly_df.drop(monthly_df.index[range(11)])

# Step 2: Construct Monthly Return Series & Perform Risk-Return Analysis

# Calculate the time-series monthly return based on direction & size
monthly_df['TSRet'] = monthly_df['Position'].shift(
    1) * monthly_df['Pos Size'].shift(1) * (btc_cum_rets.loc['2015-10-31':])

# Create a dataframe for annualized mean, stdev, and Sharpe ratio
stats = pd.Series(index = ['Annual Mean', 'Annual Stdev', 'Annual Sharpe'],
                  dtype = 'float64')

# Calculate annualized mean return
stats['Annual Mean'] = monthly_df['TSRet'].loc['2016-01-31':].mean() * 12

# Calculate annualized stdev of returns
stats['Annual Stdev'] = monthly_df['TSRet'].loc['2016-01-31':].std() * math.sqrt(12) 

# Calcualte Sharpe ratio
stats['Annual Sharpe'] = stats['Annual Mean'] / stats['Annual Stdev']

# Merge the monthly factors with the monthly_df
monthly_df = monthly_df.merge(monthly_factor,how = 'left', 
                                  left_index = True, right_index = True)

# Scatter plot between TSRet and RMRF
monthly_df.plot.scatter(x = 'TSRet', y = 'Mkt-RF')
plt.title('TSRet vs. Mkt-RF')
plt.show()

# Regressing TSRet on Mkt-RF
capm_regress = smf.ols('Q("TSRet") ~ Q("Mkt-RF")',
                          data = monthly_df.loc['2016-01-31':]).fit()

# Report the alpha, beta, and t-statistics from the CAPM regression
capm_stats = pd.Series(index = ['Alpha', 'Beta', 't-stat'], dtype = 'float64')
capm_stats['Alpha'] = capm_regress.params['Intercept']
capm_stats['Beta'] = capm_regress.params['Q("Mkt-RF")']
capm_stats['t-stat'] = capm_regress.tvalues['Intercept']

# Multiple regression with factors
ff_regress = smf.ols('Q("TSRet") ~ Q("Mkt-RF") + Q("SMB") + Q("HML") + Q("MOM")', 
                        data = monthly_df.loc['2016-01-31':]).fit()

# Report alpha, beta, and t-statistics for the ff regression
ff_stats = pd.Series(index = ['Alpha', 'Mkt-RF Beta', 'SMB Beta', 'HML Beta',
                              'MOM Beta','Mkt-RF t-stat', 'SMB t-stat',
                              'HML t-stat','MOM t-stat'], dtype = 'float64')
ff_stats['Alpha'] = ff_regress.params[0]
ff_stats['Mkt-RF Beta':'MOM Beta'] = ff_regress.params[1:]
ff_stats['Mkt-RF t-stat':] = ff_regress.tvalues[1:]

# Calculate the annual return from 2016-2021 series based on TSRet
annual_ret = monthly_df['TSRet'].resample('Y').sum()
annual_ret = annual_ret.drop(annual_ret.index[range(1)])
annual_ret.index = ['2016','2017','2018','2019','2020','2021']

# Plot a bar chart of annual returns
annual_ret.plot.bar()
plt.title('Annual Returns by Year')
plt.show()

# Compute the information ratio based on the CAPM regression
x = sm.add_constant(monthly_df['TSRet'].loc['2016-01-31':])
reg = sm.OLS(monthly_df['Mkt-RF'].loc['2016-01-31':], x).fit()
predict_vals = reg.predict()
resid = monthly_df['Mkt-RF'].loc['2016-01-31':] - predict_vals
stats['Info Ratio'] = reg.params['const'] / resid.std()

# Plot the empirical density function of TSRet with a histogram
monthly_df.hist(column = 'TSRet', bins = 10)
plt.title('Empirical Density Function of TSRet')
plt.xlabel('Time Series Return')
plt.ylabel('Number of Occurrences')

# Compute skew, kurtosis, and value at risk of monthly TSRet
stats['Skew'] = skew(monthly_df['TSRet'].loc['2016-01-31':])
stats['Kurtosis'] = kurtosis(monthly_df['TSRet'].loc['2016-01-31':])
stats['Value @ Risk'] = (monthly_df['TSRet'].loc['2016-01-31':]).quantile(0.05)

# Compute drawdown
dd_stats = pd.DataFrame(columns = ['CumRet', 'HWM', 'Drawdown', 'Max DD'])
dd_stats['CumRet'] = (1 + monthly_df['TSRet']).cumprod().loc['2016-01-31':]
dd_stats['HWM'] = dd_stats['CumRet'].cummax()
dd_stats['Drawdown'] = (dd_stats['CumRet'] - dd_stats['HWM']) / dd_stats['HWM']
dd_stats['Max DD'] = dd_stats['Drawdown'].min()

# Plot monthly DDs relative to HWM
dd_stats.plot(y = ['Drawdown'])
secondary_y = dd_stats['HWM'].plot(secondary_y = True)
plt.title('Drawdown Relative to HWM')
plt.show()

dd_stats.to_csv('Drawdown Statistics.csv')

# Outputs
print('')
print('RISK-RETURN STATISTICS')
print('')
print(stats)
print(capm_regress.summary())
print('')
print('CAPM REGRESSION STATISTICS')
print('')
print(capm_stats)
print(ff_regress.summary())
print('')
print('FOUR-FACTOR REGRESSION STATISTICS')
print('')
print(ff_stats)
print('')
print('ANNUAL RETURNS')
print('')
print(annual_ret)
print('')
print('DRAWDOWN STATISTICS')
print('')
print(dd_stats)   
    
