# initial :
#https://github.com/scikit-learn/scikit-learn/blob/ef5cb84a/sklearn/metrics/regression.py#L183
version = "Version: 0.0.1"
author = "Author: Taposh Dutta Roy"

#Libraries
import math
from sklearn.utils import check_array
import sklearn.metrics
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import pip
from dateutil.relativedelta import relativedelta
from statistics import median

# Utils class
class utils:
    def __init__(self):
        self.version = version
        self.author = author
    #import and install packages
    
    def import2(self, package):
        try:
            __import__(package)
        except ImportError:
            pip.main(['install', package]) 
                       
    
    ##TODO: Fix Typo .........
    def naive_prediction_n_periods(self, series,n_periods,verbose=False):
        """
        Returns a Series object with Naive Predictions
        Handles the frequency for yearly, monthly,daily and quarterly
        """
       
        result = pd.DataFrame()

         # Find the type of series object
        if (type(series).__name__ =="Series"):
            #Find frequency
            tsFreq = pd.infer_freq(series.index)

            if verbose == True: 
                    print("The forecast is needed for :",n_periods," periods" )
            
            if (tsFreq[0] =='A'):
                #use the same prediction for next n periods
                if n_periods >1 :
                    #Get the initial naive prediction i.e. move by 1
                    naive_prediction = series1.shift(1, freq=tsFreq[0])
                    #Get the naive dates..
                    #get next date to pass to date range
                    forecast_date = naive_prediction.index[-1] + relativedelta(years=1)
                    #this will start from next period and go to 7 periods
                    naivedates = pd.date_range(forecast_date,periods = n_periods, freq = tsFreq[0])
                    #print(naivedates)
                    #last value of series
                    last_value = series.values[-1]
                    forecast_naive=[last_value for ii in range(n_periods)]
                    #print("this should be 7:")
                    #print(forecast_naive)
                    newseries = pd.Series(forecast_naive,naivedates)
                    #print(newseries)
                    result = pd.concat([naive_prediction,newseries],verify_integrity=True)
                    #print("result")
                    #print(result)
                else :
                    naive_prediction = series1.shift(1, freq=tsFreq[0])
                    result = naive_prediction
                
                if verbose == True: 
                    print("The time series frequency is : yearly")
                    
            if (tsFreq[0] =='Q'):
                ##TODO: TEST THIS
                #use the same prediction for next n periods
                if n_periods >1 :
                    #Get the initial naive prediction i.e. move by 1
                    naive_prediction = series1.shift(1, freq=tsFreq[0])
                    #Get the naive dates..
                    #get next date to pass to date range
                    forecast_date = naive_prediction.index[-1] + relativedelta(months=3)
                    #this will start from next period and go to 7 periods
                    naivedates = pd.date_range(forecast_date,periods = n_periods, freq = tsFreq[0])
                    #print(naivedates)
                    #last value of series
                    last_value = series.values[-1]
                    forecast_naive=[last_value for ii in range(n_periods)]
                    #print("this should be 7:")
                    #print(forecast_naive)
                    newseries = pd.Series(forecast_naive,naivedates)
                    #print(newseries)
                    result = pd.concat([naive_prediction,newseries],verify_integrity=True)
                    #print("result")
                    #print(result)
                else :
                    naive_prediction = series1.shift(1, freq=tsFreq[0])
                    result = naive_prediction
                if verbose == True: 
                    print("The time series frequency is : quarterly")
                
            if (tsFreq[0] =='M'):
                #use the same prediction for next n periods
                if n_periods >1 :
                    #Get the initial naive prediction i.e. move by 1
                    naive_prediction = series1.shift(1, freq=tsFreq[0])
                    #Get the naive dates..
                    #get next date to pass to date range
                    forecast_date = naive_prediction.index[-1] + relativedelta(months=1)
                    #this will start from next period and go to 7 periods
                    naivedates = pd.date_range(forecast_date,periods = n_periods, freq = tsFreq[0])
                    #print(naivedates)
                    #last value of series
                    last_value = series.values[-1]
                    forecast_naive=[last_value for ii in range(n_periods)]
                    #print("this should be 7:")
                    #print(forecast_naive)
                    newseries = pd.Series(forecast_naive,naivedates)
                    #print(newseries)
                    result = pd.concat([naive_prediction,newseries],verify_integrity=True)
                    #print("result")
                    #print(result)
                else :
                    naive_prediction = series1.shift(1, freq=tsFreq[0])
                    result = naive_prediction
                if verbose == True: 
                    print("The time series frequency is : monthly")
                
            if (tsFreq[0] =='W'):
                #use the same prediction for next n periods
                if n_periods >1 :
                    #Get the initial naive prediction i.e. move by 1
                    naive_prediction = series1.shift(1, freq=tsFreq[0])
                    #Get the naive dates..
                    #get next date to pass to date range
                    forecast_date = naive_prediction.index[-1] + relativedelta(weeks=1)
                    #this will start from next period and go to 7 periods
                    naivedates = pd.date_range(forecast_date,periods = n_periods, freq = tsFreq[0])
                    #print(naivedates)
                    #last value of series
                    last_value = series.values[-1]
                    forecast_naive=[last_value for ii in range(n_periods)]
                    #print("this should be 7:")
                    #print(forecast_naive)
                    newseries = pd.Series(forecast_naive,naivedates)
                    #print(newseries)
                    result = pd.concat([naive_prediction,newseries],verify_integrity=True)
                    #print("result")
                    #print(result)
                else :
                    naive_prediction = series1.shift(1, freq=tsFreq[0])
                    result = naive_prediction
                if verbose == True: 
                    print("The time series frequency is : weekly")
                
            if (tsFreq[0] =='D'):
                #use the same prediction for next n periods
                if n_periods >1 :
                    #Get the initial naive prediction i.e. move by 1
                    naive_prediction = series1.shift(1, freq=tsFreq[0])
                    #Get the naive dates..
                    #get next date to pass to date range
                    forecast_date = naive_prediction.index[-1] + relativedelta(days=1)
                    #this will start from next period and go to 7 periods
                    naivedates = pd.date_range(forecast_date,periods = n_periods, freq = tsFreq[0])
                    #print(naivedates)
                    #last value of series
                    last_value = series.values[-1]
                    forecast_naive=[last_value for ii in range(n_periods)]
                    #print("this should be 7:")
                    #print(forecast_naive)
                    newseries = pd.Series(forecast_naive,naivedates)
                    #print(newseries)
                    result = pd.concat([naive_prediction,newseries],verify_integrity=True)
                    #print("result")
                    #print(result)
                else :
                    naive_prediction = series1.shift(1, freq=tsFreq[0])
                    result = naive_prediction
                if verbose == True: 
                    print("The time series frequency is : daily")
        
        return result
        
        
    
    def find_next_forecastdate(self, series,verbose=False):
        """Find frequency and forecast dates for a given series
            Keyword arguments:
            series -- pandas series object
        """
        # Find the type of series object
        if (type(series).__name__ =="Series"):
            #Find frequency
            if verbose == True:
                print("The series is :", series)
            tsFreq = pd.infer_freq(series.index)
            #print("The series is looking for frequency",tsFreq)
            #find last period
            last_date = series.index[-1]
            
            if (tsFreq[0] =='A'):
                forecast_date = last_date + relativedelta(years=1)
                if verbose == True: 
                    print("The time series frequency is : yearly")
            if (tsFreq[0] =='Q'):
                forecast_date = last_date + relativedelta(months=3)
                if verbose == True: 
                    print("The time series frequency is : quarterly")
                
            if (tsFreq[0] =='M'):
                forecast_date = last_date + relativedelta(months=1)
                if verbose == True: 
                    print("The time series frequency is : monthly")
                
            if (tsFreq[0] =='W'):
                forecast_date = last_date + relativedelta(weeks=1)
                if verbose == True: 
                    print("The time series frequency is : weekly")
                
            if (tsFreq[0] =='D'):
                forecast_date = last_date + relativedelta(days=1)
                if verbose == True: 
                    print("The time series frequency is : daily")
                
        return forecast_date
                
        
        
        
    def find_multiple_forecastdates(self, series,n_preds,verbose=False):         
        """Find Multiple forecast dates for a given series and preds
            Keyword arguments:
            series -- pandas series object
        """
        # Find the type of series object
        if (type(series).__name__ =="Series"):
            #Find frequency
            tsFreq = pd.infer_freq(series.index)
            #print("The series is looking for frequency",tsFreq)
            #find last period
            last_date = series.index[-1]
            forecast_date =[]
            if (tsFreq[0] =='A'):
                for ii in range(n_preds):
                    forecast_date = last_date + relativedelta(years=n_preds)
                    print(forecast_date)
                if verbose == True: 
                    print("The time series frequency is : yearly")
            if (tsFreq[0] =='Q'):
                n_preds_q =n_preds/3
                forecast_date = last_date + relativedelta(months=n_preds_q)
                if verbose == True: 
                    print("The time series frequency is : quarterly")
                
            if (tsFreq[0] =='M'):
                for ii in range(n_preds):
                    forecast_date = last_date + relativedelta(months=n_preds)
                    print(forecast_date)
                if verbose == True: 
                    print("The time series frequency is : monthly")
                
            if (tsFreq[0] =='W'):
                forecast_date = last_date + relativedelta(weeks=n_preds)
                if verbose == True: 
                    print("The time series frequency is : weekly")
                
            if (tsFreq[0] =='D'):
                forecast_date = last_date + relativedelta(days=n_preds)
                if verbose == True: 
                    print("The time series frequency is : daily")
                
        return forecast_date
            
        
    #function to add period to series
    def addperiodstoSeries(self, series,num_value,verbose=False):
        """Adding one period to a series.
            Keyword arguments:
            series -- pandas series object
            num_value -- forecasted value 
            verbose -- Boolean value to see details
        """
        #Drop NA's
        series = series.dropna(how='all')
            
        #Lets find out the next forecast date of the series
        forecast_date=utils.find_next_forecastdate(series)
        
        if verbose== True:
            print("Date to be Forecasted", forecast_date)
            
        #New Pandas Series
        newseries = pd.Series(num_value, index=[forecast_date])
        
        if verbose== True:
            print("Delta Series added", newseries)
        
        #append the new series to old series
        result_series = series1.append(newseries)
        
        return result_series  
      
        
        
    def addOneyeartoSeries(self, series,num_value,verbose=False):
        """Adding one period to a series.
            Keyword arguments:
            series -- pandas series object
            num_value -- forecasted value 
            verbose -- Boolean value to see details
        """
        series = series.dropna(how='all')
        #create a new series and then append to the series
        #forecast_dates = list(series1.index)
        last_date = series.index[-1]
        if verbose==True:
            print("Last Date", last_date)
        from dateutil.relativedelta import relativedelta
        forecast_date = last_date + relativedelta(years=1)
        if verbose== True:
            print("Date to be Forecasted", forecast_date)
        #New Pandas Series
        newseries = pd.Series(num_value, index=[forecast_date])
        if verbose== True:
            print("Delta Series", newseries)
        #append the new series to old series
        result_series = series1.append(newseries)
        return result_series  
    # pass in a series object and get the forecast for next period
    
    def average_forecast_one_period(self, series,verbose=False):
        """Average forecasting for one period
            Keyword arguments:
            myseries -- pandas series object
            n -- number of periods
            start_date -- start date for the series
        """
        if verbose == True:
            print("----------------------------")
            print("Average Forecast One Period")
            print("----------------------------")
        #Make sure we drop the na
        myseries = series.dropna(how='all')
        #Find the average value
        value = float(sum(myseries))/len(myseries)
        #add the next period and return the result series with value and date
        result_series = utils.addperiodstoSeries(myseries,value, verbose)
        return result_series
    
    
    
    # pass in a series object and get the forecast for next n periods
    def average_forecast_n_periods_old(self, series,n_periods,start_date,verbose=False):
        """Average forecasting for number of periods
            Keyword arguments:
            myseries -- pandas series object
            n_periods -- number of periods
            start_date -- start date for the series
        """
        if verbose == True:
            print("----------------------------")
            print("Average Forecast Multiple Periods")
            print("----------------------------")
        #list for new series
        mytmp=[]
        myseries = series #.dropna(how='all')
        mylist = list(myseries)
        n=n_periods    
        #loop
        for x in range(1,n+1):
            #get the value for the list
            val1 = math.fsum(mylist)/len(mylist)
            #append it to our list
            mytmp.append(val1)
            mylist.append(val1)
            #mylist.append(math.fsum(mylist)/len(mylist))
        #create an output list
        date_tmp =utils.find_multiple_forecastdates(myseries,n)
        #date_tmp = pd.date_range(start_date,periods =x, freq = 'A')
        s_tmp = pd.Series(mytmp, index=[date_tmp])
        #print(s_tmp)
        try:
            newseries = myseries.append(s_tmp,verify_integrity=True)
            return newseries
        except ValueError as err:
            print(err)
            print("No value returned")
            return None
        
    def average_forecast_n_periods(self, series,n_periods,start_date,verbose=False):
        """Average forecasting for number of periods
            Keyword arguments:
            myseries -- pandas series object
            n_periods -- number of periods
            start_date -- start date for the series
        """
        if verbose == True:
                print("----------------------------")
                print("Average Forecast Multiple Periods")
                print("----------------------------")
                print("we have an object of type: ", type(series).__name__)
        
        #Get forecast for 1 value
        result_series = utils.average_forecast_one_period(series,verbose)
        #add to series
        
        #input series
        inpseries = result_series
        
        #remaining n-1 periods
        for ii in range(0,n_periods):  
            result_series= utils.average_forecast_one_period(inpseries,verbose)
            inpseries = pd.concat([inpseries,result_series[len(result_series)-1:]])
            
        return inpseries
    
        
    # pass in a series object and get the forecast for next n periods
    def average_forecast_n_periods2(self, series,n,start_date,verbose=False):
        """Average forecasting for number of periods
            Keyword arguments:
            myseries -- pandas series object
            n -- number of periods
            start_date -- start date for the series
        """
        if verbose == True:
            print("----------------------------")
            print("Average Forecast Multiple Periods")
            print("----------------------------")
        #list for new series
        mytmp=[]
        myseries = series #.dropna(how='all')
        mylist = list(myseries)

        #loop
        for x in range(1,n+1):
            #get the value for the list
            val1 = math.fsum(mylist)/len(mylist)
            #append it to our list
            mytmp.append(val1)
            mylist.append(val1)
            #mylist.append(math.fsum(mylist)/len(mylist))
        #create an output list    
        date_tmp = pd.date_range(start_date,periods =x, freq = 'A')
        s_tmp = pd.Series(mytmp, index=[date_tmp])
        #print(s_tmp)
        try:
            newseries = myseries.append(s_tmp,verify_integrity=True)
            return newseries
        except ValueError as err:
            print(err)
            print("No value returned")
            return None
            
    # moving average using k last points
    def moving_average_forecast(self, series, window_size,verbose=False):
        """Moving Average Forecast
            Keyword arguments:
            series -- pandas series object
            window -- sliding window size
        """
        if verbose == True:
            print("----------------------------")
            print("Moving Average Forecast One Period")
            print("----------------------------")
        #First Period
        result_series = utils.average_forecast_one_period(series[-window_size:])
        return result_series
    
    ########################################################
    # Moving average using k last points                   #
    ########################################################
    # TODO: Test cases for this
    def moving_average_forecast_n_periods(self, series, window_size, n,start_date,verbose=False):
        """Moving Average Forecast
            Keyword arguments:
            series -- pandas series object
            window -- sliding window size
            n -- number of periods
        """
        if verbose == True:
            print("----------------------------")
            print("Moving Average Forecast")
            print("----------------------------")
            print("Series length :", len(series) )
            print("Series with window: ", len(series[-window_size:]))
            print(series[-window_size:])
        
        if window_size > len(series) :
            print("Window size should be less than series")
            return None
        else :
            #Lets get all series values
            X = series.values
            #window size
            window = window_size
            #Construct the historic and test sets
            history = [X[i] for i in range(window)]
            #The number of items to predict
            test = [X[i] for i in range(window, len(X))]
            predictions = list()
            # walk forward over time steps in test
            for t in range(n):
                length = len(history)
                yhat = np.mean([history[i] for i in range(length-window,length)])
                #obs = test[t]
                predictions.append(yhat)
                #history.append(obs)
                if verbose == True:
                    print('predicted=%f' % yhat)
            
            #Now we have both values of predictions as list and num. of periods needed
            tsFreq = pd.infer_freq(series.index)
        
            dates = pd.date_range(start_date,periods = n, freq = tsFreq[0])
            if verbose == True:
                print("The dates to be forecasted :",dates)
            pred_series = pd.Series(predictions,dates)
            if verbose == True:
                print("The Series forecasted :",pred_series)
            result_series = pd.concat([series,pred_series])
            
            return result_series
    
    def moving_average_forecast_n_periods_old(self, series,window_size,n_periods,start_date,verbose=True):
        """Moving Average Forecast for number of periods
            Keyword arguments:
            series -- pandas series object
            window_size -- sliding window size
            n_periods -- number of periods
            start_date -- start date for the series
        """
        if verbose == True:
                print("----------------------------")
                print("Moving Average Forecast Multiple Periods")
                print("----------------------------")
                print("we have an object of type: ", type(series).__name__)
        
        #Get forecast for 1 value
        result_series = utils.average_forecast_one_period(series[-window_size:],verbose)
        
        if verbose == True:
            print("result_series",result_series)
        #add to series
        
        #input series
        inpseries = result_series
        
        if verbose == True:
            print(inpseries[-window_size:])
        
        #remaining n-1 periods
        for ii in range(0,n_periods):  
            result_series= utils.average_forecast_one_period(inpseries[-window_size:],verbose)
            inpseries = pd.concat([inpseries,result_series[len(result_series)-1:]])
            
        return inpseries
    
    # weighted average, weights is a list of weights
    def weighted_average_forecast(self, series, weights,verbose=False):
        """Weighted Average Forecast
            Keyword arguments:
            series -- pandas series object
            weights -- number of periods
        """
        if verbose == True:
            print("----------------------------")
            print("Weighted Average Forecast")
            print("----------------------------")
        result = 0.0
        weights.reverse()
        for n in range(len(weights)):
            result += series[-n-1] * weights[n]
        s2 = utils.addperiodstoSeries(series,result,verbose)
        return s2

    # given a series and alpha, return series of smoothed points
    # https://pandas.pydata.org/pandas-docs/stable/timeseries.html
    def exponential_smoothing_forecast(self, series, alpha,verbose=False):
        """Exponential Smoothing Forecast
            Keyword arguments:
            series -- pandas series object
            alpha -- smoothing factor
        """
        if verbose == True:
                print("----------------------------")
                print("Exponential Smoothing Forecast")
                print("----------------------------")
                print("we have an object of type: ", type(series).__name__)

        if (type(series).__name__ =="Series"):
                expS1 = series.index

        result = [series[0]] # first value is same as series
    
        if verbose == True: 
            print("Actual for period 1 is same as forecast: ",result )
        
        for n in range(0, len(series)):
            result.append(alpha * series[n] + (1 - alpha) * result[n-1])
          
        
        if (type(series).__name__ =="Series"):
            #increment the index by one period
            forecast_date =utils.find_next_forecastdate(series,verbose)
            if verbose == True: 
                print("The forecast date is :",forecast_date)

            #Index Operations    
            expS1 = list(expS1)
            expS1.append(forecast_date)
            
            if verbose == True: 
                    print("The next period is :",forecast_date)
                    print("The periods are :", len(expS1) )
                    print("The values are: ", len(result))
            #TODO tests for monthly,weekly and daily
        
            s1 = pd.Series(result,index=expS1)
            return s1
        else :
            return result

        
    
    #Exponential Smoothing Forecast for multiple periods
    def exponential_smoothing_forecast_n_periods(self, series, alpha,n_periods,verbose=False):
        """Exponential Smoothing Forecast for multiple periods
            Keyword arguments:
            series -- pandas series object
            alpha -- smoothing factor
            n -- number of periods
            start_date -- forecast start date
        """
        if verbose == True:
                print("----------------------------")
                print("Exponential Smoothing Forecast Multiple Periods")
                print("----------------------------")
                print("we have an object of type: ", type(series).__name__)
        
        #Get forecast for 1 value
        result_series = utils.exponential_smoothing_forecast(series, alpha,verbose)
        #add to series
        
        #input series
        inpseries = result_series
        
        #remaining n-1 periods
        for ii in range(0,n_periods):  
            result_series= utils.exponential_smoothing_forecast(inpseries, alpha,verbose)
            inpseries = pd.concat([inpseries,result_series[len(result_series)-1:]])
            
        return inpseries
        
        
        
        
    # given a series,alpha and trend return series of smoothed points
    def holts_addative(self, series, alpha,beta,verbose=False):
        """Holts or Double Exponential Smoothing Forecast
            Keyword arguments:
            series -- pandas series object
            alpha -- level smoothing factor
            beta -- trend speed control factor
        """
        if verbose == True: 
                print("----------------------------")
                print("Holts Addative Forecast")
                print("----------------------------")
                print("we have an object of type: ", type(series).__name__)

        if (type(series).__name__ =="Series"):
                expS1 = series.index

        result = [series[0]] # first value is same as series
    
        if verbose == True: 
            print("Actual for period 1 is same as forecast: ",result )
        
        for n in range(1, len(series)+1):
            if n == 1:
                level, trend = series[0], series[1] - series[0]
            if n >= len(series): # we are forecasting
                value = result[-1]
            else:
                value = series[n] 
            last_level, level = level, alpha*value + (1-alpha)*(level+trend)
            trend = beta*(level-last_level) + (1-beta)*trend
            result.append(level+trend)    
            
        if (type(series).__name__ =="Series"):
            #increment the index by one period
            forecast_date =utils.find_next_forecastdate(series,verbose)
            if verbose == True: 
                print("The forecast date is :",forecast_date)
            #Index Operations    
            expS1 = list(expS1)
            expS1.append(forecast_date)
            if verbose == True: 
                    print("The next period is :",forecast_date)
                    print("The periods are :", len(expS1) )
                    print("The values are: ", len(result))
            #TODO tests for montly,weekly and daily
        
            s1 = pd.Series(result,index=expS1)
            return s1
        else :
            return result
        

    # given a series,alpha and trend return series of smoothed points
    def holts_multiplicative(self, series, alpha,beta,verbose=False):
        """Holts or Double Exponential Smoothing Forecast
            Keyword arguments:
            series -- pandas series object
            alpha -- level smoothing factor
            beta -- trend speed control factor
        """
        if verbose == True: 
                print("----------------------------")
                print("Holts Multiplicative Forecast")
                print("----------------------------")
                print("we have an object of type: ", type(series).__name__)

        if (type(series).__name__ =="Series"):
                expS1 = series.index

        result = [series[0]] # first value is same as series
    
        if verbose == True: 
            print("Actual for period 1 is same as forecast: ",result )
        
        for n in range(1, len(series)+1):
            if n == 1:
                level, trend = series[0], series[1] - series[0]
            if n >= len(series): # we are forecasting
                value = result[-1]
            else:
                value = series[n] 
            last_level, level = level, alpha*value + (1-alpha)*(level+trend)
            trend = beta*(level-last_level) + (1-beta)*trend
            result.append(level*trend)    
            
        if (type(series).__name__ =="Series"):
            #increment the index by one period
            forecast_date =utils.find_next_forecastdate(series,verbose)
            if verbose == True: 
                print("The forecast date is :",forecast_date)
            #Index Operations
            expS1 = list(expS1)
            expS1.append(forecast_date)
            if verbose == True: 
                    print("The next period is :",forecast_date)
                    print("The periods are :", len(expS1) )
                    print("The values are: ", len(result))
            #TODO tests for montly,weekly and daily
        
            s1 = pd.Series(result,index=expS1)
            return s1
        else :
            return result
   
    #for Holt-winters Initial Trend
    def initial_trend(self, series, slen):
        sum = 0.0
        for i in range(slen):
            sum += float(series[i+slen] - series[i]) / slen
        return sum / slen

    #for Holt-winters Initial Season
    def initial_seasonal_components(series, slen):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(series)/slen)
        # compute season averages
        for j in range(n_seasons):
            season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
        # compute initial values
        for i in range(slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals
    
    #Holt-winters 
    def holt_winters_forecasting(self, series, slen, alpha, beta, gamma, n_preds):
        result = []
        seasonals = utils.initial_seasonal_components(series, slen)
        for i in range(len(series)+n_preds):
            if i == 0: # initial values
                smooth = series[0]
                trend = utils.initial_trend(series, slen)
                result.append(series[0])
                continue
            if i >= len(series): # we are forecasting
                m = i - len(series) + 1
                result.append((smooth + m*trend) + seasonals[i%slen])
            else:
                val = series[i]
                last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
                trend = beta * (smooth-last_smooth) + (1-beta)*trend
                seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
                result.append(smooth+trend+seasonals[i%slen])
        
        return result
    
        #Holt-winters 
    def holt_winters_forecasting2(self, series, slen, alpha, beta, gamma, n_preds):
        result = []
        seasonals = utils.initial_seasonal_components(series, slen)
        for i in range(len(series)+n_preds):
            if i == 0: # initial values
                smooth = series[0]
                trend = utils.initial_trend(series, slen)
                result.append(series[0])
                continue
            if i >= len(series): # we are forecasting
                m = i - len(series) + 1
                result.append((smooth + m*trend) + seasonals[i%slen])
            else:
                val = series[i]
                last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
                trend = beta * (smooth-last_smooth) + (1-beta)*trend
                seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
                result.append(smooth+trend+seasonals[i%slen])
        #create a date-range
        forecast_date = series.index[-1] + relativedelta(months=n_preds)
        
        print(forecastdates)
        
        return result
    
class metrics:
    def __init__(self):
        self.version = "0.0.1"
        self.author = author
                
    def mean_absolute_percentage_error(self, y_true, y_pred): 
        """Mean absolute percentage error
            Keyword arguments:
            y_true -- truth value
            y_pred -- Predicted value 
            n -- number of periods
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        result= np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return result
        
    def root_mean_square_error(self, y_true, y_pred, n=0):
        """Root mean squared error
            Keyword arguments:
            y_true -- truth value
            y_pred -- Predicted value 
            n -- number of periods
        """
        if n == 0:
            #This is directly from the sklearn metric
            mse = mean_squared_error(y_true,y_pred)
            result = math.sqrt(mse)
            return result
        else:
            #This is directly from the sklearn metric
            mse = mean_squared_error(y_true,y_pred,n)
            result = math.sqrt(mse)
            return result
        
    def mean_squared_error(self, y_true,y_pred):
        """Root mean squared error (Overloaded from sklearn)
            Keyword arguments:
            y_true -- truth value
            y_pred -- Predicted value 
            n -- number of periods
        """
        mse = mean_squared_error(y_true,y_pred)
        return mse
    
    def median_relative_absolute_error(self, y_true,y_pred,naive):
        """Median RAE (Overloaded from sklearn)
            Keyword arguments:
            y_true -- truth value
            y_pred -- Predicted value 
            n -- number of periods
        """
        result = []
        for indx in range(len(y_pred)):
            ii = np.abs(np.array(y_true) - np.array(y_pred))/ np.abs(np.array(naive)-np.array(y_true))
            if ii is None:
                result.append(1)
            else:
                result.append(ii)
        return np.ma.median(result)
