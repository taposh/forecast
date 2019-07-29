from forecast import metrics

series1 = [1, 2, 43]

# print(x.naive_prediction_n_periods(series1,7))

m = metrics()

print(m.mean_absolute_percentage_error(1, 0.91))
