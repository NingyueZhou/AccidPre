import pickle
import pandas as pd
from datetime import datetime

def calc_periods(date):
    
    start = datetime(day=1,month=1,year=2000)
    end = datetime(day=1,month=12,year=2020)
    
    if date < start:
        periods = -1
    elif start <= date <= end:
        periods = 0
    elif date > end:
        periods = (date.year - end.year) * 12 + (date.month - end.month)
    
    return periods

def predict_accid(year, month, category='alk', type='ins'):
    
    date = datetime(year=year, month=month, day=1)
    periods = calc_periods(date)
    
    if periods == -1:
        predicted_accid = -1
    
    elif periods == 0:
        data = pd.read_csv('../data/data_' + category + '_' + type + '.csv')
        print(data.loc[data.ds == date, 'y'].values)
        predicted_accid = data.loc[data.ds == str(date)[:10], 'y']
        
    else:
        data = pd.read_csv('../data/data_' + category + '_' + type + '.csv')
        
        with open('../models/neuralprophet/model_' + category + '_' + type +'.pkl', 'rb') as f:
            ff = f.read().replace(b'\r\n', b'\n')
            model = pickle.load(ff)
            model.restore_trainer()
        
        future = model.make_future_dataframe(data, periods=periods, n_historic_predictions=len(data))
        future.sort_values(by='ds', inplace=True, ascending=True)
        future.reset_index(drop=True, inplace=True)
        forecast = model.predict(future)
        
        predicted_accid = forecast['yhat1'].iloc[-1]
        predicted_accid = round(predicted_accid)
        
    return predicted_accid