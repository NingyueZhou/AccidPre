from neuralprophet import save, load

def predict_accid(year, month, category='alk', type='ins'):
    
    
    
    model = load('../models/neuralprophet/model_' + category + type +'.np')
    
    future = model.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    
    # round prediction to nearest integer!!!
    
    return forecast