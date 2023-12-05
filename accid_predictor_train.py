import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from neuralprophet import NeuralProphet, save, load
import itertools
import logging
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')
os.remove('./train_log/neuralprophet/train.log')

df = pd.read_csv('./data/monatszahlen2307_verkehrsunfaelle_10_07_23_nosum.csv')
df.dropna(axis=0, subset=['WERT'], inplace=True)
df.reset_index(drop=True, inplace=True)
df['MONAT'] = pd.to_datetime(df['MONAT'], format='%Y%m', errors='coerce')

data = df.copy()

data_grouped = data.groupby(['MONATSZAHL', 'AUSPRAEGUNG'])

df_names = ['data_alk_ins', 'data_alk_vug', 'data_flu_ins', 'data_flu_vug', 'data_ver_ins', 'data_ver_mps', 'data_ver_vug']
df_list = []
count = 0

for i in range(len(data.MONATSZAHL.unique())):
    data_tmp = data[data['MONATSZAHL'] == data.MONATSZAHL.unique()[i]]
    for j in range(len(data_tmp.AUSPRAEGUNG.unique())):
        data_tmp_2 = data_grouped.get_group((data.MONATSZAHL.unique()[i], data_tmp.AUSPRAEGUNG.unique()[j]))
        df_name = df_names[count]
        df_name = data_tmp_2[['MONAT', 'WERT']]
        df_name.sort_values(by='MONAT', inplace=True, ascending=True)
        df_name.reset_index(drop=True, inplace=True)
        df_list.append(df_name)
        count += 1

sns.set(rc={'figure.figsize':(8, 4)})
sns.set_style("whitegrid")

logging.basicConfig(filename='./train_log/neuralprophet/train.log', encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

for i in range(len(df_list)):
    sns.lineplot(x='MONAT', y='WERT', data=df_list[i]).set_title(df_names[i])
    plt.savefig('./train_log/neuralprophet/' + df_names[i] +'.png')
    plt.show()
    
    data = df_list[i].copy()
    data.columns = ['ds', 'y']
    
    data_train_val = data[data['ds'] < '2021-01-01']
    data_train_val.reset_index(drop=True, inplace=True)
    data_train_val.to_csv('./data/' + df_names[i] + '.csv', index=False)
    logger.info(data_train_val.shape)
    print(data_train_val.shape)
    data_test = data[data['ds'] >= '2021-01-01']
    data_test.reset_index(drop=True, inplace=True)
    logger.info(data_test.shape)
    print(data_test.shape)
    
    model = NeuralProphet(yearly_seasonality='auto', seasonality_mode='additive')
    data_train, data_val = model.split_df(data_train_val, freq='MS', valid_p=0.15)
    logger.info("Dataset size:" + str(len(data)))
    logger.info("Train dataset size:" + str(len(data_train)))
    logger.info("Validation dataset size:"+ str(len(data_val)))
    print("Dataset size:", len(data))
    print("Train dataset size:", len(data_train))
    print("Validation dataset size:", len(data_val))

    param_grid = {  
        'learning_rate': [0.001, 0.008, 0.01, 0.1],
        'normalize': ['minmax', 'soft', 'standardize'],
        'epochs': [300],
        'batch_size': [3, 6, 12]
    }

    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    maes_train = []; rmses_train = []; maes_val = []; rmses_val = []; maes_test = []; rmses_test = []

    for params in all_params:
        m = NeuralProphet(**params, yearly_seasonality='auto', seasonality_mode='additive') 
        m = m.add_country_holidays(country_name='DE')
        m.set_plotting_backend('plotly-static')
        metrics_train = m.fit(data_train, validation_df=data_val, freq='MS', early_stopping=False, progress=False)
        maes_train.append(metrics_train['MAE'].values[-1])
        rmses_train.append(metrics_train['RMSE'].values[-1])
        maes_val.append(metrics_train['MAE_val'].values[-1])
        rmses_val.append(metrics_train['RMSE_val'].values[-1])
        metrics_test = m.test(data_test)
        maes_test.append(metrics_test['MAE_val'].values[-1])
        rmses_test.append(metrics_test['RMSE_val'].values[-1])
        logger.info(params)
        logger.info('training MAE:' + str(metrics_train['MAE'].values[-1]))
        logger.info('training RMSE:' + str(metrics_train['RMSE'].values[-1]))
        logger.info('validation MAE:' + str(metrics_train['MAE_val'].values[-1]))
        logger.info('validation RMSE:' + str(metrics_train['RMSE_val'].values[-1]))
        logger.info('test MAE:' + str(metrics_test['MAE_val'].values[-1]))
        logger.info('test RMSE:' + str(metrics_test['RMSE_val'].values[-1]))
        logger.info('---------------------------------')
        print(params)
        print('training MAE:' + str(metrics_train['MAE'].values[-1]))
        print('training RMSE:' + str(metrics_train['RMSE'].values[-1]))
        print('validation MAE:' + str(metrics_train['MAE_val'].values[-1]))
        print('validation RMSE:' + str(metrics_train['RMSE_val'].values[-1]))
        print('test MAE:' + str(metrics_test['MAE_val'].values[-1]))
        print('test RMSE:' + str(metrics_test['RMSE_val'].values[-1]))
        print('---------------------------------')
        
    tuning_results = pd.DataFrame(all_params)
    tuning_results['MAE_train'] = maes_train
    tuning_results['RMSE_train'] = rmses_train
    tuning_results['MAE_val'] = maes_val
    tuning_results['RMSE_val'] = rmses_val
    tuning_results['MAE_test'] = maes_test
    tuning_results['RMSE_test'] = rmses_test
    tuning_results.sort_values(by='RMSE_test', ascending=True, inplace=True)
    tuning_results.reset_index(drop=True, inplace=True)
    logger.info(tuning_results)
    print(tuning_results)

    best_params = all_params[np.argmin(rmses_test)]
    logger.info(best_params)
    print(best_params)
    
    model = NeuralProphet(**best_params, yearly_seasonality='auto', seasonality_mode='additive') 
    model = model.add_country_holidays(country_name='DE')
    model.set_plotting_backend('plotly-static')
    metrics_train = model.fit(data_train, validation_df=data_val, freq='MS', early_stopping=False, progress=False)
    logger.info(metrics_train)
    print(metrics_train)
    
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(metrics_train["MAE"], '-o', label="Training Loss")  
    ax.plot(metrics_train["MAE_val"], '-r', label="Validation Loss")
    ax.legend(loc='center right', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel("Epoch", fontsize=28)
    ax.set_ylabel("Loss", fontsize=28)
    plt.savefig('./train_log/neuralprophet/loss_' + df_names[i][5:] +'.png')
    plt.show()
    
    forecast = model.predict(data_test)
    model.plot(forecast)

    metrics_test = model.test(data_test)
    logger.info(metrics_test)
    print(metrics_test)
    
    forecast = model.predict(data_test)
    model.plot(forecast)

    metrics_test = model.test(data_test)
    logger.info(metrics_test)
    print(metrics_test)
    
    future = model.make_future_dataframe(data_train_val, periods=24, n_historic_predictions=len(data_train_val))
    future.sort_values(by='ds', inplace=True, ascending=True)
    future.reset_index(drop=True, inplace=True)
    logger.info(future)
    print(future)
    forecast = model.predict(future)
    logger.info(forecast)
    print(forecast)

    fig_forecast = model.plot(forecast)
    fig_components = model.plot_components(forecast)
    fig_model = model.plot_parameters()
    
    save(model, './models/neuralprophet/model_' + df_names[i][5:] +'.np')
    logger.info('model_'+ df_names[i][5:] +' saved.')