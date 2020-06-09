import boto3
import s3fs
import sagemaker
from sagemaker import get_execution_role
import time
import numpy as np
import pandas as pd
import json
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import bs4 as bs
import pickle
import requests
from datetime import timedelta
from sklearn.metrics import accuracy_score


def get_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker[:-1]
        tickers.append(ticker)

    
    tickers.sort()
    tickers.remove('BF.B')
    tickers.remove('BRK.B')
    tickers.remove('CARR')
    tickers.remove('DPZ')
    tickers.remove('DXCM')
    tickers.remove('OTIS')
    tickers.remove('WST')
    
    return tickers

def get_target_distribution(df):
    print("-1: ", len(df[df['target']==-1]))
    print(" 0: ", len(df[df['target']==0]))
    print(" 1: ", len(df[df['target']==1]))
    

def write_json_dataset(filename, data): 
    with open(filename, 'wb') as f:
        # for each of our times series, there is one JSON line
        for d in data:
            json_line = json.dumps(d) + '\n'
            json_line = json_line.encode('utf-8')
            f.write(json_line) 
            
def copy_to_s3(local_file, s3_path, s3_bucket, override=False):
    assert s3_path.startswith('s3://')
    split = s3_path.split('/')
    bucket = split[2]
    path = '/'.join(split[3:])
    buk = boto3.resource('s3').Bucket(bucket)
    
    if len(list(buk.objects.filter(Prefix=path))) > 0:
        if not override:
            print('File s3://{}/{} already exists.\nSet override to upload anyway.\n'.format(s3_bucket, s3_path))
            return
        else:
            print('Overwriting existing file')
    with open(local_file, 'rb') as data:
        print('Uploading file to {}'.format(s3_path))
        buk.put_object(Key=path, Body=data)
        
        
def encode_request(instance, num_samples, quantiles):
        configuration = {
            "num_samples": num_samples,
            "output_types": ["quantiles"],
            "quantiles": quantiles
        }
        
        http_request_data = {
            "instances": [instance],
            "configuration": configuration
        }
        
        return json.dumps(http_request_data).encode('utf-8')


def get_stock_prediction(ticker,date, df,predictor,dynamic_feat,cat):
    date_pred = pd.Timestamp(date, freq='D')
    date_start = date_pred-timedelta(days=94)
    pred_df = df.loc[(slice(str(date_start),str(date_pred)), ticker), :]
    result_df = pred_df.loc[(slice(str(date_pred),str(date_pred)), ticker), :]
    pred = {
        "start": str(date_pred),
        "target": pred_df['target'][date_start:date_pred-timedelta(days=1)].tolist(),
        "cat": cat,
        "dynamic_feat": pred_df[dynamic_feat][date_start:date_pred].values.T.tolist()
    }

    req = encode_request(instance=pred, num_samples=50, quantiles=['0.1', '0.5', '0.9'])
    res = predictor.predict(req)

    prediction_data = json.loads(res.decode('utf-8'))
    pred = round(prediction_data['predictions'][0]['quantiles']['0.5'][0])
    result_df['prediction'] = pred
    return result_df


def get_prediction_accuracy(ticker, date_index,df,predictor,dynamic_feat, cat):
    ticker = str(ticker)
    i = 0
    target = []
    prediction = []

    for date in date_index:
        target.append(get_stock_prediction(ticker, date,df,predictor,dynamic_feat, cat)['target'].values[0])
        prediction.append(int(get_stock_prediction(ticker, date,df,predictor,dynamic_feat, cat)['prediction'].values[0]))
    target = list(np.array(target).reshape(252))
    prediction = list(np.array(prediction).reshape(252))
    data = {'target': list(target), 'prediction': list(prediction)}
    prediction_df = pd.DataFrame(data=data,index=date_index, columns=['target','prediction'])
    
    return accuracy_score(target, prediction)