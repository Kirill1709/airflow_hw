import logging
import os
import json

import pandas as pd
import dill

def predict():
    logging.info('Начинаем предсказание')
    path = os.path.expanduser('~/airflow_hw')
    with open(f'{path}/data/models/cars_pipe.pkl', 'rb') as file:
        model = dill.load(file)
    folder_test = f'{path}/data/test'
    list_data = []
    for file in os.listdir(folder_test):
        if file.endswith('json'):
            file_path = os.path.join(folder_test, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            list_data.append(data)
    logging.info('Все json файлы перенесены в датафрейм')
    df = pd.DataFrame.from_dict(list_data)
    df['pred'] = model.predict(df)
    df.to_csv(f'{path}/data/predictions/predictions.csv')
    logging.info('Результаты записаны в файл')


if __name__ == '__main__':
    predict()
