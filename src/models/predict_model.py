# -*- coding: utf-8 -*-
import sys
sys.path.append('../hse_workshop_classification-main/src')
sys.path.append('../hse_workshop_classification-main/models')
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from utils import *
from data.preprocess import *
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import config as cfg


@click.command()
@click.argument('models_dir', type=click.Path(exists=True)) # models
@click.argument('inf_data_path', type=click.Path(exists=True)) # data/processed/test.pkl
@click.argument('model_name') # catboost or logistic_regression

def main(models_dir, inf_data_path, model_name):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('training model on final data')
    
    inf_data = pd.read_pickle(inf_data_path)

    inf_dict = {'id': inf_data.index}
   
    if model_name == 'logreg':
        ohe = load_ohe('models\models_logreg_ohe')
        scaler = load_scaler('models\models_logreg_scaler')

        inf_data = log_reg_preprocess(inf_data, ohe, scaler)
        
    for col in cfg.TARGET_COLS:
        model = load_model(models_dir + '/models_' + model_name + '_' + col + '.sav')
        inf_dict[col] = model.predict(inf_data)

    submission_df = pd.DataFrame(inf_dict, columns=inf_dict.keys())
    submission_df = submission_df.set_index(['id'])
    print(submission_df)

    submission_df.to_csv('submission_file_' + model_name + '.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


# python src/models/predict_model.py 'models' 'data/processed/test.pkl' 'catboost'

# dvc stage add -n evaluate catboost -d models -o submission_file_catboost.csv python src/models/predict_model.py 'models' 'data/processed/test.pkl' 'catboost'

