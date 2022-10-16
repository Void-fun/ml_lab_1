# -*- coding: utf-8 -*-
import sys
sys.path.append('../hse_workshop_classification-main/src')
sys.path.append('../hse_workshop_classification-main/data')
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
# from ..utils import save_as_pickle
# from src.utils import save_as_pickle
from utils import save_as_pickle
# import src 
from preprocess import preprocess_data, preprocess_target, extract_target
import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_data_filepath', type=click.Path())
@click.argument('output_target_filepath', type=click.Path())
def main(input_filepath, output_data_filepath, output_target_filepath=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df = pd.read_csv(input_filepath)
    df = preprocess_data(df)
    if len(output_target_filepath) > 1:
        df, target = extract_target(df)
        target = preprocess_target(target)
        save_as_pickle(target, output_target_filepath)
    save_as_pickle(df, output_data_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()


# python src/data/make_dataset.py 'data/raw/train.csv' 'data/processed/train.pkl' 'data/processed/target.pkl'
# python src/data/make_dataset.py 'data/raw/test.csv' 'data/processed/test.pkl' ' '

# python src/data/make_dataset.py 'C:\Users\User\Desktop\Ivanov_lab1\hse_workshop_classification-main\data\raw\train.pkl' 'src/data/processed/train.pkl' 'src/data/processed/target.pkl'

'''dvc stage add -n final_preprocess_data -d data/raw -o data/processed python src/data/make_dataset.py 'data/raw/train.csv' 'data/processed/train.pkl' 'data/processed/target.pkl'; python src/data/make_dataset.py 'data/raw/test.csv' 'data/processed/test.pkl' ' ' '''