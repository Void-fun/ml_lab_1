import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import config as cfg
from config import TARGET_COLS


def drop_unnecesary_id(df: pd.DataFrame) -> pd.DataFrame:
    if 'ID_y' in df.columns:
        df = df.drop('ID_y', axis=1)
    return df


def drop_nan_col(df: pd.DataFrame) -> pd.DataFrame:
    if 'Частота пасс кур' in df.columns:
        df = df.drop('Частота пасс кур', axis=1)
    return df


def add_ord_edu(df: pd.DataFrame) -> pd.DataFrame:
    df[f'{cfg.EDU_COL}_ord'] = df[cfg.EDU_COL].str.slice(0, 1).astype(np.int8).values
    return df


def fill_sex(df: pd.DataFrame) -> pd.DataFrame:
    most_freq = df[cfg.SEX_COL].value_counts().index[0]
    df[cfg.SEX_COL] = df[cfg.SEX_COL].fillna(most_freq)
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('category')

    ohe_int_cols = df[cfg.OHE_COLS].select_dtypes('number').columns
    df[ohe_int_cols] = df[ohe_int_cols].astype(np.int8)

    df[cfg.REAL_COLS] = df[cfg.REAL_COLS].astype(np.float32)
    return df


def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    df = df.set_index(idx_col)
    return df


def replace_nans(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        nan_cols = df.columns[df.isna().any()].tolist()
        if col in nan_cols:
            if col not in cfg.REAL_COLS:
                most_frequent_value = df[col].value_counts().index[0]
                # df[col] = df[col].cat.add_categories(most_frequent_value)
                df[col] = df.fillna(most_frequent_value)
            else:
                mean_value = int(df[col].mean())
                # df[col] = df[col].cat.add_categories(mean_value)
                df[col].fillna(mean_value, inplace=True)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, cfg.ID_COL)
    df = drop_unnecesary_id(df)
    df = drop_nan_col(df)
    # df = add_ord_edu(df)
    df = replace_nans(df)
    df = fill_sex(df)
    df = cast_types(df)
    return df


def preprocess_target(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.TARGET_COLS] = df[cfg.TARGET_COLS].astype(np.int8)
    return df


def extract_target(df: pd.DataFrame):
    df, target = df.drop(cfg.TARGET_COLS, axis=1), df[TARGET_COLS]
    return df, target


def process_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df['Время засыпания'] = pd.to_datetime(df['Время засыпания']).dt.hour
    df['Время пробуждения'] = pd.to_datetime(df['Время пробуждения']).dt.hour

    ind = df[df['Время засыпания']==0].index
    df.loc[ind,'Время засыпания'] = 24
    ind = df[df['Время засыпания']>=18].index
    df['AM_before'] = 0
    df.loc[ind,'AM_before'] = 1
    df['sleep_duration'] = 0
    df.loc[ind, 'sleep_duration'] = 24-df['Время засыпания']+df['Время пробуждения']
    ind = df[df['Время засыпания']<18].index
    df.loc[ind, 'sleep_duration'] = df['Время пробуждения']-df['Время засыпания']

    return df


def one_hot_encode(df: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame: 
    ohe_df = pd.DataFrame(ohe.transform(df[cfg.CAT_COLS]), index=df.index)
    ohe_df = ohe_df.astype(np.int8)
    return df.merge(ohe_df, left_index=True, right_index=True)


def normalize_real_cols(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    x_scaled = scaler.transform(df[cfg.REAL_COLS])
    scaled_cols = pd.DataFrame(x_scaled, index=df.index, columns=['Возраст_курения_scaled', 'Сигарет_в_день_scaled', 'Возраст_алког_scaled'])
    
    return df.merge(scaled_cols, left_index=True, right_index=True)


def log_reg_preprocess(df: pd.DataFrame, ohe: OneHotEncoder, scaler: MinMaxScaler) -> pd.DataFrame:
    df = process_datetime(df)
    df = one_hot_encode(df, ohe)
    df = normalize_real_cols(df, scaler)

    all_cols = df.columns
    log_reg_cols = list(set(all_cols) - set(['Время засыпания', 'Время пробуждения']) - set(cfg.REAL_COLS) - set(cfg.CAT_COLS))

    return df[log_reg_cols]
