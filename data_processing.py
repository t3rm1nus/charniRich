import os
import pandas as pd
import numpy as np
from indicadores import agregar_indicadores_avanzados

def load_or_process_data(normalized_path="data/normalizados.pkl", raw_path="final_combinado.csv", indicadores=None):
    """
    Load or preprocess financial data, ensuring all required columns and tickers are present.

    Args:
        normalized_path (str): Path to the normalized data pickle file.
        raw_path (str): Path to the raw CSV data file.
        indicadores (list): List of technical indicator column names.

    Returns:
        pd.DataFrame: Processed DataFrame with required columns and no NaN/infinite values.

    Raises:
        ValueError: If required columns are missing, data is invalid, or tickers are inconsistent.
    """
    required_cols = ["date", "tic", "open", "high", "low", "close", "volume"] + indicadores
    if os.path.exists(normalized_path):
        try:
            df = pd.read_pickle(normalized_path)
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols or len(df.columns) != len(required_cols):
                os.remove(normalized_path)
            else:
                expected_tickers = {'LTCUSDT', 'MATICUSDT', 'TRXUSDT', 'XLMUSDT', 'XRPUSDT', 'ADAUSDT',
                                   'ATOMUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT', 'ETCUSDT', 'ETHUSDT',
                                   'LINKUSDT', 'SOLUSDT', 'DOTUSDT', 'UNIUSDT', 'AVAXUSDT', 'NEARUSDT', 'APTUSDT'}
                numeric_cols = [col for col in required_cols if col not in ['date', 'tic']]
                nan_cols = df[numeric_cols].isna().any()
                inf_cols = df[numeric_cols].apply(lambda x: np.isinf(x).any() if pd.api.types.is_numeric_dtype(x) else False)
                if set(df['tic'].unique()) == expected_tickers and not (nan_cols.any() or inf_cols.any()):
                    return df
                os.remove(normalized_path)
        except Exception:
            os.remove(normalized_path)

    df = pd.read_csv(raw_path, parse_dates=["date"])
    required_input_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
    missing_input_cols = [col for col in required_input_cols if col not in df.columns]
    if missing_input_cols:
        raise ValueError(f"Missing required columns in input DataFrame: {missing_input_cols}")

    for ticker in df['tic'].unique():
        sub_df = df[df['tic'] == ticker]
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if not pd.api.types.is_numeric_dtype(sub_df[col]):
                raise ValueError(f"Column {col} for ticker {ticker} is not numeric")
            if sub_df[col].isna().any():
                raise ValueError(f"Column {col} for ticker {ticker} contains NaN values")
            if np.isinf(sub_df[col]).any():
                raise ValueError(f"Column {col} for ticker {ticker} contains infinite values")

    expected_tickers = {'LTCUSDT', 'MATICUSDT', 'TRXUSDT', 'XLMUSDT', 'XRPUSDT', 'ADAUSDT',
                       'ATOMUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT', 'ETCUSDT', 'ETHUSDT',
                       'LINKUSDT', 'SOLUSDT', 'DOTUSDT', 'UNIUSDT', 'AVAXUSDT', 'NEARUSDT', 'APTUSDT'}
    df = df[df['tic'].isin(expected_tickers)]
    common_dates = df.groupby('date').filter(lambda x: set(x['tic']) == expected_tickers)['date'].unique()
    df = df[df['date'].isin(common_dates)].sort_values(['date', 'tic']).reset_index(drop=True)

    df = agregar_indicadores_avanzados(df, save_path=normalized_path)
    extra_cols = [col for col in df.columns if col not in required_cols]
    if extra_cols:
        df = df[required_cols]

    numeric_cols = [col for col in required_cols if col not in ['date', 'tic']]
    for ticker in df['tic'].unique():
        mask = df['tic'] == ticker
        df.loc[mask, numeric_cols] = df.loc[mask, numeric_cols].ffill().bfill()

    nan_counts = df[numeric_cols].isna().sum()
    if nan_counts.sum() > 0:
        df = df.dropna(subset=numeric_cols).reset_index(drop=True)
        new_tickers = df['tic'].unique()
        if set(new_tickers) != expected_tickers:
            missing = expected_tickers - set(new_tickers)
            raise ValueError(f"Missing tickers after NaN drop: {missing}")

    return df


def filtrar_dias_incompletos(df, tickers):
    """
    Filter out incomplete trading days where not all tickers have data.

    Args:
        df (pd.DataFrame): Input DataFrame with financial data.
        tickers (list): List of ticker symbols to check.

    Returns:
        pd.DataFrame: Filtered DataFrame with complete trading days.
    """
    dates_to_keep = df.groupby('date').filter(
        lambda x: set(x['tic']) == set(tickers) and len(x) == len(tickers)
    )['date'].unique()
    df_filtered = df[df['tic'].isin(tickers) & df['date'].isin(dates_to_keep)].copy()
    return df_filtered.sort_values(['date', 'tic']).reset_index(drop=True)


def verificar_datos_por_ticker(df, tickers, indicadores):
    """
    Verify that each ticker has valid data and all required columns.

    Args:
        df (pd.DataFrame): Input DataFrame with financial data.
        tickers (list): List of ticker symbols to verify.
        indicadores (list): List of technical indicator column names.

    Raises:
        ValueError: If data is missing, contains NaN/infinite values, or lacks required columns.
    """
    required_cols = ["date", "tic", "open", "high", "low", "close", "volume"] + indicadores
    for tic in tickers:
        df_tic = df[df['tic'] == tic]
        if df_tic.empty:
            raise ValueError(f"Ticker {tic} has no data.")
        if len(df_tic) < 2:
            raise ValueError(f"Ticker {tic} has too few rows: {len(df_tic)}")
        missing_cols = [col for col in required_cols if col not in df_tic.columns]
        if missing_cols:
            raise ValueError(f"Ticker {tic} missing columns: {missing_cols}")
        numeric_cols = [col for col in required_cols if col not in ['date', 'tic']]
        if df_tic[numeric_cols].isna().any().any():
            raise ValueError(f"Ticker {tic} contains NaN values")
        inf_counts = df_tic[numeric_cols].apply(
            lambda x: np.isinf(x).any() if pd.api.types.is_numeric_dtype(x) else False)
        if inf_counts.any():
            raise ValueError(f"Ticker {tic} contains infinite values in columns: {inf_counts[inf_cols].index.tolist()}")