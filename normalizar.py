import pandas as pd
import numpy as np
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator
from indicadores import agregar_indicadores_avanzados
import os
import logging
import psutil
from time import time
import gc
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define indicators to match entrenar.py
INDICADORES = [
    'macd', 'rsi', 'cci', 'adx', 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi',
    'volume_em', 'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi', 'volume_nvi',
    'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'volatility_bbw', 'volatility_bbp',
    'volatility_bbhi', 'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
    'volatility_kcw', 'volatility_kcp', 'volatility_kchi', 'volatility_kcli', 'volatility_dcl',
    'volatility_dch', 'volatility_dcm', 'volatility_dcw', 'volatility_dcp', 'volatility_atr',
    'volatility_ui', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
    'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 'trend_vortex_ind_pos',
    'trend_vortex_ind_neg', 'trend_vortex_ind_diff', 'trend_trix', 'trend_mass_index',
    'trend_dpo', 'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
    'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_stc', 'trend_adx',
    'trend_adx_pos', 'trend_adx_neg', 'trend_cci', 'trend_visual_ichimoku_a',
    'trend_visual_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down', 'trend_aroon_ind',
    'trend_psar_up', 'trend_psar_down', 'trend_psar_up_indicator', 'trend_psar_down_indicator',
    'momentum_rsi', 'momentum_stoch_rsi', 'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d',
    'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
    'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal', 'momentum_pvo',
    'momentum_pvo_signal', 'momentum_kama'
]


def check_hardware():
    """Check system resources"""
    ram_available = psutil.virtual_memory().available / (1024 ** 3)
    ram_total = psutil.virtual_memory().total / (1024 ** 3)
    logger.info(f"RAM: {ram_available:.2f} GB disponible / {ram_total:.2f} GB total")
    if ram_available < 5:
        logger.warning("RAM críticamente baja (<5 GB disponible). Cierra otras aplicaciones.")
    disk_path = "data/final_combinado.csv" if os.path.exists("data/final_combinado.csv") else "."
    disk_usage = psutil.disk_usage(os.path.dirname(disk_path))
    logger.info(f"Espacio libre en disco: {disk_usage.free / (1024 ** 3):.2f} GB")


def compute_indicators(df, ticker):
    """Compute indicators for a ticker’s full data"""
    try:
        df = df.sort_values('date')
        if len(df) < 26:  # Minimum periods for indicators like MACD
            logger.warning(f"Datos insuficientes para indicadores en {ticker}: {len(df)} filas")
            for col in INDICADORES:
                df[col] = np.nan
            return df
        # Calculate indicators using agregar_indicadores_avanzados
        df = agregar_indicadores_avanzados(df, save_path=f"data/temp_indicators/{ticker}_temp.pkl")
        # Rename ta columns to match INDICADORES
        rename_dict = {f"ta_{col}": col for col in INDICADORES if f"ta_{col}" in df.columns}
        df = df.rename(columns=rename_dict)
        # Ensure all INDICADORES columns are present
        missing_indicators = [col for col in INDICADORES if col not in df.columns]
        if missing_indicators:
            logger.warning(f"Indicadores faltantes para {ticker}: {missing_indicators}")
            for col in missing_indicators:
                df[col] = np.nan
        gc.collect()
        return df
    except Exception as e:
        logger.error(f"Error calculando indicadores para {ticker}: {e}")
        for col in INDICADORES:
            if col not in df.columns:
                df[col] = np.nan
        return df


def process_ticker(ticker, input_file, output_dir, chunksize=50000):
    logger.info(f"Procesando ticker: {ticker}")
    output_file = os.path.join(output_dir, f"{ticker}.parquet")
    if os.path.exists(output_file):
        logger.info(f"Archivo Parquet para {ticker} ya existe, omitiendo procesamiento")
        return
    ticker_dfs = []
    total_rows = 0
    try:
        for chunk in pd.read_csv(
                input_file,
                parse_dates=['date'],
                usecols=['date', 'tic', 'open', 'high', 'low', 'close', 'volume'],
                chunksize=chunksize
        ):
            chunk = chunk[chunk['tic'] == ticker]
            if chunk.empty:
                logger.warning(f"No se encontraron datos para {ticker} en este fragmento")
                continue
            total_rows += len(chunk)
            ticker_dfs.append(chunk)
            logger.info(f"Recolectado fragmento para {ticker}, {len(chunk)} filas, total {total_rows}")
            del chunk
            gc.collect()
        if not ticker_dfs:
            logger.error(f"No se encontraron datos para {ticker} en {input_file}")
            return
        ticker_df = pd.concat(ticker_dfs)
        logger.info(f"Total filas para {ticker} antes de indicadores: {len(ticker_df)}")
        ticker_df = compute_indicators(ticker_df, ticker)
        logger.info(f"Total filas para {ticker} después de indicadores: {len(ticker_df)}")
        ticker_df.to_parquet(output_file, engine='pyarrow', index=False)
        logger.info(f"Guardado ticker {ticker} en {output_file}")
    except Exception as e:
        logger.error(f"Error procesando {ticker}: {e}")

def compute_stats_chunks(tickers, input_dir, cols_to_standardize, chunksize=50000):
    """Compute statistics for standardization from Parquet files"""
    stats = {col: {'sum': 0, 'sum_sq': 0, 'count': 0} for col in cols_to_standardize}
    for ticker in tickers:
        logger.info(f"Calculando estadísticas para ticker: {ticker}")
        input_file = os.path.join(input_dir, f"{ticker}.parquet")
        if not os.path.exists(input_file):
            logger.warning(f"Archivo Parquet para {ticker} no encontrado, omitiendo")
            continue
        try:
            df = pd.read_parquet(input_file, engine='pyarrow')
            for start in range(0, len(df), chunksize):
                chunk = df.iloc[start:start + chunksize]
                for col in cols_to_standardize:
                    if col in chunk.columns:
                        col_data = chunk[col].dropna()
                        stats[col]['sum'] += col_data.sum()
                        stats[col]['sum_sq'] += (col_data ** 2).sum()
                        stats[col]['count'] += col_data.count()
                del chunk
                gc.collect()
            del df
            gc.collect()
        except Exception as e:
            logger.error(f"Fallo al calcular estadísticas para {ticker}: {e}")
            continue
    for col in cols_to_standardize:
        if stats[col]['count'] > 0:
            mean = stats[col]['sum'] / stats[col]['count']
            variance = (stats[col]['sum_sq'] / stats[col]['count']) - (mean ** 2)
            std = np.sqrt(variance) if variance > 0 else 1.0
            stats[col] = {'mean': mean, 'std': std}
            logger.info(f"Estadísticas para {col}: media={mean:.4f}, desv. estándar={std:.4f}")
        else:
            stats[col] = {'mean': 0, 'std': 1.0}
            logger.warning(f"No se encontraron datos válidos para {col}, usando media=0, std=1")
    return stats


def standardize_ticker(ticker, input_dir, output_dir, stats, cols_to_standardize, chunksize=50000):
    """Standardize a ticker’s data and save to Parquet"""
    logger.info(f"Estandarizando ticker: {ticker}")
    input_file = os.path.join(input_dir, f"{ticker}.parquet")
    output_file = os.path.join(output_dir, f"{ticker}.parquet")
    if not os.path.exists(input_file):
        logger.warning(f"Archivo Parquet para {ticker} no encontrado, omitiendo")
        return
    try:
        df = pd.read_parquet(input_file, engine='pyarrow')
        for col in cols_to_standardize:
            if col in df.columns and stats[col].get('std', 1.0) != 0:
                df[col] = (df[col] - stats[col]['mean']) / stats[col]['std']
            else:
                logger.warning(f"No se puede estandarizar {col} para {ticker}: columna faltante o std=0")
                if col not in df.columns:
                    df[col] = np.nan
        df.to_parquet(output_file, engine='pyarrow', index=False)
        logger.info(f"Guardado datos estandarizados para {ticker} en {output_file}")
        del df
        gc.collect()
    except Exception as e:
        logger.error(f"Fallo al estandarizar {ticker}: {e}")


def clean_data(df, ticker, handle_nans='drop', min_rows=100):
    logger.info(f"Limpiando datos para {ticker}, filas iniciales: {len(df)}")
    numeric_cols = [col for col in df.columns if col not in ['date', 'tic']]
    nan_counts = df[numeric_cols].isna().sum()
    inf_counts = np.isinf(df[numeric_cols].select_dtypes(include=[np.number])).sum()
    logger.info(f"NaNs antes de limpieza para {ticker}: {nan_counts[nan_counts > 0].to_dict()}")
    logger.info(f"Infinitos antes de limpieza para {ticker}: {inf_counts[inf_counts > 0].to_dict()}")
    if handle_nans == 'drop':
        df = df.dropna()
    elif handle_nans == 'interpolate':
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        df = df.dropna()  # Drop remaining NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    logger.info(f"Filas después de limpieza para {ticker}: {len(df)}")
    if len(df) < min_rows:
        logger.warning(f"Ticker {ticker} tiene solo {len(df)} filas tras limpieza, omitiendo")
        return None
    return df

def save_pickle(dfs, output_path):
    """Save concatenated DataFrame to Pickle"""
    logger.info(f"Guardando datos combinados en {output_path}...")
    start_time = time()
    try:
        df_pandas = pd.concat(dfs, ignore_index=True).sort_values(["date", "tic"]).reset_index(drop=True)
        df_pandas.to_pickle(output_path)
        logger.info(f"Completado guardado en Pickle en {time() - start_time:.2f} segundos")
        return df_pandas
    except Exception as e:
        logger.error(f"Fallo al guardar Pickle: {e}")
        raise


if __name__ == '__main__':
    check_hardware()
    try:
        required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume'] + INDICADORES
        expected_tickers = {
            'LTCUSDT', 'MATICUSDT', 'TRXUSDT', 'XLMUSDT', 'XRPUSDT', 'ADAUSDT',
            'ATOMUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT', 'ETCUSDT', 'ETHUSDT',
            'LINKUSDT', 'SOLUSDT', 'DOTUSDT', 'UNIUSDT', 'AVAXUSDT', 'NEARUSDT', 'APTUSDT'
        }
        input_file = "data/final_combinado.csv"
        temp_indicators_dir = "data/temp_indicators"
        temp_standardized_dir = "data/temp_standardized"
        output_path = "data/normalizados.pkl"
        handle_nans = 'drop'  # Options: 'drop', 'interpolate'
        min_rows = 100  # Minimum rows per ticker
        os.makedirs(temp_indicators_dir, exist_ok=True)
        os.makedirs(temp_standardized_dir, exist_ok=True)

        # Step 1: Process tickers and compute indicators
        start_time = time()
        logger.info("Procesando tickers y calculando indicadores...")
        for ticker in expected_tickers:
            process_ticker(ticker, input_file, temp_indicators_dir, chunksize=50000)
        logger.info(f"Completado cálculo de indicadores en {time() - start_time:.2f} segundos")

        # Step 2: Compute statistics for standardization
        logger.info("Calculando estadísticas para estandarización...")
        numeric_cols = [col for col in required_cols if col not in ['date', 'tic']]
        stats = compute_stats_chunks(expected_tickers, temp_indicators_dir, numeric_cols, chunksize=50000)

        # Step 3: Standardize data
        logger.info("Estandarizando datos...")
        start_time = time()
        for ticker in expected_tickers:
            standardize_ticker(ticker, temp_indicators_dir, temp_standardized_dir, stats, numeric_cols, chunksize=50000)
        logger.info(f"Completada estandarización en {time() - start_time:.2f} segundos")

        # Step 4: Clean data and handle NaNs/infinities
        logger.info("Limpiando datos y manejando NaNs/infinitos...")
        start_time = time()
        final_dfs = []
        for ticker in expected_tickers:
            try:
                input_file = os.path.join(temp_standardized_dir, f"{ticker}.parquet")
                if not os.path.exists(input_file):
                    logger.warning(f"Archivo Parquet para {ticker} no encontrado, omitiendo")
                    continue
                df = pd.read_parquet(input_file, engine='pyarrow')
                df = clean_data(df, ticker, handle_nans=handle_nans, min_rows=min_rows)
                if df is not None:
                    final_dfs.append(df)
                    logger.info(f"Datos limpios para {ticker}: {len(df)} filas")
                del df
                gc.collect()
            except Exception as e:
                logger.error(f"Fallo al limpiar datos para {ticker}: {e}")
                continue
        logger.info(f"Completada limpieza en {time() - start_time:.2f} segundos")

        # Step 5: Combine and save final output as Pickle
        logger.info("Combinando y guardando salida final...")
        start_time = time()
        if not final_dfs:
            raise ValueError("No se generaron datos válidos para ningún ticker")
        df_pandas = save_pickle(final_dfs, output_path)

        # Verify output
        logger.info(f"Dataset guardado: {output_path}")
        logger.info(f"Shape: {df_pandas.shape}")
        logger.info(f"Columnas: {df_pandas.columns.tolist()}")
        logger.info(f"Tickers: {sorted(df_pandas['tic'].unique())}")
        logger.info(f"Rango de fechas: {df_pandas['date'].min()} a {df_pandas['date'].max()}")
        nan_counts = df_pandas[numeric_cols].isna().sum()
        inf_counts = np.isinf(df_pandas[numeric_cols].select_dtypes(include=[np.number])).sum()
        logger.info(f"NaNs por columna: {nan_counts[nan_counts > 0].to_dict()}")
        logger.info(f"Infinitos por columna: {inf_counts[inf_counts > 0].to_dict()}")

    except Exception as e:
        logger.error(f"El script falló con error: {e}")
        raise
    finally:
        # Clean up temporary files
        try:
            shutil.rmtree(temp_indicators_dir, ignore_errors=True)
            shutil.rmtree(temp_standardized_dir, ignore_errors=True)
            logger.info("Archivos temporales limpiados")
        except Exception as e:
            logger.error(f"Fallo al limpiar archivos temporales: {e}")
        gc.collect()