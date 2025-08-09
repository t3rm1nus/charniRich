import pandas as pd
import numpy as np

# Load training data
df = pd.read_csv("data/final_combinado.csv")
# Filter to 7 tickers (adjust based on modelo_1_hp2.zip)
tickers = ['AVAXUSDT', 'ADAUSDT', 'DOGEUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'ETHUSDT']
df_test = df[df['tic'].isin(tickers)].copy()

# Ensure 84 indicators (exclude momentum_ppo_hist, others_dr, others_dlr)
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
required_cols = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume'] + INDICADORES
df_test = df_test[required_cols]

# Filter validation period (e.g., last 20% of dates)
df_test['date'] = pd.to_datetime(df_test['date'])
dates = sorted(df_test['date'].unique())
val_start = dates[int(0.8 * len(dates))]
df_test = df_test[df_test['date'] >= val_start]

# Ensure all tickers have data for all dates
date_counts = df_test.groupby('date')['tic'].nunique()
valid_dates = date_counts[date_counts == len(tickers)].index
df_test = df_test[df_test['date'].isin(valid_dates)]

# Scale features
numeric_cols = [col for col in df_test.columns if col not in ['date', 'tic']]
df_test[numeric_cols] = (df_test[numeric_cols] - df_test[numeric_cols].mean()) / df_test[numeric_cols].std()

# Save
df_test.to_csv("data/df_test.csv", index=False)
print(f"Saved df_test: {len(df_test)} rows, {len(df_test['tic'].unique())} tickers, {len(df_test.columns)} columns")