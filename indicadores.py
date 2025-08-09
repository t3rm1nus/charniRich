
import os
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator

def agregar_indicadores_avanzados(df, save_path="data/normalizados.pkl"):
    print("🔄 Agregando indicadores técnicos avanzados...")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    required_cols = ["date", "tic", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Faltan columnas requeridas: {missing_cols}")

    tickers = sorted(df["tic"].unique())
    print(f"Tickers procesados: {tickers}")
    df_list = []

    for ticker in tickers:
        print(f"📋 Procesando ticker: {ticker}")
        df_ticker = df[df["tic"] == ticker].copy()
        df_ticker = df_ticker.sort_values("date").reset_index(drop=True)

        if len(df_ticker) < 26:
            print(f"⚠️ Ticker {ticker} tiene insuficientes datos: {len(df_ticker)} filas")
            continue

        try:
            # Add all TA features
            df_ticker = add_all_ta_features(
                df_ticker,
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                fillna=True  # Fill NaNs with appropriate methods
            )

            # Explicitly calculate critical indicators to ensure correctness
            macd = MACD(close=df_ticker["close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
            df_ticker["macd"] = macd.macd()
            df_ticker["macd_signal"] = macd.macd_signal()
            df_ticker["macd_diff"] = macd.macd_diff()

            rsi = RSIIndicator(close=df_ticker["close"], window=14, fillna=True)
            df_ticker["rsi"] = rsi.rsi()

            cci = CCIIndicator(high=df_ticker["high"], low=df_ticker["low"], close=df_ticker["close"], window=20, fillna=True)
            df_ticker["cci"] = cci.cci()

            adx = ADXIndicator(high=df_ticker["high"], low=df_ticker["low"], close=df_ticker["close"], window=14, fillna=True)
            df_ticker["adx"] = adx.adx()
            df_ticker["adx_pos"] = adx.adx_pos()
            df_ticker["adx_neg"] = adx.adx_neg()

            # Remove extra columns not in INDICADORES
            expected_cols = [
                "date", "tic", "open", "high", "low", "close", "volume",
                "macd", "rsi", "cci", "adx", "volume_adi", "volume_obv", "volume_cmf", "volume_fi",
                "volume_em", "volume_sma_em", "volume_vpt", "volume_vwap", "volume_mfi", "volume_nvi",
                "volatility_bbm", "volatility_bbh", "volatility_bbl", "volatility_bbw", "volatility_bbp",
                "volatility_bbhi", "volatility_bbli", "volatility_kcc", "volatility_kch", "volatility_kcl",
                "volatility_kcw", "volatility_kcp", "volatility_kchi", "volatility_kcli", "volatility_dcl",
                "volatility_dch", "volatility_dcm", "volatility_dcw", "volatility_dcp", "volatility_atr",
                "volatility_ui", "trend_macd", "trend_macd_signal", "trend_macd_diff", "trend_sma_fast",
                "trend_sma_slow", "trend_ema_fast", "trend_ema_slow", "trend_vortex_ind_pos",
                "trend_vortex_ind_neg", "trend_vortex_ind_diff", "trend_trix", "trend_mass_index",
                "trend_dpo", "trend_kst", "trend_kst_sig", "trend_kst_diff", "trend_ichimoku_conv",
                "trend_ichimoku_base", "trend_ichimoku_a", "trend_ichimoku_b", "trend_stc", "trend_adx",
                "trend_adx_pos", "trend_adx_neg", "trend_cci", "trend_visual_ichimoku_a",
                "trend_visual_ichimoku_b", "trend_aroon_up", "trend_aroon_down", "trend_aroon_ind",
                "trend_psar_up", "trend_psar_down", "trend_psar_up_indicator", "trend_psar_down_indicator",
                "momentum_rsi", "momentum_stoch_rsi", "momentum_stoch_rsi_k", "momentum_stoch_rsi_d",
                "momentum_tsi", "momentum_uo", "momentum_stoch", "momentum_stoch_signal", "momentum_wr",
                "momentum_ao", "momentum_roc", "momentum_ppo", "momentum_ppo_signal", "momentum_pvo",
                "momentum_pvo_signal", "momentum_kama"
            ]
            extra_cols = [col for col in df_ticker.columns if col not in expected_cols]
            if extra_cols:
                print(f"⚠️ Eliminando columnas extra para {ticker}: {extra_cols}")
                df_ticker = df_ticker[expected_cols]

            # Verify data integrity
            numeric_cols = [col for col in df_ticker.columns if col not in ["date", "tic"]]
            nan_counts = df_ticker[numeric_cols].isna().sum()
            inf_counts = df_ticker[numeric_cols].apply(lambda x: np.isinf(x).sum() if pd.api.types.is_numeric_dtype(x) else 0)
            if nan_counts.sum() > 0:
                print(f"⚠️ NaN detectado en ticker {ticker} para columnas: {nan_counts[nan_counts > 0].index.tolist()}")
            if inf_counts.sum() > 0:
                print(f"⚠️ Valores infinitos detectados en ticker {ticker} para columnas: {inf_counts[inf_counts > 0].index.tolist()}")

            print(f"📋 Columnas generadas para {ticker}: {df_ticker.columns.tolist()}")
            df_list.append(df_ticker)

        except Exception as e:
            print(f"❌ Error procesando ticker {ticker}: {str(e)}")
            continue

    if not df_list:
        raise ValueError("❌ No se generaron datos válidos para ningún ticker")

    df_result = pd.concat(df_list, ignore_index=True).sort_values(["date", "tic"]).reset_index(drop=True)

    # Final NaN and infinite value check
    numeric_cols = [col for col in df_result.columns if col not in ["date", "tic"]]
    nan_counts = df_result[numeric_cols].isna().sum()
    inf_counts = df_result[numeric_cols].apply(lambda x: np.isinf(x).sum() if pd.api.types.is_numeric_dtype(x) else 0)
    if nan_counts.sum() > 0:
        print(f"⚠️ Encontrados valores NaN, eliminando filas...")
        df_result = df_result.dropna(subset=numeric_cols).reset_index(drop=True)
    if inf_counts.sum() > 0:
        print(f"⚠️ Encontrados valores infinitos, eliminando filas...")
        df_result = df_result[np.isfinite(df_result[numeric_cols]).all(axis=1)].reset_index(drop=True)

    # Verify all expected tickers and dates
    date_counts = df_result.groupby("tic")["date"].nunique()
    print(f"Unique dates per ticker after processing: {date_counts.to_dict()}")
    expected_tickers = set(tickers)
    actual_tickers = set(df_result["tic"].unique())
    if actual_tickers != expected_tickers:
        missing = expected_tickers - actual_tickers
        raise ValueError(f"❌ Missing tickers after processing: {missing}")

    # Save processed data
    df_result.to_pickle(save_path)
    print(f"💾 Indicadores guardados en {save_path}, shape: {df_result.shape}, columns: {df_result.columns.tolist()}")
    return df_result