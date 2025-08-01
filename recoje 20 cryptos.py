import time
import datetime
import pandas as pd
import numpy as np
from binance.client import Client
from ta import add_all_ta_features

# Cliente Binance (sin API key)
client = Client(api_key="", api_secret="")

def date_to_milliseconds(date_str):
    dt = datetime.datetime.strptime(date_str, "%d %b, %Y")
    return int(dt.timestamp() * 1000)

def get_binance_klines(symbol, interval, start_str, end_str=None):
    start_ts = date_to_milliseconds(start_str)
    end_ts = date_to_milliseconds(end_str) if end_str else None

    output_data = []
    limit = 1000
    while True:
        temp_data = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            startTime=start_ts,
            endTime=end_ts,
        )
        if not temp_data:
            break
        output_data.extend(temp_data)
        start_ts = temp_data[-1][0] + 1
        if len(temp_data) < limit:
            break
        time.sleep(0.25)  # Pausa para evitar rate limits
    return output_data

def klines_to_df(klines, symbol):
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit='ms')
    df["tic"] = symbol
    df = df[["date", "tic", "open", "high", "low", "close", "volume"]]
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def add_ta_indicators(df):
    df = df.copy()
    df = df.sort_values("date")
    df = add_all_ta_features(
        df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fillna=True
    )
    return df

if __name__ == "__main__":
    # Rango de fechas
    end_date_dt = datetime.datetime.now()
    start_date_dt = end_date_dt - datetime.timedelta(days=5*365)
    start_date = start_date_dt.strftime("%d %b, %Y")
    end_date = end_date_dt.strftime("%d %b, %Y")

    # 🔝 20 criptomonedas populares (puedes cambiar el orden o símbolos)
    symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT", "DOTUSDT",
        "TRXUSDT", "LINKUSDT", "LTCUSDT", "UNIUSDT", "XLMUSDT",
        "ATOMUSDT", "ETCUSDT", "NEARUSDT", "APTUSDT", "FILUSDT"
    ]

    interval = Client.KLINE_INTERVAL_5MINUTE
    all_dfs = []

    for symbol in symbols:
        print(f"\n📥 Descargando {symbol} de {start_date} a {end_date} (5m)...")
        klines = get_binance_klines(symbol, interval, start_date, end_date)
        df = klines_to_df(klines, symbol)
        print(f"📊 Añadiendo indicadores técnicos para {symbol}...")
        df = add_ta_indicators(df)
        all_dfs.append(df)

    # Combinar todos los activos
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.sort_values(by=["date", "tic"])
    final_df.reset_index(drop=True, inplace=True)

    filename = f"crypto20_finrl_5min_{start_date_dt.year}_{end_date_dt.year}.csv"
    final_df.to_csv(filename, index=False)
    print(f"\n✅ Datos para 20 criptos guardados en: {filename}")
