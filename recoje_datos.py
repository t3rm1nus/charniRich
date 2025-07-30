import datetime
import pandas as pd
from binance.client import Client

# Inicializa el cliente sin API key para datos públicos
client = Client(api_key="", api_secret="")

def date_to_milliseconds(date_str):
    """Convierte fecha tipo '1 Jan, 2025' a timestamp en milisegundos"""
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
        start_ts = temp_data[-1][0] + 1  # timestamp del último + 1 ms

        if len(temp_data) < limit:
            break
    return output_data

def klines_to_df(klines):
    """Convierte la lista de velas a DataFrame"""
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
    # Convertir columnas numéricas a float
    for col in ["open", "high", "low", "close", "volume",
                "quote_asset_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]:
        df[col] = df[col].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)
    return df

if __name__ == "__main__":
    # Fecha actual
    end_date_dt = datetime.datetime.now()
    # Fecha de inicio hace 5 años
    start_date_dt = end_date_dt - datetime.timedelta(days=5*365)

    # Formatear fechas al formato "d M, Y" para la función
    start_date = start_date_dt.strftime("%d %b, %Y")
    end_date = end_date_dt.strftime("%d %b, %Y")

    print(f"Descargando datos de {start_date} a {end_date} para BTCUSDT, intervalo 1 hora...")
    klines = get_binance_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, start_date, end_date)
    df = klines_to_df(klines)
    filename = f"BTCUSDT_1h_{start_date_dt.year}_{end_date_dt.year}.csv"
    df.to_csv(filename, index=False)
    print(f"Datos guardados en {filename}")
