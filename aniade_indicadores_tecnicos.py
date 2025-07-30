import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands

# Cargar tu dataset
df = pd.read_csv("BTCUSDT_finrl_2025.csv", parse_dates=["date"])

# Asegúrate de que esté ordenado por fecha
df = df.sort_values("date").reset_index(drop=True)

# Añadir RSI
rsi = RSIIndicator(close=df["close"], window=14)
df["rsi"] = rsi.rsi()

# Añadir MACD
macd = MACD(close=df["close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_diff"] = macd.macd_diff()

# Añadir EMA (puedes cambiar la ventana si lo deseas)
ema = EMAIndicator(close=df["close"], window=20)
df["ema_20"] = ema.ema_indicator()

# Añadir CCI
cci = CCIIndicator(high=df["high"], low=df["low"], close=df["close"], window=20)
df["cci"] = cci.cci()

# Añadir Bollinger Bands
bb = BollingerBands(close=df["close"], window=20, window_dev=2)
df["bb_bbm"] = bb.bollinger_mavg()
df["bb_bbh"] = bb.bollinger_hband()
df["bb_bbl"] = bb.bollinger_lband()

# Eliminar las primeras filas con valores NaN por las ventanas de cálculo
df = df.dropna().reset_index(drop=True)

# Guardar el nuevo dataset
df.to_csv("BTCUSDT_finrl_2025_indicadores.csv", index=False)
print("✅ Indicadores técnicos añadidos y guardados.")
