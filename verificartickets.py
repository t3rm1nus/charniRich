import pandas as pd

df_raw = pd.read_csv("data/final_combinado.csv")
print("Primeras 10 fechas para BTCUSDT:")
print(df_raw[df_raw['tic'] == 'BTCUSDT']['date'].head(10))
print("\nDiferencia temporal entre las primeras dos filas para BTCUSDT:")
time_diff = pd.to_datetime(df_raw[df_raw['tic'] == 'BTCUSDT']['date'].iloc[1]) - pd.to_datetime(df_raw[df_raw['tic'] == 'BTCUSDT']['date'].iloc[0])
print(f"Diferencia: {time_diff}")