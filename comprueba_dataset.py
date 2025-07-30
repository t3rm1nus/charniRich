import pandas as pd

# Carga tu dataset (ajusta nombre de archivo)
df = pd.read_csv("BTCUSDT_finrl_2025.csv", parse_dates=["date"])

# 1. Ver las primeras filas para inspección visual rápida
print("Primeras filas:")
print(df.head())

# 2. Revisar información general del dataframe
print("\nInformación general:")
print(df.info())

# 3. Verificar rango temporal completo
print("\nRango temporal del dataset:")
print(f"Fecha mínima: {df['date'].min()}")
print(f"Fecha máxima: {df['date'].max()}")

# 4. Comprobar si hay filas duplicadas o fechas faltantes
print("\nCantidad total de filas:")
print(len(df))

print("\n¿Hay fechas duplicadas?")
print(df['date'].duplicated().sum())

# 5. Verificar frecuencia de las fechas (intervalo esperado)
df_sorted = df.sort_values("date")
df_sorted["diff"] = df_sorted["date"].diff()

print("\nIntervalos entre filas:")
print(df_sorted["diff"].value_counts().head())

# 6. Opcional: estadísticas básicas de precios para validar datos
print("\nResumen estadístico de columnas relevantes:")
print(df[["open", "high", "low", "close", "volume"]].describe())
