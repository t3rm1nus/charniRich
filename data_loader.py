import os
import pandas as pd
from indicadores import agregar_indicadores_avanzados
from escalado import escala_precios_por_ticker

RUTA_DF_PREPROCESADO = "df_preprocesado.csv"


def cargar_dataframe(archivo_original="final_combinado.csv"):
    if os.path.exists(RUTA_DF_PREPROCESADO):
        print("📂 Cargando dataframe preprocesado...")
        df = pd.read_csv(RUTA_DF_PREPROCESADO, parse_dates=["date"])
    else:
        print("📅 Cargando datos originales...")
        df = pd.read_csv(archivo_original, parse_dates=["date"])

        df = agregar_indicadores_avanzados(df)
        df = escala_precios_por_ticker(df)

        print("💾 Guardando dataframe preprocesado para próximas ejecuciones...")
        df.to_csv(RUTA_DF_PREPROCESADO, index=False)
    return df
