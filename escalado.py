import pandas as pd
import os
import joblib
from pathlib import Path
from typing import List, Optional


def escala_precios_por_ticker(
        df: pd.DataFrame,
        price_cols: List[str] = ["open", "high", "low", "close"],
        save_path: str = "data/precios_escalados.pkl",
        force_reprocess: bool = False,
        verbose: bool = True
) -> pd.DataFrame:
    """
    Escala precios por ticker usando el precio de apertura como base y cachea los resultados.

    Args:
        df: DataFrame con columnas ['date', 'tic', 'open', 'high', 'low', 'close', ...]
        price_cols: Columnas de precios a escalar
        save_path: Ruta para guardar los datos procesados
        force_reprocess: Si True, reprocesa incluso si existe cache
        verbose: Si True, muestra mensajes de progreso

    Returns:
        DataFrame con precios escalados
    """
    # Crear directorios necesarios
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Cargar datos procesados si existen y no se fuerza reprocesamiento
    if not force_reprocess and os.path.exists(save_path):
        if verbose:
            print(f"📦 Cargando datos escalados desde caché: {save_path}")
        try:
            cached_data = joblib.load(save_path)

            # Verificar integridad de los datos cacheados
            required_cols = ['date', 'tic'] + price_cols
            if all(col in cached_data.columns for col in required_cols):
                if verbose:
                    print("✅ Datos cacheados validados correctamente")
                return cached_data
            else:
                if verbose:
                    print("⚠️ Datos cacheados incompletos, reprocesando...")
        except Exception as e:
            if verbose:
                print(f"⚠️ Error cargando caché: {str(e)}, reprocesando...")

    if verbose:
        print("🔄 Procesando escalado de precios por ticker...")

    # Verificar columnas requeridas
    required_columns = ['date', 'tic'] + price_cols
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Faltan columnas requeridas: {missing_cols}")

    # Procesamiento principal
    df_scaled = df.copy()
    tickers = df_scaled["tic"].unique()

    for ticker in tickers:
        mask = df_scaled["tic"] == ticker
        ticker_data = df_scaled.loc[mask]

        if len(ticker_data) == 0:
            continue

        base_price = ticker_data["open"].iloc[0]

        if base_price == 0:
            raise ValueError(f"⚠️ Precio base 0 encontrado para {ticker}")

        if verbose:
            print(f"   - Escalando {ticker} (precio base: {base_price:.4f})")

        # Escalar precios relativos al precio base
        df_scaled.loc[mask, price_cols] = df_scaled.loc[mask, price_cols] / base_price

    # Guardar resultados en caché
    try:
        joblib.dump(df_scaled, save_path)
        if verbose:
            print(f"💾 Datos escalados guardados en: {save_path}")
    except Exception as e:
        print(f"⚠️ No se pudo guardar caché: {str(e)}")

    if verbose:
        print("✅ Escalado de precios completado")

    return df_scaled


# Ejemplo de uso integrado con tu pipeline
if __name__ == "__main__":
    # Configuración (simulando tu entorno)
    os.makedirs("data", exist_ok=True)

    # 1. Cargar datos (simulados para el ejemplo)
    print("\n📂 Cargando datos...")
    data = {
        'date': pd.date_range('2023-01-01', periods=5).repeat(2),
        'tic': ['BTC', 'ETH'] * 5,
        'open': [100, 50, 105, 52, 103, 55, 107, 53, 110, 57],
        'high': [102, 52, 108, 54, 106, 56, 109, 55, 112, 60],
        'low': [98, 48, 103, 50, 101, 52, 105, 51, 108, 55],
        'close': [101, 51, 106, 53, 104, 54, 108, 54, 111, 58],
        'volume': [1000, 2000, 1100, 2100, 1050, 2050, 1150, 2150, 1200, 2200]
    }
    df = pd.DataFrame(data)
    print(df.head())

    # 2. Procesar datos con caché
    print("\n⚙️ Procesando datos...")
    df_procesado = escala_precios_por_ticker(
        df,
        price_cols=["open", "high", "low", "close"],
        save_path="data/precios_escalados.pkl",
        verbose=True
    )

    # 3. Mostrar resultados
    print("\n📊 Resultados:")
    print(df_procesado.head())

    # 4. Verificar que se carga desde caché en la siguiente ejecución
    print("\n🔄 Intentando cargar desde caché...")
    df_cargado = escala_precios_por_ticker(
        df,
        price_cols=["open", "high", "low", "close"],
        save_path="data/precios_escalados.pkl",
        verbose=True
    )

    print("\n✅ Pipeline completado")