import os
import pandas as pd
import numpy as np
import ta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# 1. Carga el dataset
df = pd.read_csv("crypto20_finrl_5min_2020_2025.csv", parse_dates=["date"])

# 2. Indicadores técnicos
df["macd"] = ta.trend.macd(df["close"])
df["rsi"] = ta.momentum.rsi(df["close"])
df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"])
df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"])
df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
df["ema"] = ta.trend.ema_indicator(df["close"], window=20)


# ✅ 2.1. Escalar precios (después de indicadores)
precio_cols = ["open", "high", "low", "close"]
df[precio_cols] = df[precio_cols] / 1000  # o usar np.log() si lo preferís


# 3. Elimina NaNs
tech_cols = ["macd", "rsi", "cci", "adx", "obv", "ema"]
print("Verificando NaNs en indicadores técnicos:")
print(df[tech_cols].isnull().sum())
df.dropna(inplace=True)

# 4. Split: 80% train, 20% test
split_index = int(len(df) * 0.8)
df_train = df.iloc[:split_index]
df_test = df.iloc[split_index:]

# 5. Configuración del entorno
ticker_list = ["BTCUSDT"]
tech_indicator_list = tech_cols
common_kwargs = {
    "stock_dim": len(ticker_list),
    "hmax": 100,
    "initial_amount": 100000,
    "num_stock_shares": [0] * len(ticker_list),
    "buy_cost_pct": [0.001] * len(ticker_list),
    "sell_cost_pct": [0.001] * len(ticker_list),
    "reward_scaling": 1e-4,
    "tech_indicator_list": tech_indicator_list,
    "make_plots": False,
}

# 6. Crear entorno de entrenamiento
train_env = DummyVecEnv([lambda: StockTradingEnv(df=df_train, **common_kwargs)])

# 7. Verifica obs antes de entrenar
obs = train_env.reset()
print("Primer obs:", obs)
print("¿Contiene NaN?", np.isnan(obs).any())
if np.isnan(obs).any():
    raise ValueError("❌ obs contiene NaN, abortando entrenamiento.")

# 8. Entrenamiento o carga del modelo
model_path = "models/ppo_stock_trading_modelaco2"
if os.path.exists(model_path + ".zip"):
    print("📦 Modelo encontrado. Cargando...")
    model = PPO.load(model_path, env=train_env)
else:
    print("🚀 Entrenando nuevo modelo...")
    model = PPO("MlpPolicy", train_env, verbose=1, ent_coef=0.01)
    model.learn(total_timesteps=1_000_000)
    model.save(model_path)
    print(f"✅ Modelo guardado en {model_path}.zip")

# 9. Evaluación
test_env = DummyVecEnv([lambda: StockTradingEnv(df=df_test, **common_kwargs)])
obs = test_env.reset()
done = False
total_reward = 0
step_count = 0
portfolio_values = []

print("\n🕵️ Debug de evaluación paso a paso:")
while not done:
    action, _states = model.predict(obs)
    print(f"Tipo de action: {type(action)}")
    obs, reward, done, info = test_env.step(action)
    total_reward += reward[0]
    step_count += 1

    # Guardamos el valor del portafolio
    if "portfolio_value" in info[0]:
        portfolio_values.append(info[0]["portfolio_value"])
    else:
        portfolio_values.append(np.nan)

    # Debug paso a paso
    print(f"Paso {step_count}")
    print(f"Acción tomada: {action}")
    print(f"Recompensa obtenida: {reward[0]:.6f}")
    print(f"Valor portafolio: {info[0].get('portfolio_value', 'No disponible')}")
    print("-" * 30)

# 10. Guardar resultados
os.makedirs("resultados", exist_ok=True)
if portfolio_values:
    df_eval = pd.DataFrame({
        "step": list(range(len(portfolio_values))),
        "portfolio_value": portfolio_values
    })
    df_eval.to_csv("resultados/evaluacion_portafolio.csv", index=False)
    print("📁 Resultados guardados en 'resultados/evaluacion_portafolio.csv'")

# 11. Métricas finales
print("\n📈 Métricas de evaluación:")
print(f"👉 Total de pasos: {step_count}")
print(f"💰 Recompensa total acumulada: {total_reward:.6f}")
if portfolio_values:
    print(f"💼 Valor final del portafolio: {portfolio_values[-1]:.2f}")
    print(f"📊 Valor inicial del portafolio: {portfolio_values[0]:.2f}")
    print(f"📈 Ganancia neta: {portfolio_values[-1] - portfolio_values[0]:.2f}")
else:
    print("⚠️ No se encontró 'portfolio_value' en info[]. Asegúrate de retornarlo en tu entorno.")

print(f"\nRango total: {df['date'].min()} a {df['date'].max()}")
print(f"Tamaño total del dataset: {len(df)} filas")
print(f"Tamaño entrenamiento: {len(df_train)} filas")
print(f"Rango entrenamiento: {df_train['date'].min()} a {df_train['date'].max()}")
print(f"Tamaño test: {len(df_test)} filas")
print(f"Rango test: {df_test['date'].min()} a {df_test['date'].max()}")