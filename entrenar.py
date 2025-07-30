import os
import pandas as pd
import numpy as np
import ta
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_env import CustomStockTradingEnv

# 1. Carga el dataset
df = pd.read_csv("BTCUSDT_finrl_2025.csv", parse_dates=["date"])

# 2. Indicadores técnicos
df["macd"] = ta.trend.macd(df["close"])
df["rsi"] = ta.momentum.rsi(df["close"])
df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"])
df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"])
df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
df["ema"] = ta.trend.ema_indicator(df["close"], window=20)

# 3. Elimina NaNs
tech_cols = ["macd", "rsi", "cci", "adx", "obv", "ema"]
print("Verificando NaNs en indicadores técnicos:")
print(df[tech_cols].isnull().sum())
df.dropna(inplace=True)

# Limpieza adicional: aseguramos que todos los valores sean escalares y no arrays/listas
for col in tech_cols:
    df = df[df[col].apply(lambda x: np.isscalar(x) and not isinstance(x, (list, np.ndarray)))]

# Verificación extra de tipos
print("\nVerificando tipos de datos en indicadores técnicos:")
print(df[tech_cols].applymap(type).head())

# 4. Split: 80% train, 20% test
split_index = int(len(df) * 0.8)
df_train = df.iloc[:split_index]
df_test = df.iloc[split_index:]

# Reordena las columnas del DataFrame para asegurar el orden correcto
column_order = [
    'date', 'tic', 'open', 'high', 'low', 'close', 'volume',
    'macd', 'rsi', 'cci', 'adx', 'obv', 'ema'
]
df_train = df_train[column_order]
df_test = df_test[column_order]

print("\nColumnas de df_train:", df_train.columns.tolist())

# Chequeo extra: imprime el primer registro de df_train y muestra si alguna columna es secuencia
print("\nChequeo de primer registro de df_train:")
first_row = df_train.iloc[0]
for col, val in first_row.items():
    if isinstance(val, (list, np.ndarray)):
        print(f"Columna '{col}' contiene una secuencia: {val}")
    else:
        print(f"Columna '{col}': {val} ({type(val)})")

# 5. Configuración del entorno

ticker_list = ["BTCUSDT"]
tech_indicator_list = tech_cols

# Debug: muestra cómo sería el estado inicial esperado por el entorno
initial_amount = np.float32(100000.0)
stock_owned = np.float32(0.0)  # Un solo valor en lugar de array
stock_price = np.float32(df_train.iloc[0]['close'])  # Un solo valor en lugar de array
techs = np.array([float(df_train.iloc[0][ind]) for ind in tech_indicator_list], dtype=np.float32)

# Construye el estado como una lista plana primero
state_debug = np.array([
    initial_amount,
    stock_owned,
    stock_price,
    *techs  # Desempaqueta los valores técnicos
], dtype=np.float32)

print(f"\nDEBUG estado inicial esperado por el entorno:")
print(f"initial_amount = {initial_amount}")
print(f"stock_owned = {stock_owned}")
print(f"stock_price = {stock_price}")
print(f"techs = {techs}")
print(f"state_debug = {state_debug}")
print(f"state_debug.shape = {state_debug.shape}")
print(f"Tipos en state_debug: {state_debug.dtype}")

print(f"\nDEBUG: stock_dim = {len(ticker_list)}")
print(f"DEBUG: state_space = {1 + 2 * len(ticker_list) + len(tech_indicator_list)}")
print(f"DEBUG: tech_indicator_list = {tech_indicator_list}")
print(f"DEBUG: df_train.shape = {df_train.shape}")

common_kwargs = {
    "stock_dim": len(ticker_list),
    "hmax": 1000,  # Aumentado para permitir operaciones más grandes
    "initial_amount": float(initial_amount),
    "num_stock_shares": [float(0)] * len(ticker_list),
    "buy_cost_pct": [0.001] * len(ticker_list),
    "sell_cost_pct": [0.001] * len(ticker_list),
    "reward_scaling": 1.0,  # Aumentado para hacer las recompensas más significativas
    "state_space": 1 + 2 * len(ticker_list) + len(tech_indicator_list),
    "action_space": len(ticker_list),
    "tech_indicator_list": tech_indicator_list,
    "make_plots": False,
}

# 6. Crear entorno de entrenamiento
train_env = DummyVecEnv([lambda: CustomStockTradingEnv(df=df_train, **common_kwargs)])

# 7. Verifica obs antes de entrenar
obs = train_env.reset()
print("Primer obs:", obs)
print("¿Contiene NaN?", np.isnan(obs).any())
if np.isnan(obs).any():
    raise ValueError("❌ obs contiene NaN, abortando entrenamiento.")

# 8. Entrenamiento o carga del modelo
model_path = "models/ppo_stock_trading_modelaco"
if os.path.exists(model_path + ".zip"):
    print("📦 Modelo encontrado. Cargando...")
    model = PPO.load(model_path, env=train_env)
else:
    print("🚀 Entrenando nuevo modelo...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=0.0001,  # Ajustado para un aprendizaje más estable
        n_steps=2048,  # Aumentado para mejor estabilidad
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Reducido para menos aleatoriedad
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 128, 64],  # Red más profunda
                vf=[256, 128, 64]
            ),
            activation_fn=torch.nn.ReLU
        )
    )
    print("Iniciando entrenamiento con 500,000 pasos...")
    model.learn(total_timesteps=500_000)  # Aumentado significativamente
    model.save(model_path)
    print(f"✅ Modelo guardado en {model_path}.zip")

# 9. Evaluación
test_env = DummyVecEnv([lambda: CustomStockTradingEnv(df=df_test, **common_kwargs)])
obs = test_env.reset()

done = False
truncated = False
total_reward = 0
step_count = 0
portfolio_values = []
trades_history = []
initial_portfolio = None
actions_taken = []
trades = {
    'compras': [],
    'ventas': [],
    'holds': [],
    'precios': []
}

print("\n🕵️ Debug de evaluación paso a paso:")
trade_counter = {'buys': 0, 'sells': 0, 'holds': 0}
successful_trades = 0
unsuccessful_trades = 0
total_profit = 0
total_loss = 0

while not done and not truncated:
    action, _states = model.predict(obs, deterministic=True)

    # Debug de la acción
    print(f"\nPaso {step_count + 1}:")
    action_type = 'Comprar' if action == 2 else 'Vender' if action == 0 else 'Hold'
    print(f"Acción decidida: {action_type}")
    actions_taken.append(action)

    # El DummyVecEnv devuelve 4 valores
    next_obs, reward, dones, info = test_env.step(action)
    done = dones[0]

    # Extraer información detallada
    current_price = info[0].get('current_price', 0)
    cash_balance = info[0].get('cash_balance', 0)
    num_shares = info[0].get('num_shares', 0)
    total_assets = info[0].get('total_assets', 0)
    trade_executed = info[0].get('trade_executed', False)
    shares_traded = info[0].get('shares_traded', 0)
    portfolio_change = info[0].get('portfolio_change', 0)

    # Registrar información detallada de la operación
    portfolio_value = cash_balance + (num_shares * current_price)

    # Mostrar información detallada
    print(f"\n📊 Estado actual:")
    print(f"Precio BTC: {current_price:,.2f}")
    print(f"Balance en efectivo: {cash_balance:,.2f}")
    print(f"BTC en cartera: {num_shares:.4f}")
    print(f"Valor total del portafolio: {portfolio_value:,.2f}")

    if trade_executed:
        print(f"\n🔄 Operación ejecutada:")
        if shares_traded > 0:
            trade_counter['buys'] += 1
            print(f"Compra de {shares_traded:.4f} BTC a {current_price:,.2f}")
            print(f"Monto total: {(shares_traded * current_price):,.2f}")
        elif shares_traded < 0:
            trade_counter['sells'] += 1
            print(f"Venta de {abs(shares_traded):.4f} BTC a {current_price:,.2f}")
            print(f"Monto total: {(abs(shares_traded) * current_price):,.2f}")

        if portfolio_change > 0:
            successful_trades += 1
            total_profit += portfolio_change
            print(f"✅ Operación rentable: +{portfolio_change:,.2f}")
        elif portfolio_change < 0:
            unsuccessful_trades += 1
            total_loss += abs(portfolio_change)
            print(f"❌ Operación perdedora: {portfolio_change:,.2f}")
    else:
        trade_counter['holds'] += 1
        if action != 1:  # Si intentó operar pero no pudo
            print(f"⚠️ No se pudo ejecutar la operación de {action_type}")

    print(f"Recompensa: {reward[0]:.6f}")

    trades['precios'].append(current_price)

    # Registrar operación en el historial
    trade_info = {
        'step': step_count,
        'type': action_type.lower(),
        'price': current_price,
        'shares': num_shares,
        'cash': cash_balance,
        'portfolio_value': portfolio_value,
        'executed': trade_executed,
        'shares_traded': shares_traded,
        'portfolio_change': portfolio_change
    }
    trades_history.append(trade_info)

    # Actualizar observación y recompensa total
    obs = next_obs
    total_reward += reward[0]
    step_count += 1

    # Capturar valor inicial del portafolio
    if step_count == 1:
        initial_portfolio = portfolio_value
        print(f"\n📈 Valor inicial del portafolio: {initial_portfolio:,.2f}")

    # Guardamos el valor del portafolio
    portfolio_values.append(portfolio_value)

    if step_count % 100 == 0:
        print(f"\n📊 Resumen paso {step_count}:")
        print(f"Valor del portafolio: {portfolio_value:,.2f}")
        print(f"Efectivo disponible: {cash_balance:,.2f}")
        print(f"BTC en cartera: {num_shares:.4f}")
        print(f"Recompensa acumulada: {total_reward:.6f}")
        print(f"Operaciones exitosas/fallidas: {successful_trades}/{unsuccessful_trades}")
        if successful_trades > 0:
            win_rate = (successful_trades / (successful_trades + unsuccessful_trades)) * 100
            print(f"Win rate actual: {win_rate:.2f}%")

# Análisis final
print("\n📊 Estadísticas finales de trading:")
print(f"Total de operaciones: {successful_trades + unsuccessful_trades}")
print(f"Compras ejecutadas: {trade_counter['buys']}")
print(f"Ventas ejecutadas: {trade_counter['sells']}")
print(f"Holds: {trade_counter['holds']}")
print(f"Operaciones exitosas: {successful_trades}")
print(f"Operaciones fallidas: {unsuccessful_trades}")
if successful_trades + unsuccessful_trades > 0:
    win_rate = (successful_trades / (successful_trades + unsuccessful_trades)) * 100
    print(f"Win rate final: {win_rate:.2f}%")
    print(f"Beneficio total: {total_profit:,.2f}")
    print(f"Pérdida total: {total_loss:,.2f}")
    if total_loss > 0:
        profit_factor = total_profit / total_loss
        print(f"Factor de beneficio: {profit_factor:.2f}")

print("\n💰 Resultados finales:")
print(f"Capital inicial: {initial_portfolio:,.2f}")
print(f"Capital final: {portfolio_values[-1]:,.2f}")
net_gain = portfolio_values[-1] - initial_portfolio
print(f"Ganancia/Pérdida neta: {net_gain:,.2f}")
if initial_portfolio > 0:
    roi = (net_gain / initial_portfolio) * 100
    print(f"ROI: {roi:.2f}%")

# Calcular métricas de trading
if len(trades['compras']) > 0:
    avg_buy_price = np.mean([p for _, p in trades['compras']])
    print(f"Precio promedio de compra: {avg_buy_price:.2f}")
    print(f"Volumen total de compras: {len(trades['compras'])}")

if len(trades['ventas']) > 0:
    avg_sell_price = np.mean([p for _, p in trades['ventas']])
    print(f"Precio promedio de venta: {avg_sell_price:.2f}")
    print(f"Volumen total de ventas: {len(trades['ventas'])}")

# Después del entrenamiento, obtener el último estado del entorno
final_state = train_env.env_method('get_state')[0]
final_portfolio = float(final_state[0])  # El primer elemento del estado es el balance en efectivo
shares_owned = float(final_state[1])     # El segundo elemento es la cantidad de acciones
current_price = float(final_state[2])    # El tercer elemento es el precio actual

# Calcular el valor total del portafolio incluyendo efectivo y acciones
total_final_portfolio = final_portfolio + (shares_owned * current_price)

# Calcular métricas de rendimiento
net_gain = total_final_portfolio - initial_portfolio
portfolio_values = train_env.env_method('get_portfolio_value_history')[0]

print("\n📊 Análisis de rendimiento:")
print(f"Capital inicial: {initial_portfolio:.2f}")
print(f"Capital final: {total_final_portfolio:.2f}")
print(f"Ganancia/Pérdida neta: {net_gain:.2f}")
if initial_portfolio > 0:
    rendimiento_porcentual = (net_gain / initial_portfolio) * 100
    print(f"Rendimiento porcentual: {rendimiento_porcentual:.2f}%")

# Métricas de riesgo/rendimiento
if len(portfolio_values) > 1:
    retornos_diarios = np.diff(portfolio_values) / portfolio_values[:-1]
    volatilidad_anual = np.std(retornos_diarios) * np.sqrt(365)
    print(f"Volatilidad anual: {volatilidad_anual:.2f}")

    if volatilidad_anual > 0:
        sharpe_ratio = (rendimiento_porcentual / 100) / volatilidad_anual
        print(f"Ratio de Sharpe: {sharpe_ratio:.2f}")

# Crear DataFrame con resultados
results_df = pd.DataFrame({
    'fecha': pd.date_range(start=df_test.date.iloc[0], periods=len(portfolio_values), freq='h'),
    'valor_portafolio': portfolio_values,
    'precio_btc': df_test.iloc[:len(portfolio_values)]['close'].values
})

# Guardar resultados
os.makedirs('resultados', exist_ok=True)
results_df.to_csv('resultados/evaluacion_portafolio.csv', index=False)

print("\n📁 Resultados guardados en 'resultados/evaluacion_portafolio.csv'")

# Imprimir información sobre los rangos de datos
print(f"\nRango total: {df.date.iloc[0]} a {df.date.iloc[-1]}")
print(f"Tamaño total del dataset: {len(df)} filas")
print(f"Tamaño entrenamiento: {len(df_train)} filas")
print(f"Rango entrenamiento: {df_train.date.iloc[0]} a {df_train.date.iloc[-1]}")
print(f"Tamaño test: {len(df_test)} filas")
print(f"Rango test: {df_test.date.iloc[0]} a {df_test.date.iloc[-1]}")
