import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INDICADORES = [
    'macd', 'rsi', 'cci', 'adx', 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi',
    'volume_em', 'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi', 'volume_nvi',
    'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'volatility_bbw', 'volatility_bbp',
    'volatility_bbhi', 'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
    'volatility_kcw', 'volatility_kcp', 'volatility_kchi', 'volatility_kcli', 'volatility_dcl',
    'volatility_dch', 'volatility_dcm', 'volatility_dcw', 'volatility_dcp', 'volatility_atr',
    'volatility_ui', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
    'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 'trend_vortex_ind_pos',
    'trend_vortex_ind_neg', 'trend_vortex_ind_diff', 'trend_trix', 'trend_mass_index',
    'trend_dpo', 'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
    'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_stc', 'trend_adx',
    'trend_adx_pos', 'trend_adx_neg', 'trend_cci', 'trend_visual_ichimoku_a',
    'trend_visual_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down', 'trend_aroon_ind',
    'trend_psar_up', 'trend_psar_down', 'trend_psar_up_indicator', 'trend_psar_down_indicator',
    'momentum_rsi', 'momentum_stoch_rsi', 'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d',
    'momentum_tsi', 'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
    'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal', 'momentum_pvo',
    'momentum_pvo_signal', 'momentum_kama'
]

class OptimizedStockTradingEnv(StockTradingEnv):
    def __init__(self, df, **kwargs):
        self.df = df.sort_values(['date', 'tic'], inplace=False).reset_index(drop=True, inplace=False)
        super().__init__(df, **kwargs)

def load_or_process_data(normalized_path="data/normalizados.parquet"):
    try:
        logger.info(f"📂 Cargando datos normalizados desde {normalized_path}")
        df = pd.read_parquet(normalized_path, engine='pyarrow')
        expected_tickers = {'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT'}
        df = df[df['tic'].isin(expected_tickers)].copy()
        date_counts = df.groupby('date')['tic'].nunique()
        valid_dates = date_counts[date_counts == len(expected_tickers)].index
        df = df[df['date'].isin(valid_dates)].copy()
        logger.info(f"Fechas válidas: {len(valid_dates)}, Filas: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"❌ Error al cargar datos: {e}")
        raise

# Crear entorno de validación
tickers = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOTUSDT']
df = load_or_process_data()
env_kwargs = {
    "stock_dim": len(tickers),
    "hmax": 0.02,
    "initial_amount": 550.0,
    "num_stock_shares": [0.0] * len(tickers),
    "buy_cost_pct": [0.001] * len(tickers),
    "sell_cost_pct": [0.001] * len(tickers),
    "reward_scaling": 0.5,  # Aumentado para mejorar recompensa
    "state_space": 1 + len(tickers) + len(tickers) + (len(tickers) * len(INDICADORES)),
    "action_space": len(tickers),
    "tech_indicator_list": INDICADORES,
    "print_verbosity": 10,
    "max_steps": 1000
}
val_env = DummyVecEnv([lambda: OptimizedStockTradingEnv(df.copy(), **env_kwargs)])

# Cargar modelo
model_path = "models/modelo_1_hp3.zip"
logger.info(f"📂 Cargando modelo desde {model_path}")
model = PPO.load(model_path, env=val_env, device="cpu")  # Forzado a CPU

# Evaluar
logger.info("🚀 Iniciando evaluación del modelo")
mean_reward, std_reward = evaluate_policy(model, val_env, n_eval_episodes=10)
logger.info(f"🎯 Recompensa media: {mean_reward:.2f}, Desviación estándar: {std_reward:.2f}")