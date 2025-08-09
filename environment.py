from stable_baselines3.common.vec_env import SubprocVecEnv
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

def calculate_state_space(stock_dim, num_indicators):
    """
    Calculate the state space size for the trading environment.

    Args:
        stock_dim (int): Number of stocks (tickers).
        num_indicators (int): Number of technical indicators per stock.

    Returns:
        int: Total state space size (cash + shares + prices + indicators).
    """
    state_space = 1 + stock_dim + stock_dim + (stock_dim * num_indicators)
    return state_space


def make_env(data_frame, name, tickers, common_kwargs, indicadores):
    """
    Create a single environment for stock trading.

    Args:
        data_frame (pd.DataFrame): DataFrame with financial data.
        name (str): Name of the environment for logging.
        tickers (list): List of ticker symbols.
        common_kwargs (dict): Common environment parameters.
        indicadores (list): List of technical indicator column names.

    Returns:
        callable: Function that initializes the StockTradingEnv.
    """
    def _init():
        try:
            env = StockTradingEnv(df=data_frame.copy(), **common_kwargs)
            state, _ = env.reset()
            test_state = env._initiate_state()
            if len(test_state) != common_kwargs["state_space"]:
                raise ValueError(f"State length {len(test_state)} does not match expected {common_kwargs['state_space']}")
            if set(env.tech_indicator_list) != set(indicadores):
                raise ValueError(f"tech_indicator_list mismatch: Expected {indicadores}, Got {env.tech_indicator_list}")
            return env
        except Exception as e:
            raise ValueError(f"Error creating environment '{name}': {str(e)}")
    return _init