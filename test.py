from finrl import config, config_tickers
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.marketdata.yahoodownloader import YahooDownloader

# Descarga datos históricos de acciones de S&P500
df = YahooDownloader(start_date='2020-01-01',
                     end_date='2021-01-01',
                     ticker_list=config_tickers.DOW_30_TICKER).fetch_data()

print(df.head())

# Crea el entorno de trading
env = StockTradingEnv(df)

obs = env.reset()
print(f"Observación inicial: {obs}")
