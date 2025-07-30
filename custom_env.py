import numpy as np
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from gymnasium import spaces

class CustomStockTradingEnv(StockTradingEnv):
    def __init__(self, df, **kwargs):
        # Inicializar variables básicas antes de llamar a super().__init__
        self.df = df
        self.stock_dim = kwargs.get('stock_dim', 1)
        self.initial_amount = kwargs.get('initial_amount', 100000)
        self.tech_indicator_list = kwargs.get('tech_indicator_list', [])

        # Inicializar costos de transacción
        self.buy_cost_pct = kwargs.get('buy_cost_pct', [0.001])[0]
        self.sell_cost_pct = kwargs.get('sell_cost_pct', [0.001])[0]
        self.transaction_cost_pct = self.buy_cost_pct

        # Calcular state_space
        self.state_space = kwargs.get('state_space',
            1 + 2 * self.stock_dim + len(self.tech_indicator_list))

        # Definir espacio de acción discreto con 7 acciones
        # -1: venta fuerte, -0.5: venta media, -0.25: venta suave
        # 0: mantener
        # 0.25: compra suave, 0.5: compra media, 1: compra fuerte
        self.action_space = spaces.Discrete(7)
        self.action_map = {
            0: -1.0,    # Venta fuerte
            1: -0.5,    # Venta media
            2: -0.25,   # Venta suave
            3: 0.0,     # Mantener
            4: 0.25,    # Compra suave
            5: 0.5,     # Compra media
            6: 1.0      # Compra fuerte
        }

        # Definir espacio de observación normalizado
        self.observation_space = spaces.Box(
            low=-3,  # 3 desviaciones estándar hacia abajo
            high=3,  # 3 desviaciones estándar hacia arriba
            shape=(self.state_space,),
            dtype=np.float32
        )

        # Parámetros de trading ajustados
        self.min_trade_fraction = 0.3   # Mínimo 30% del balance disponible
        self.max_trade_fraction = 1.0   # Máximo 100% del balance disponible
        self.hold_penalty = 0.001       # Penalización por hold
        self.transaction_cost_pct = 0.001  # 0.1% de comisión por operación

        # Variables de seguimiento
        self.total_trades = 0
        self.successful_trades = 0
        self.portfolio_value_history = [self.initial_amount]
        self.portfolio_value = self.initial_amount
        self.transaction_history = []

        # Inicializar variables de memoria y datos
        self.state_memory = [0] * self.stock_dim
        self.asset_memory = [self.initial_amount]
        self.date_memory = [self.df.date.iloc[0]]
        self.previous_total_asset = self.initial_amount

        # Preparar datos y calcular estadísticas para normalización
        self._prepare_data()

        # Llamar al constructor padre
        super().__init__(df, **kwargs)

    def _prepare_data(self):
        """Prepara los datos y calcula estadísticas para normalización"""
        self.data = self.df
        self.closings = self.df.close.values

        # Calcular estadísticas para normalización
        self.price_mean = np.mean(self.closings)
        self.price_std = np.std(self.closings)

        # Calcular estadísticas para indicadores técnicos
        self.tech_means = {}
        self.tech_stds = {}
        for tech in self.tech_indicator_list:
            self.tech_means[tech] = np.mean(self.df[tech].values)
            self.tech_stds[tech] = np.std(self.df[tech].values)

    def _normalize_observation(self, state):
        """Normaliza el estado usando z-score"""
        normalized = []

        # Normalizar balance (usando el balance inicial como referencia)
        normalized.append((state[0] - self.initial_amount) / (self.initial_amount * 0.5))

        # Normalizar acciones poseídas (usando el máximo posible como referencia)
        max_shares = self.initial_amount / np.mean(self.closings)
        normalized.append(state[1] / max_shares)

        # Normalizar precio actual
        normalized.append((state[2] - self.price_mean) / self.price_std)

        # Normalizar indicadores técnicos
        for i, tech in enumerate(self.tech_indicator_list):
            value = state[3 + i]
            norm_value = (value - self.tech_means[tech]) / (self.tech_stds[tech] + 1e-8)
            normalized.append(norm_value)

        return np.array(normalized, dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Necesario para compatibilidad con gymnasium
        super().reset(seed=seed)

        self.day = 0
        self.state_memory = [0] * self.stock_dim
        self.asset_memory = [self.initial_amount]
        self.date_memory = [self.df.date.iloc[0]]

        # Inicializar el estado
        self.state = self._initiate_state()

        # Reset infos
        self.infos = {
            "day": self.day,
            "portfolio_value": self.initial_amount,
            "total_reward": 0.0,
        }

        return self._normalize_observation(self.state), self.infos

    def step(self, action):
        """
        Ejecuta un paso en el entorno.

        Args:
            action: 0 (vender), 1 (hold), 2 (comprar)
        """
        self.day += 1
        terminated = self.day >= len(self.df.index.unique()) - 1
        truncated = False

        if not terminated:
            # Obtener precio actual y calcular valor inicial del portafolio
            current_price = float(self.closings[self.day])
            prev_price = float(self.closings[self.day - 1])
            price_change = (current_price - prev_price) / prev_price

            cash_before = float(self.state[0])
            shares_before = float(self.state_memory[-1])
            portfolio_before = cash_before + (shares_before * current_price)

            # Mapear la acción discreta a un valor de intensidad
            action_intensity = self.action_map[action]
            trade_executed = False
            shares_traded = 0

            if action_intensity > 0:  # Comprar
                if cash_before >= current_price:
                    max_possible_shares = cash_before / (current_price * (1 + self.transaction_cost_pct))
                    # Usar la intensidad de la acción para determinar cuánto comprar
                    shares_to_buy = max_possible_shares * action_intensity
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost_pct)

                    if cost <= cash_before:
                        self.state[0] = cash_before - cost
                        self.state_memory[-1] = shares_before + shares_to_buy
                        shares_traded = shares_to_buy
                        trade_executed = True

            elif action_intensity < 0:  # Vender
                if shares_before > 0:
                    # Usar el valor absoluto de la intensidad para determinar cuánto vender
                    shares_to_sell = shares_before * abs(action_intensity)
                    revenue = shares_to_sell * current_price * (1 - self.transaction_cost_pct)

                    self.state[0] = cash_before + revenue
                    self.state_memory[-1] = shares_before - shares_to_sell
                    shares_traded = -shares_to_sell
                    trade_executed = True

            # Actualizar estado y calcular nuevo valor del portafolio
            self.state = self._update_state()
            cash_after = float(self.state[0])
            shares_after = float(self.state_memory[-1])
            portfolio_after = cash_after + (shares_after * current_price)

            # Calcular recompensa base
            immediate_reward = portfolio_after - portfolio_before

            # Recompensas adicionales basadas en la precisión de la acción
            if trade_executed:
                if action_intensity > 0 and price_change > 0:  # Compra acertada
                    immediate_reward += abs(price_change) * portfolio_before * action_intensity * 0.2
                elif action_intensity < 0 and price_change < 0:  # Venta acertada
                    immediate_reward += abs(price_change) * portfolio_before * abs(action_intensity) * 0.2
            else:
                if action == 3:  # Hold deliberado (acción central)
                    if (price_change > 0 and shares_after > 0) or (price_change < 0 and cash_after > 0):
                        immediate_reward += abs(price_change) * portfolio_before * 0.05
                else:  # Intento fallido de compra/venta
                    immediate_reward -= portfolio_before * 0.001

            # Actualizar historial
            self.asset_memory.append(portfolio_after)
            self.date_memory.append(self.df.iloc[self.day].date)

            info = {
                'day': self.day,
                'total_assets': float(portfolio_after),
                'cash_balance': float(cash_after),
                'num_shares': float(shares_after),
                'current_price': float(current_price),
                'price_change': float(price_change),
                'action_taken': int(action),
                'action_intensity': float(action_intensity),
                'trade_executed': trade_executed,
                'shares_traded': float(shares_traded),
                'portfolio_change': float(portfolio_after - portfolio_before),
                'reward': float(immediate_reward)
            }

            return self._normalize_observation(self.state), immediate_reward, terminated, truncated, info

        else:
            # Estado terminal
            final_price = float(self.closings[self.day])
            cash_final = float(self.state[0])
            shares_final = float(self.state_memory[-1])
            portfolio_final = cash_final + (shares_final * final_price)

            info = {
                'day': self.day,
                'total_assets': float(portfolio_final),
                'cash_balance': float(cash_final),
                'num_shares': float(shares_final),
                'current_price': float(final_price),
                'action_taken': 1,
                'terminal_info': 'Episode ended',
                'final_portfolio': float(portfolio_final),
                'initial_portfolio': float(self.asset_memory[0]),
                'total_return': float((portfolio_final - self.asset_memory[0]) / self.asset_memory[0])
            }

            return self._normalize_observation(self.state), 0, terminated, truncated, info

    def _update_state(self):
        try:
            state = []

            # Balance actual (asegurar que sea float)
            state.append(float(self.state[0]))

            # Estado actual de acciones poseídas
            state.extend([float(self.state_memory[-1]) for _ in range(self.stock_dim)])

            # Precio actual de cada ticker
            state.extend([float(self.closings[self.day]) for _ in range(self.stock_dim)])

            # Indicadores técnicos actuales
            for tech in self.tech_indicator_list:
                value = float(self.df.iloc[self.day][tech])
                state.append(value)

            return np.array(state, dtype=np.float32)

        except Exception as e:
            print(f"Error en _update_state: {str(e)}")
            print(f"Variables disponibles:")
            print(f"day: {self.day}")
            print(f"state: {self.state}")
            print(f"state_memory: {self.state_memory}")
            print(f"closings shape: {self.closings.shape}")
            raise

    def _initiate_state(self):
        if len(self.df.tic.unique()) > 1:
            raise ValueError("Este entorno está diseñado para un solo ticker")

        try:
            state = []

            # Balance inicial
            state.append(float(self.initial_amount))

            # Estado inicial de acciones poseídas
            state.extend([float(self.state_memory[0]) for _ in range(self.stock_dim)])

            # Precio actual de cada ticker
            state.extend([float(self.closings[0]) for _ in range(self.stock_dim)])

            # Indicadores técnicos
            for tech in self.tech_indicator_list:
                value = float(self.data[tech].values[0])
                state.append(value)

            print("\nDEBUG desde _initiate_state:")
            print(f"state antes de convertir: {state}")
            print(f"longitud del estado: {len(state)}")
            print(f"state_space esperado: {self.state_space}")
            print(f"tipos de los elementos: {[type(x) for x in state]}")

            # Verificar que la longitud del estado coincida con state_space
            if len(state) != self.state_space:
                raise ValueError(f"La longitud del estado ({len(state)}) no coincide con state_space ({self.state_space})")

            state_array = np.array(state, dtype=np.float32)
            print(f"state después de convertir: {state_array}")
            print(f"shape del state: {state_array.shape}")

            return state_array

        except Exception as e:
            print(f"Error en _initiate_state: {str(e)}")
            print(f"Variables disponibles:")
            print(f"initial_amount: {self.initial_amount}")
            print(f"state_memory: {self.state_memory}")
            print(f"closings: {self.closings[:5]}")
            print(f"tech_indicator_list: {self.tech_indicator_list}")
            raise

    def get_state(self):
        """Devuelve el estado actual del entorno."""
        return self.state

    def get_portfolio_value_history(self):
        """Devuelve el historial de valores del portafolio."""
        return self.asset_memory
