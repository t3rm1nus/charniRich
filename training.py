import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from environment import calculate_state_space, make_env

def entrenar_en_tramo(tickers, etapa, hyperparams, df, indicadores, device, total_timesteps=10_000, prev_model_path=None, writer=None):
    """
    Train a PPO model for a specific stage with given tickers.

    Args:
        tickers (list): List of ticker symbols.
        etapa (str): Stage identifier (e.g., 'etapa_1', '1_hp1').
        hyperparams (dict): Hyperparameters for the PPO model.
        df (pd.DataFrame): DataFrame with financial data.
        indicadores (list): List of technical indicator column names.
        device (torch.device): Device to use for training (CPU/GPU).
        total_timesteps (int): Total training timesteps.
        prev_model_path (str, optional): Path to a previous model for fine-tuning.
        writer (SummaryWriter, optional): TensorBoard writer for logging.

    Returns:
        tuple: (model_path, val_env, df_test) - Path to saved model, validation environment, and test DataFrame.
    """
    state_space = calculate_state_space(len(tickers), len(indicadores))
    common_kwargs = {
        "stock_dim": len(tickers),
        "hmax": 0.02,
        "initial_amount": 550.0,
        "buy_cost_pct": [0.001] * len(tickers),
        "sell_cost_pct": [0.001] * len(tickers),
        "reward_scaling": 0.01,
        "state_space": state_space,
        "action_space": len(tickers),
        "tech_indicator_list": indicadores,
        "print_verbosity": 10,
        "max_steps": 1000,
        "num_stock_shares": [0.0] * len(tickers),
    }

    if etapa.startswith("1_hp"):
        model_num = int(etapa.split("_hp")[1])
        modelo_path = f"models/modelo_1_hp{model_num}.zip"
        if os.path.exists(modelo_path):
            val_env = SubprocVecEnv([make_env(df[df['tic'].isin(tickers)].copy(), "validación", tickers, common_kwargs, indicadores)])
            df_test = df[df['tic'].isin(tickers)].copy()
            return modelo_path, val_env, df_test

    df_tramo = df[df['tic'].isin(tickers)].copy()
    df_tramo = df_tramo.sort_values(['date', 'tic']).reset_index(drop=True)
    from data_processing import filtrar_dias_incompletos
    df_tramo = filtrar_dias_incompletos(df_tramo, tickers)
    if len(df_tramo) == 0:
        raise ValueError("No valid data after filtering")

    fechas_unicas = sorted(df_tramo["date"].unique())
    num_fechas = len(fechas_unicas)
    if num_fechas < 3:
        raise ValueError(f"Insufficient unique dates: {num_fechas}")
    split_train = int(num_fechas * 0.8)
    split_val = int(num_fechas * 0.9)

    df_train = df_tramo[df_tramo["date"].isin(fechas_unicas[:split_train])].copy()
    df_val = df_tramo[df_tramo["date"].isin(fechas_unicas[split_train:split_val])].copy()
    df_test = df_tramo[df_tramo["date"].isin(fechas_unicas[split_val:])].copy()

    for subset, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        if len(subset) == 0:
            raise ValueError(f"Subconjunto {name} está vacío")
        first_date = subset['date'].min()
        if len(subset[subset['date'] == first_date]) != len(tickers):
            raise ValueError(f"Primera fecha de {name} no contiene todos los tickers")

    num_cpu = 2
    train_env = SubprocVecEnv([make_env(df_train, "entrenamiento", tickers, common_kwargs, indicadores) for _ in range(num_cpu)])
    val_env = SubprocVecEnv([make_env(df_val, "validación", tickers, common_kwargs, indicadores) for _ in range(num_cpu)])

    policy_kwargs = {
        "activation_fn": torch.nn.ReLU,
        "net_arch": dict(pi=[64, 64], vf=[64, 64])
    }

    if prev_model_path:
        modelo = PPO.load(prev_model_path, env=train_env, device=device, tensorboard_log="./tensorboard/", **hyperparams)
    else:
        modelo = PPO('MlpPolicy', train_env, verbose=1, tensorboard_log="./tensorboard/", policy_kwargs=policy_kwargs, device=device, batch_size=4096, **hyperparams)

    def callback(locals_, globals_):
        if 'infos' in locals_ and locals_['infos']:
            for info in locals_['infos']:
                if 'episode' in info and 'r' in info['episode']:
                    writer.add_scalar('train/reward', info['episode']['r'], locals_['self'].num_timesteps)
                if 'episode' in info and 'l' in info['episode']:
                    writer.add_scalar('train/ep_len', info['episode']['l'], locals_['self'].num_timesteps)
        return True

    with tqdm(total=total_timesteps, desc=f"Entrenamiento Etapa {etapa}") as pbar:
        modelo.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name=f"etapa_{etapa}", progress_bar=False)

    modelo_path = f"models/modelo_{etapa}.zip"
    modelo.save(modelo_path)

    plt.figure(figsize=(12, 6))
    rewards = modelo.ep_info_buffer['r'] if hasattr(modelo, 'ep_info_buffer') else []
    if rewards:
        plt.plot(rewards, label='Recompensa por episodio')
        plt.plot(pd.Series(rewards).rolling(10).mean(), 'r-', label='Media móvil (10)')
        plt.title(f"Recompensas durante el entrenamiento - Etapa {etapa}")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plots/recompensas_etapa_{etapa}.png", dpi=300)
        plt.close()

    mean_reward, std_reward = evaluate_policy(modelo, val_env, n_eval_episodes=10)

    del modelo
    torch.cuda.empty_cache()
    return modelo_path, val_env, df_test


def check_saved_models(tickers_list):
    """
    Check for existing saved models for each stage.

    Args:
        tickers_list (list): List of ticker lists for each stage.

    Returns:
        list: List of completed stage indices.
    """
    completed_stages = []
    for i, tickers in enumerate(tickers_list, 1):
        if i == 1:
            completed_hps = []
            for hp in range(1, 4):
                model_path = f"models/modelo_1_hp{hp}.zip"
                if os.path.exists(model_path):
                    completed_hps.append(hp)
            if len(completed_hps) == 3:
                completed_stages.append(i)
        else:
            model_path = f"models/modelo_etapa_{i}.zip"
            if os.path.exists(model_path):
                completed_stages.append(i)
    return completed_stages


def load_last_completed_model(tickers_list, df, indicadores, device):
    """
    Load the last completed model from the training pipeline.

    Args:
        tickers_list (list): List of ticker lists for each stage.
        df (pd.DataFrame): DataFrame with financial data.
        indicadores (list): List of technical indicator column names.
        device (torch.device): Device to use for model loading.

    Returns:
        tuple: (model_path, last_stage) - Path to the last model and its stage index.
    """
    completed = check_saved_models(tickers_list)
    if not completed:
        return None, 0
    last_stage = max(completed)
    if last_stage == 1:
        best_reward = -np.inf
        best_model_path = None
        for hp in range(1, 4):
            model_path = f"models/modelo_1_hp{hp}.zip"
            if os.path.exists(model_path):
                try:
                    state_space = calculate_state_space(len(tickers_list[0]), len(indicadores))
                    val_env = SubprocVecEnv([make_env(
                        df[df['tic'].isin(tickers_list[0])].copy(),
                        "validación",
                        tickers_list[0],
                        {
                            "stock_dim": len(tickers_list[0]),
                            "hmax": 0.02,
                            "initial_amount": 550.0,
                            "num_stock_shares": [0.0] * len(tickers_list[0]),
                            "buy_cost_pct": [0.001] * len(tickers_list[0]),
                            "sell_cost_pct": [0.001] * len(tickers_list[0]),
                            "reward_scaling": 0.01,
                            "state_space": state_space,
                            "action_space": len(tickers_list[0]),
                            "tech_indicator_list": indicadores,
                            "print_verbosity": 10,
                            "max_steps": 1000
                        },
                        indicadores
                    )])
                    model = PPO.load(model_path, env=val_env, device=device)
                    mean_reward, _ = evaluate_policy(model, val_env, n_eval_episodes=10)
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        best_model_path = model_path
                    del model, val_env
                    torch.cuda.empty_cache()
                except Exception:
                    continue
        return best_model_path, last_stage
    else:
        model_path = f"models/modelo_etapa_{last_stage}.zip"
        return model_path, last_stage