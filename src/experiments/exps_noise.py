from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from utils import register_custom_env_envs

# É importante chamar o registro antes de definir os experimentos
register_custom_env_envs()

# --- Parâmetros Base ---
# Estes são os parâmetros que serão compartilhados por todos os experimentos.
# Note que os valores estão dentro de listas, como o ParamGrid espera.
_base_params = [
    ("seed", [2222]),
    ("train_for_env_steps", [3_000_000]),
    ("algo", ["APPO"]),
    ("use_rnn", [True]),
    ("batch_size", [1024]),
    ("env", ["health_gathering_glaucoma"]),
    # Parâmetros otimizados
    ("ae_lr", [8e-4]),
    ("ae_loss_coeff", [0.5]),
    ("learning_rate", [1e-4]),
    ("exploration_loss_coeff", [0.0002]),
    ("intrinsic_reward_coeff", [0.01]),
    ("intrinsic_reward_anneal_steps", [5_000_000]),
]

WITH = [True]
WITHOUT = [False]
BOTH = [False, True]
DAE_CURRENT_CHOICE = BOTH
RND_CURRENT_CHOICE = WITHOUT
GLOBAL_STEPS_BEFORE_NOISE = 5 # O ruído começa após 5 passos sem kit


# --- Experimento 1: Controle (Sem Ruído) ---
_params_no_noise = ParamGrid(
    _base_params + [ # Combinamos a lista base com os parâmetros variáveis
        ("noise_type", ["none"]),
        ("noise_intensity", [0.0]),
        ("use_rnd", RND_CURRENT_CHOICE),
        ("use_aux_autoencoder", DAE_CURRENT_CHOICE),
    ]
)

# --- Experimento 2: Visão de Túnel ---
_params_tunnel = ParamGrid(
    _base_params + [
        ("noise_type", ["tunnel"]),
        ("noise_intensity", [0.1, 0.3, 0.5]), # é o decay_factor (um float de 0.0 a 1.0)
        ("steps_before_noise", [GLOBAL_STEPS_BEFORE_NOISE]), # O ruído começa após 20 passos sem kit
        ("noise_increment_step", [1.0]), # A cada passo, 1% a mais de ruído é adicionado
        ("use_rnd", RND_CURRENT_CHOICE),
        ("use_aux_autoencoder", DAE_CURRENT_CHOICE),
    ]
)

# --- Experimento 3: Manchas Cegas ---
_params_scotomas = ParamGrid(
    _base_params + [
        ("noise_type", ["scotomas"]),
        ("noise_intensity", [20, 50, 100]), # é o número de manchas (um inteiro, e.g., 10)
        ("steps_before_noise", [20]), # O ruído começa após 20 passos sem kit
        ("noise_increment_step", [1.0]), # A cada passo, 1% a mais de ruído é adicionado
        ("use_rnd", RND_CURRENT_CHOICE),
        ("use_aux_autoencoder", DAE_CURRENT_CHOICE),    ]
)

# --- Experimento 4: Sal e Pimenta ---
_params_salt_pepper = ParamGrid(
    _base_params + [
        ("noise_type", ["salt_pepper"]),
        ("noise_intensity", [50.0, 80.0, 99.0]),
        ("steps_before_noise", [20]), # O ruído começa após 20 passos sem kit
        ("noise_increment_step", [1.0]), # A cada passo, 1% a mais de ruído é adicionado
        ("use_rnd", RND_CURRENT_CHOICE),
        ("use_aux_autoencoder", DAE_CURRENT_CHOICE),    ]
)

# --- Montagem da Descrição da Execução ---
_experiments = [
    Experiment(
        "NoiseTest_Control",
        "uv run sf/train.py --num_workers=4 --num_envs_per_worker=4 --num_policies=1 --device=gpu --obs_scale=255.0",
        _params_no_noise.generate_params(randomize=False),
    ),
    Experiment(
        "NoiseTest_Tunnel",
        "uv run sf/train.py --num_workers=4 --num_envs_per_worker=4 --num_policies=1 --device=gpu --obs_scale=255.0",
        _params_tunnel.generate_params(randomize=False),
    ),
    Experiment(
        "NoiseTest_Scotomas",
        "uv run sf/train.py --num_workers=4 --num_envs_per_worker=4 --num_policies=1 --device=gpu --obs_scale=255.0",
        _params_scotomas.generate_params(randomize=False),
    ),
    Experiment(
        "NoiseTest_SaltPepper",
        "uv run sf/train.py --num_workers=4 --num_envs_per_worker=4 --num_policies=1 --device=gpu --obs_scale=255.0",
        _params_salt_pepper.generate_params(randomize=False),
    ),
]

# A variável final que o launcher do Sample Factory espera encontrar
RUN_DESCRIPTION = RunDescription("doom_noise_comparison", experiments=_experiments, customize_experiment_name=False
)

# log de execução dos treinos
# full 5_000_000 de passos
# [2025-08-12 10:22:07,711][03559] Done! Total time: 6h 37min 52s

