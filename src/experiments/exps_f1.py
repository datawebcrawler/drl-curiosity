# experiments.py (versão para o Teste 1.1)
from utils import register_custom_env_envs
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

register_custom_env_envs()

# --- FASE 1: TESTE 1.1 ---
# Foco: Encontrar a melhor taxa de aprendizado para o autoencoder (ae_lr)
# Mantemos ae_loss_coeff fixo e variamos ae_lr.
# Executaremos cada treino por 5 milhões de passos.
_params = ParamGrid(
    [
        ("seed", [1111]), # Usamos uma única seed para comparabilidade
        # ("seed", [1111, 2222, 3333]), # seeds para o artigo o ideal sao pelo menos 5
        ("train_for_env_steps", [5_000_000]), # Duração para o artigo
        # --- Parâmetros Padrão do Ambiente ---
        ("algo", ["APPO"]),
        ("use_rnn", [True]),
        ("batch_size", [1024]),
        ("glaucoma_level", [20, 30, 40]), # Usamos um único nível de glaucoma
        ("env", ["health_gathering_glaucoma"]),

        # --- Intervalo dos parâmetros para serem otimizados ---
        # ("ae_loss_coeff", [0.1, 0.5, 1.0]),
        # --- Hiperparâmetro a ser Otimizado ---
        # ("ae_lr", [6e-4, 1e-3]), # Variamos a taxa de aprendizado para achar os Melhores
        # ("intrinsic_reward_coeff", [0.005, 0.01, 0.02]), # Variamos o "volume" da curiosidade
        # ("learning_rate", [5e-5, 1e-4, 1.5e-4]), # Variamos a LR principal
        # ("exploration_loss_coeff", [0.001, 0.0005, 0.0002]),

        # --- Back up dos Parâmetros Otimizados (da nossa implementação) ---
        # ("use_rnd", [True]),
        # ("use_aux_autoencoder", [True]),
        # ("ae_loss_coeff", [0.5]),   # AE loss otimizado
        # ("ae_lr", [8e-4]),          # Taxa de aprendizado do AE otimizado
        # ("intrinsic_reward_coeff", [0.02]), # Coeficiente do intrinsic reward otimizado
        # ("intrinsic_reward_final_coeff", [0.0001]), # ✅ Para testar o RND puro sem decaimento, defina o coeficiente final para ser igual ao inicial
        # ("intrinsic_reward_anneal_steps", [2_500_000]), # Valor final otimizado fixo
        # ("exploration_loss_coeff", [0.0002]),
        # ("learning_rate", [5e-5]), # O Valor 1e-4 também foi muito bom, convergindo mais rápido


        # --- Parâmetros Otimizados (da nossa implementação) ---
        ("use_rnd", [False]),
        ("use_aux_autoencoder", [False]),
        # ("ae_loss_coeff", [0.5]),   # AE loss otimizado
        # ("ae_lr", [8e-4]),          # Taxa de aprendizado do AE otimizado
        # ("intrinsic_reward_coeff", [0.02]), # Coeficiente do intrinsic reward otimizado
        # ("intrinsic_reward_final_coeff", [0.0001]), # ✅ Para testar o RND puro sem decaimento, defina o coeficiente final para ser igual ao inicial
        # ("intrinsic_reward_anneal_steps", [2_500_000]), # Valor final otimizado fixo
        # ("exploration_loss_coeff", [0.0002]),
        # ("learning_rate", [5e-5]), # O Valor 1e-4 também foi muito bom, convergindo mais rápido


    ]
)

_experiments = [
    Experiment(
        # O nome do grupo de experimentos. Todos os testes aparecerão sob esta pasta.
        "Optimize_AELR", 
        "uv run sf/train.py --num_workers=4 --num_envs_per_worker=4 --num_policies=1 --device=gpu --obs_scale=255.0",
        _params.generate_params(randomize=False),
    ),
]

RUN_DESCRIPTION = RunDescription("doom_health_gathering", experiments=_experiments)
