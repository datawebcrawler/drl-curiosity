import sys

from sample_factory.enjoy import enjoy

from utils import register_custom_env_envs, parse_args

def custom_env_override_defaults(_env, parser):
    # Modify the default arguments when using this env.
    # These can still be changed from the command line. See configuration guide for more details.
    parser.set_defaults(
        # normalize_input=True,
        obs_scale=255.0,
        gamma=0.99,
        learning_rate=1e-4,
        lr_schedule="constant",
        adam_eps=1e-5,  
        train_for_env_steps=100_000_000,
        algo="APPO",
        env_frameskip=4,
        use_rnn=True,
        batch_size=2048, 
        num_workers=4, 
        num_envs_per_worker=4, 
        device="gpu",
        num_policies=1,
        # experiment="glaucoma250_intrinsic",
        experiment="doom_health_gathering/glaucoma_/01_glaucoma_see_1111_t.f.e.ste_20000000_alg_APPO_u.rnn_True_b.siz_2048_g.lev_100_env_health_gathering_glaucoma",
        glaucoma_level=200,
        # save_video = True,
        # video_frames=6000,
    )

if __name__ == "__main__":
    register_custom_env_envs()
    cfg = parse_args(custom_env_override_defaults, evaluation=True)
    status = enjoy(cfg)
    sys.exit(status)
