import asyncio
import platform
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
import pygame # For quit event and display surface in final eval

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from Env import (
    SplixIOEnv, SplixIOevalEnv, GridToImageWrapper,
    ACTUAL_GRID_SIZE, OBS_SIZE,
    FIXED_EVAL_START_POSITIONS, FIXED_EVAL_START_DIRECTION
)
from model import CustomCNN

HIDDEN_DIM = 128


class EvalAndSaveCallback(BaseCallback):
    def __init__(self, eval_freq: int, n_eval_episodes: int, log_path: str,
                 eval_env_kwargs: dict, total_training_timesteps: int, verbose: int = 1):
        super(EvalAndSaveCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.eval_env_kwargs = eval_env_kwargs
        self.best_mean_reward = -np.inf
        self.eval_env = None
        self.pbar = None
        self.total_training_timesteps = total_training_timesteps
        self.train_reward_history = []
        self.eval_reward_history = []
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)
        self.fixed_start_positions = FIXED_EVAL_START_POSITIONS
        self.fixed_start_direction = FIXED_EVAL_START_DIRECTION

    def _init_callback(self) -> None:
        raw_eval_env = SplixIOevalEnv(**self.eval_env_kwargs)
        self.eval_env = GridToImageWrapper(raw_eval_env)
        if self.verbose > 0:
            self.pbar = tqdm(total=self.total_training_timesteps, desc="Training", unit="timestep")

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(self.num_timesteps - self.pbar.n)

        log_items = {}
        if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
            recent_train_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            if recent_train_rewards:
                mean_train_reward = np.mean(recent_train_rewards)
                log_items['train_reward'] = f"{mean_train_reward:.2f}"
                self.train_reward_history.append((self.num_timesteps, mean_train_reward))

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            episode_rewards = []
            num_fixed_starts = len(self.fixed_start_positions)
            for i in range(self.n_eval_episodes):
                current_start_idx = i % num_fixed_starts
                current_start_pos = self.fixed_start_positions[current_start_idx]
                obs, _ = self.eval_env.reset(start_pos=current_start_pos, start_direction=self.fixed_start_direction)
                done = False
                truncated = False
                current_episode_reward = 0
                while not (done or truncated):
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, reward, done_step, truncated_step, info = self.eval_env.step(action.item()) # Renamed done, truncated
                    done = done_step
                    truncated = truncated_step
                    current_episode_reward += reward
                episode_rewards.append(current_episode_reward)

            mean_reward = np.mean(episode_rewards) if episode_rewards else -np.inf
            std_reward = np.std(episode_rewards) if episode_rewards else 0
            log_items['eval_reward'] = f"{mean_reward:.2f} +/- {std_reward:.2f}"
            self.eval_reward_history.append((self.num_timesteps, mean_reward, std_reward))

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.log_path is not None:
                    save_path = os.path.join(self.log_path, "best_model.zip")
                    self.model.save(save_path)
                    if self.verbose > 0:
                        print(f"\nStep {self.num_timesteps}: New best model saved to {save_path} with mean reward: {self.best_mean_reward:.2f}")
        if self.best_mean_reward > -np.inf:
            log_items['best_eval_reward'] = f"{self.best_mean_reward:.2f}"
        if self.verbose > 0 and self.pbar and log_items:
            self.pbar.set_postfix(log_items, refresh=True)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()
            self.pbar = None
        if self.eval_env:
            self.eval_env.close()


def plot_reward_curves(train_history, eval_history, save_dir):
    if not train_history and not eval_history:
        print("No reward history to plot.")
        return
    os.makedirs(save_dir, exist_ok=True)
    if train_history:
        timesteps_train, rewards_train = zip(*train_history)
        plt.figure(figsize=(10, 5))
        plt.plot(timesteps_train, rewards_train, label='Mean Training Reward (rolling)')
        plt.xlabel('Timesteps'); plt.ylabel('Mean Reward'); plt.title('Training Reward Curve')
        plt.legend(); plt.grid(True); plt.savefig(os.path.join(save_dir, "train_reward.png")); plt.close()
        print(f"Training reward curve saved to {os.path.join(save_dir, 'train_reward.png')}")
    if eval_history:
        timesteps_eval, rewards_eval_mean, rewards_eval_std = zip(*eval_history)
        rewards_eval_mean_np = np.array(rewards_eval_mean) # Renamed for clarity
        rewards_eval_std_np = np.array(rewards_eval_std)   # Renamed for clarity
        plt.figure(figsize=(10, 5))
        plt.plot(timesteps_eval, rewards_eval_mean_np, label='Mean Evaluation Reward')
        plt.fill_between(timesteps_eval, rewards_eval_mean_np - rewards_eval_std_np, rewards_eval_mean_np + rewards_eval_std_np, alpha=0.2, label='Std Dev')
        plt.xlabel('Timesteps'); plt.ylabel('Mean Reward'); plt.title('Evaluation Reward Curve')
        plt.legend(); plt.grid(True); plt.savefig(os.path.join(save_dir, "eval_reward.png")); plt.close()
        print(f"Evaluation reward curve saved to {os.path.join(save_dir, 'eval_reward.png')}")


async def train_and_evaluate(total_timesteps=100000, eval_episodes_final=10, load_model_path=None,
                             eval_freq_callback=5000, n_eval_episodes_callback=5, load_model_flag=False,
                             save_video_path: str = None):
    MODEL_SAVE_DIR = "./ppo_splix_partial_obs_checkpoints/"
    FINAL_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "ppo_splix_io_partial_obs_final")
    ENV_STATS_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "ppo_splix_io_partial_obs_env_stats.pkl")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    if save_video_path:
        os.makedirs(save_video_path, exist_ok=True)

    env_kwargs_train = {'grid_size': ACTUAL_GRID_SIZE, 'obs_size': OBS_SIZE, 'render_mode': 'none'}
    env_kwargs_eval_callback = {'grid_size': ACTUAL_GRID_SIZE, 'obs_size': OBS_SIZE, 'render_mode': 'none'}
    env_kwargs_final_eval = {'grid_size': ACTUAL_GRID_SIZE, 'obs_size': OBS_SIZE, 'render_mode': 'none' if save_video_path else 'none'}

    vec_env = make_vec_env(SplixIOEnv, n_envs=4, env_kwargs=env_kwargs_train, wrapper_class=GridToImageWrapper)
    env = VecNormalize(vec_env, norm_obs=False, norm_reward=False, gamma=0.99)

    eval_callback = EvalAndSaveCallback(
        eval_freq=max(eval_freq_callback // env.num_envs, 1),
        n_eval_episodes=n_eval_episodes_callback,
        log_path=MODEL_SAVE_DIR,
        eval_env_kwargs=env_kwargs_eval_callback,
        total_training_timesteps=total_timesteps
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=HIDDEN_DIM),
        normalize_images=False
    )

    model = None
    if load_model_path and os.path.exists(load_model_path + ".zip"):
        print(f"Loading pre-trained model from {load_model_path}.zip ...")
        model = PPO.load(load_model_path, env=env, custom_objects={'policy_kwargs': policy_kwargs}) # Pass env for VecNormalize wrapper
        env_stats_path_to_load = load_model_path + "_env_stats.pkl" # Adjusted name to match saving
        if os.path.exists(env_stats_path_to_load):
            print(f"Loading environment stats from {env_stats_path_to_load}...")
            # When loading a model that was trained with VecNormalize,
            # the loaded VecNormalize wrapper should be used.
            # PPO.load() handles setting the loaded VecNormalize wrapper if `env` is passed.
            # However, if you want to ensure the stats are loaded into the *current* `env` instance:
            temp_env = VecNormalize.load(env_stats_path_to_load, vec_env)
            env.obs_rms = temp_env.obs_rms
            env.ret_rms = temp_env.ret_rms
            env.clip_obs = temp_env.clip_obs
            env.clip_reward = temp_env.clip_reward
            del temp_env # clean up
            print("Environment stats loaded into current VecNormalize wrapper.")
        else:
            print(f"Warning: Environment stats file not found at {env_stats_path_to_load}. Using fresh stats for loaded model.")
        env.training = True
        model.set_env(env) # Ensure the potentially updated env is set
    else:
        if load_model_path: print(f"Warning: Model file not found at {load_model_path}.zip. Initializing new model.")
        print("Initializing new PPO model...")
        model = PPO("MultiInputPolicy", env, verbose=0, learning_rate=3e-4, n_steps=2048, batch_size=64,
                    n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    policy_kwargs=policy_kwargs, ent_coef=0.01)

    if not load_model_flag:
        print(f"Starting training for {total_timesteps} timesteps...")
        model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=False)
        print(f"Saving final model to {FINAL_MODEL_SAVE_PATH}.zip and environment stats to {ENV_STATS_SAVE_PATH}...")
        model.save(FINAL_MODEL_SAVE_PATH)
        env.save(ENV_STATS_SAVE_PATH) # Save stats of the current env
    else:
        print("Skipping training as load_model_flag is True.")

    plot_reward_curves(eval_callback.train_reward_history, eval_callback.eval_reward_history, MODEL_SAVE_DIR)

    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.zip")
    model_to_eval = model # Default to current model (either freshly trained or loaded)
    print(best_model_path,os.path.exists(best_model_path), load_model_flag, load_model_path)
    if os.path.exists(best_model_path) : # Prioritize best model if training occurred
        print(f"Loading best saved model from {best_model_path} for final evaluation...")
        model_to_eval = PPO.load(best_model_path, custom_objects={'policy_kwargs': policy_kwargs})
    elif load_model_flag and load_model_path and os.path.exists(load_model_path + ".zip"):
         print(f"Using initially loaded model for final evaluation: {load_model_path}.zip")
         # model_to_eval is already the loaded model
    elif not os.path.exists(best_model_path) and load_model_flag and load_model_path:
        print(f"Initially loaded model specified but not found at {load_model_path}.zip. Using current model if available.")
    else:
        print("Using the final/current model for evaluation.")


    if eval_episodes_final > 0 and model_to_eval is not None:
        print(f"\nStarting final evaluation for {eval_episodes_final} episodes...")
        eval_env_raw_final = SplixIOevalEnv(**env_kwargs_final_eval)
        eval_env_wrapped_final = GridToImageWrapper(eval_env_raw_final)
        
        fixed_start_positions_final = FIXED_EVAL_START_POSITIONS
        fixed_start_direction_final = FIXED_EVAL_START_DIRECTION
        num_fixed_starts_final = len(fixed_start_positions_final)
        video_writer_global = None 

        try:
            total_rewards = []
            total_steps = []
            for episode_idx in tqdm(range(eval_episodes_final), desc="Final Evaluation", unit="episode"):
                current_start_idx = episode_idx % num_fixed_starts_final
                start_pos = fixed_start_positions_final[current_start_idx]
                start_dir = fixed_start_direction_final
                video_writer_episode = None

                if save_video_path and eval_env_wrapped_final.render_mode == 'human':
                    try:
                        video_fps = eval_env_wrapped_final.unwrapped.FPS
                        save_video_path_i = os.path.join(save_video_path, f"episode_{episode_idx}_pos_{start_pos[0]}_{start_pos[1]}.mp4")
                        video_writer_episode = imageio.get_writer(save_video_path_i, fps=video_fps, macro_block_size=None)
                        video_writer_global = video_writer_episode 
                    except Exception as e:
                        print(f"Could not initialize video writer for episode {episode_idx}: {e}.")
                        video_writer_episode = None; video_writer_global = None

                obs, _ = eval_env_wrapped_final.reset(start_pos=start_pos, start_direction=start_dir)
                terminated, truncated = False, False
                episode_reward, episode_steps = 0, 0
                while not (terminated or truncated):
                    action_array, _ = model_to_eval.predict(obs, deterministic=False)
                    action = action_array.item()
                    obs, reward, term_step, trunc_step, info = eval_env_wrapped_final.step(action) # Renamed
                    terminated = term_step; truncated = trunc_step
                    episode_reward += reward
                    episode_steps += 1
                    if eval_env_wrapped_final.render_mode == 'human':
                        eval_env_wrapped_final.render()
                        if video_writer_episode:
                            frame_surface = pygame.display.get_surface()
                            if frame_surface:
                                frame_array = pygame.surfarray.array3d(frame_surface)
                                frame_array_rgb = frame_array.transpose([1, 0, 2])
                                video_writer_episode.append_data(frame_array_rgb)
                        for event in pygame.event.get(): 
                            if event.type == pygame.QUIT:
                                print("Pygame window closed during final evaluation.")
                                terminated = True; break 
                        if terminated: break 
                        await asyncio.sleep(0.01) 
                
                total_rewards.append(episode_reward)
                total_steps.append(episode_steps)
                if video_writer_episode:
                    try: video_writer_episode.close()
                    except Exception as e: print(f"Error closing video writer for ep {episode_idx}: {e}")
                video_writer_global = None 
                if terminated and pygame.display.get_init() and any(event.type == pygame.QUIT for event in pygame.event.get(pygame.QUIT)): # Check if quit was due to window close
                     break
            if total_rewards:
                print(f"\nAverage reward over {len(total_rewards)} final episodes: {np.mean(total_rewards):.2f}+-{np.std(total_rewards):.2f}")
                print(f"Average steps over {len(total_steps)} final episodes: {np.mean(total_steps):.2f}")
            else: print("No final evaluation episodes were completed.")
        finally:
            if video_writer_global: 
                try: video_writer_global.close()
                except Exception as e: print(f"Error closing global video writer in finally: {e}")
            if 'eval_env_wrapped_final' in locals() and eval_env_wrapped_final: eval_env_wrapped_final.close()
    elif model_to_eval is None:
        print("No model available for final evaluation.")
    
    if 'env' in locals() and env: env.close()


if __name__ == "__main__":
    TOTAL_TIMESTEPS_MAIN = 1_000_000 
    EVAL_EPISODES_FINAL_VISUAL_MAIN = 2000
    MODEL_SAVE_DIR_MAIN_CONFIG = "./ppo_splix_partial_obs_checkpoints/"
    # LOAD_MODEL_PATH_MAIN_CONFIG = os.path.join(MODEL_SAVE_DIR_MAIN_CONFIG, "best_model") 
    LOAD_MODEL_PATH_MAIN_CONFIG = None
    LOAD_MODEL_FLAG_MAIN_CONFIG = True # True to skip training, False to train
    SAVE_VIDEO_PATH_MAIN_CONFIG = os.path.join(MODEL_SAVE_DIR_MAIN_CONFIG, "eval_video")
    # SAVE_VIDEO_PATH_MAIN_CONFIG = None # Disable video

    EVAL_FREQ_CALLBACK_MAIN = 50000
    N_EVAL_EPISODES_CALLBACK_MAIN = 20

    if platform.system() == "Linux" and "microsoft" in platform.release().lower():
        if os.environ.get("DISPLAY") is None:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

    args_train_eval_main = dict(
        total_timesteps=TOTAL_TIMESTEPS_MAIN,
        eval_episodes_final=EVAL_EPISODES_FINAL_VISUAL_MAIN,
        load_model_path=LOAD_MODEL_PATH_MAIN_CONFIG,
        eval_freq_callback=EVAL_FREQ_CALLBACK_MAIN,
        n_eval_episodes_callback=N_EVAL_EPISODES_CALLBACK_MAIN,
        load_model_flag=LOAD_MODEL_FLAG_MAIN_CONFIG,
        save_video_path=SAVE_VIDEO_PATH_MAIN_CONFIG
    )

    if platform.system() == "Emscripten":
        asyncio.ensure_future(train_and_evaluate(**args_train_eval_main))
    else:
        asyncio.run(train_and_evaluate(**args_train_eval_main))