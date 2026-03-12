python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-G1-Inspire-v0

python scripts/reinforcement_learning/rsl_rl/play.py --num_envs=1 --task=Isaac-Velocity-Flat-G1-Inspire-v0 --checkpoint=logs/rsl_rl/g1_flat/2026-02-02_17-19-26/model_1499.pt

I'll break down exactly how the play.py script works when evaluating a trained policy:                                                          
  
  Complete Workflow of play.py                                                                                                                    
                 
  1. Argument Parsing & Setup (Lines 10-52)

  python scripts/reinforcement_learning/rsl_rl/play.py \
    --num_envs=1 \
    --task=Isaac-Velocity-Flat-G1-Inspire-v0 \
    --checkpoint=logs/rsl_rl/g1_flat/2026-02-02_17-19-26/model_1499.pt

  What happens:
  - Parses CLI arguments including --num_envs, --task, --checkpoint
  - Additional available args: --video, --video_length, --real-time, --use_pretrained_checkpoint
  - Separates Hydra args from custom args (line 42)
  - Launches Isaac Sim application via AppLauncher (line 51)

  2. Hydra Configuration Loading (Line 83)

  @hydra_task_config(args_cli.task, args_cli.agent)
  def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):

  What happens:
  - Hydra decorator loads two configuration files:
    a. Environment config from task name: Isaac-Velocity-Flat-G1-Inspire-v0
        - Located in isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py
      - Contains: scene setup, terrain, robot config, observation/action spaces, rewards, terminations
    b. Agent config: Default rsl_rl_cfg_entry_point
        - Located in isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py
      - Contains: PPO hyperparameters, network architecture, training settings

  3. Configuration Override (Lines 87-97)

  What happens:
  - Extracts task name: g1_flat from Isaac-Velocity-Flat-G1-Inspire-v0
  - Overrides num_envs to 1 (from CLI)
  - Sets environment seed from agent config
  - Configures device (GPU/CPU)

  4. Checkpoint Path Resolution (Lines 100-111)

  What happens:
  - Sets log directory: logs/rsl_rl/<experiment_name>/
  - Three checkpoint loading modes:
    a. Pretrained (--use_pretrained_checkpoint): Downloads from Nucleus
    b. Explicit path (--checkpoint): Uses your specified path ✓ This is your case
    c. Auto-discovery: Finds latest checkpoint in log directory

  Your command loads: logs/rsl_rl/g1_flat/2026-02-02_17-19-26/model_1499.pt

  5. Environment Creation (Lines 118-138)

  What happens:
  env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

  - Creates Gymnasium environment for Isaac-Velocity-Flat-G1-Inspire-v0
  - Spawns 1 parallel environment (from --num_envs=1)
  - Each environment contains:
    - G1 robot with Inspire hands (53 DOF)
    - Flat terrain
    - Physics simulation at configured timestep
  - Wraps with video recorder if --video flag set
  - Wraps with RslRlVecEnvWrapper for RSL-RL compatibility (line 138)

  6. Model Loading (Lines 142-148)

  What happens:
  if agent_cfg.class_name == "OnPolicyRunner":
      runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
  runner.load(resume_path)

  - Creates RSL-RL runner (PPO trainer object)
  - Loads checkpoint weights from model_1499.pt into the policy network
  - Checkpoint contains:
    - Actor-critic network weights
    - Observation normalizer (running mean/std statistics)
    - Training iteration metadata

  7. Policy Extraction (Lines 151-168)

  What happens:
  policy = runner.get_inference_policy(device=env.unwrapped.device)
  policy_nn = runner.alg.policy  # or runner.alg.actor_critic for older versions
  normalizer = policy_nn.actor_obs_normalizer  # or student_obs_normalizer

  - Extracts inference-ready policy (actor network only, no critic)
  - Gets observation normalizer for preprocessing inputs
  - Policy network architecture (from training config):
    - Input: 171-dim observation vector (base lin/ang vel, projected gravity, commands, joint pos/vel, previous actions)
    - Hidden: Typically [512, 256, 128] with ELU activation
    - Output: 53-dim action vector (joint position targets for G1+Inspire)

  8. Policy Export (Lines 171-173)

  What happens:
  export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
  export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
  export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

  - Automatically exports the policy to deployment formats:
    - JIT (TorchScript): exported/policy.pt - for C++ deployment or ROS nodes
    - ONNX: exported/policy.onnx - for cross-platform deployment
  - These exports include the normalizer baked in
  - Saves to: logs/rsl_rl/g1_flat/2026-02-02_17-19-26/exported/

  9. Inference Loop (Lines 175-198)

  What happens:
  obs = env.get_observations()
  while simulation_app.is_running():
      with torch.inference_mode():
          actions = policy(obs)          # Neural network forward pass
          obs, _, _, _ = env.step(actions)  # Physics simulation step

      if args_cli.real_time and sleep_time > 0:
          time.sleep(sleep_time)

  Each iteration:
  1. Get observations (171-dim vector):
    - Base linear/angular velocity from IMU
    - Gravity vector in base frame
    - Velocity commands (lin_vel_x, lin_vel_y, ang_vel_z)
    - Joint positions and velocities (53 joints)
    - Previous actions
  2. Normalize observations (via normalizer in policy)
  3. Forward pass through neural network → outputs 53-dim action vector
  4. Apply actions to robot:
    - Actions are joint position targets (PD control at lower level)
    - Physics simulation steps forward
    - Robot executes motion
  5. Render in Isaac Sim viewport (you can see the robot moving)
  6. Real-time mode (optional): Sleeps to match real-world timing
  7. Video recording (if --video flag): Saves frames for specified length

  10. Cleanup (Line 201)

  env.close()
  simulation_app.close()

  - Shuts down Isaac Sim
  - Closes GPU resources

  ---
  Key Differences from Training

  ┌────────────────┬────────────────────────────────┬───────────────────────────────┐
  │     Aspect     │      Training (train.py)       │        Play (play.py)         │
  ├────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ Environments   │ Many parallel (default 4096)   │ Few (you used 1)              │
  ├────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ Policy updates │ Gradient updates every N steps │ No updates (frozen weights)   │
  ├────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ Exploration    │ Adds noise to actions          │ Deterministic (no noise)      │
  ├────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ Terminations   │ Resets on failure              │ Resets on failure             │
  ├────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ Rewards        │ Computed and logged            │ Ignored                       │
  ├────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ Speed          │ Fast (headless)                │ Real-time option available    │
  ├────────────────┼────────────────────────────────┼───────────────────────────────┤
  │ Output         │ Checkpoints, logs, tensorboard │ Visualization, video, exports │
  └────────────────┴────────────────────────────────┴───────────────────────────────┘

  ---
  What You See When Running

  When you run your command, you'll see:
  1. Isaac Sim window opens with 1 G1 robot
  2. Robot executes the learned locomotion policy
  3. Terminal prints:
    - Loading checkpoint from model_1499.pt
    - Exported policy.pt and policy.onnx paths
  4. Robot behavior: Walks based on velocity commands in the environment
  5. Loop continues until you close the window

  The exported models in exported/ are ready to deploy to ROS via the g1_fullbody_controller.py node!