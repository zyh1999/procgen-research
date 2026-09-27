"""Shared PPO B512, fixed Adam LR3e-4, full6M, no PopArt or entropy."""
from shared_procgen.ppo_training import main

if __name__ == '__main__':
    main('normal_ppo')
