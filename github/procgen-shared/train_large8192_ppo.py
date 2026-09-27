"""Shared PPO B8192, fixed Adam LR3e-4, full15M, no PopArt or entropy."""
from shared_procgen.ppo_training import main

if __name__ == '__main__':
    main('large8192_ppo')
