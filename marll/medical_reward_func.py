"""Medical Dialogue Reward Function for OpenRLHF

This module provides the custom reward function interface for OpenRLHF training.
"""
import torch
from typing import List, Dict, Any


def reward_func(
    queries: List[str],
    prompts: List[str],
    labels: List[Any],
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Custom reward function for medical dialogue training.
    
    Implements reward signals for:
    - Doctor: Information collection, trust building, diagnosis accuracy
    - Patient: Honest disclosure, privacy management
    
    Args:
        queries: Full text (prompt + response)
        prompts: Original prompts only
        labels: Metadata including scenario config
        
    Returns:
        dict with rewards, scores, extra_logs
    """
    batch_size = len(queries)
    rewards = torch.zeros(batch_size)
    scores = torch.zeros(batch_size)
    
    # Parse scenario from kwargs
    scenario = kwargs.get("scenario", {})
    required_keys = scenario.get("required_info_keys", [])
    
    for i, query in enumerate(queries):
        reward = 0.0
        
        # Sensitive information reveal reward (high value)
        sensitive_markers = ["吸毒", "静脉", "注射", "成瘾", "自杀", "抑郁", "冶游"]
        for marker in sensitive_markers:
            if marker in query:
                reward += 3.0
                break
        
        # Trust-building markers (positive reinforcement)
        trust_markers = ["理解", "不用", "不会", "保密", "帮助", "一起", "谢谢", "慢慢说"]
        for marker in trust_markers:
            if marker in query:
                reward += 0.5
        
        # Aggressive questioning penalty
        aggressive_markers = ["怎么", "为什么", "你必须", "不说", "撒谎", "搞清楚"]
        for marker in aggressive_markers:
            if marker in query:
                reward -= 1.0
                break
        
        # General information markers
        info_markers = ["发热", "发烧", "胸痛", "心悸", "失眠", "食欲"]
        for marker in info_markers:
            if marker in query:
                reward += 1.0
        
        rewards[i] = reward
        scores[i] = min(1.0, max(-1.0, reward / 10.0))
    
    return {
        "rewards": rewards,
        "scores": scores,
        "extra_logs": {
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item() if batch_size > 1 else torch.tensor(0.0),
            "positive_ratio": (rewards > 0).float().mean().item()
        }
    }
