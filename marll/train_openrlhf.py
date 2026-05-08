"""train.py - Medical Dialogue Training with OpenRLHF

This module provides training for medical dialogue agents using OpenRLHF framework.
OpenRLHF supports PPO, REINFORCE++, GRPO, RLOO algorithms with Ray + vLLM distributed architecture.
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for medical dialogue training."""
    model_name: str = "Qwen/Qwen2.5-7B"
    algorithm: str = "ppo"
    num_episodes: int = 1000
    max_turns: int = 15
    batch_size: int = 32
    lr: float = 1e-6
    kl_coef: float = 0.01
    device: str = "auto"


class MedicalDialogueRewardFunction:
    """
    Custom reward function for medical dialogue training.
    
    Implements reward calculation based on:
    - Information reveal (doctor)
    - Trust building (doctor)
    - Honest disclosure (patient)
    - Diagnosis accuracy
    """
    
    def __init__(self, scenario_config: Dict[str, Any]):
        self.scenario_config = scenario_config
        self.required_info_keys = scenario_config.get("required_info_keys", [])
        
        # Reward weights
        self.weights = {
            "info_reveal": 1.0,
            "sensitive_reveal": 3.0,
            "trust_gain": 0.5,
            "aggressive_penalty": -1.0,
            "diagnosis_correct": 10.0,
            "honesty_high_trust": 2.0,
            "honesty_low_trust": -0.5,
            "privacy_breach_penalty": -2.0
        }
    
    def compute_reward(
        self,
        doctor_speech: str,
        patient_speech: str,
        trust_score: float,
        phase: int,
        collected_info: Dict[str, Any],
        is_terminal: bool = False,
        ground_truth: Optional[Dict] = None
    ) -> float:
        """
        Compute reward for a single dialogue step.
        
        Args:
            doctor_speech: Doctor's current utterance
            patient_speech: Patient's current utterance
            trust_score: Current trust score (0-10)
            phase: Current consultation phase
            collected_info: Information collected so far
            is_terminal: Whether this is the final step
            ground_truth: Ground truth for diagnosis evaluation
            
        Returns:
            Combined reward signal
        """
        doctor_reward = 0.0
        patient_reward = 0.0
        
        # Doctor: Aggressive questioning penalty
        if self._is_aggressive(doctor_speech):
            doctor_reward += self.weights["aggressive_penalty"]
        
        # Doctor: Trust building bonus
        trust_effect = self._analyze_trust_effect(doctor_speech)
        doctor_reward += trust_effect * self.weights["trust_gain"]
        
        # Patient: Information reveal
        sensitive_revealed = self._check_sensitive_reveal(patient_speech)
        if sensitive_revealed:
            doctor_reward += self.weights["sensitive_reveal"]
            if trust_score > 5:
                patient_reward += self.weights["honesty_high_trust"]
            else:
                patient_reward += self.weights["honesty_low_trust"]
        
        # Doctor: General info reveal
        info_revealed = self._check_info_reveal(patient_speech)
        doctor_reward += info_revealed * self.weights["info_reveal"]
        
        # Terminal: Diagnosis evaluation
        if is_terminal and ground_truth:
            diagnosis_result = self._evaluate_diagnosis(collected_info, ground_truth)
            if diagnosis_result["is_correct"]:
                doctor_reward += self.weights["diagnosis_correct"]
            return doctor_reward + patient_reward
        
        return doctor_reward + patient_reward
    
    def _is_aggressive(self, speech: str) -> bool:
        """Check if doctor is being aggressive."""
        aggressive_markers = ["怎么", "为什么", "你必须", "不说", "撒谎", "搞清楚"]
        return any(marker in speech for marker in aggressive_markers)
    
    def _analyze_trust_effect(self, speech: str) -> float:
        """Analyze trust-building effect of doctor's speech."""
        trust_markers = [
            "理解", "不用", "不会", "保密", "帮助", "一起",
            "谢谢", "慢慢说", "没关系", "正常", "共情"
        ]
        return min(1.0, sum(1 for m in trust_markers if m in speech) * 0.3)
    
    def _check_sensitive_reveal(self, speech: str) -> bool:
        """Check if sensitive information was revealed."""
        sensitive_markers = ["吸毒", "静脉", "注射", "成瘾", "不洁", "自杀", "抑郁"]
        return any(marker in speech for marker in sensitive_markers)
    
    def _check_info_reveal(self, speech: str) -> int:
        """Check how much general information was revealed."""
        count = 0
        info_markers = {
            "发热": ["发热", "发烧", "体温"],
            "胸痛": ["胸痛", "胸口"],
            "心悸": ["心悸", "心跳"],
            "呼吸困难": ["呼吸", "气短"]
        }
        for key, markers in info_markers.items():
            if any(m in speech for m in markers):
                count += 1
        return count
    
    def _evaluate_diagnosis(
        self,
        collected_info: Dict,
        ground_truth: Dict
    ) -> Dict[str, Any]:
        """Evaluate diagnosis correctness."""
        required = ground_truth.get("required_info_keys", [])
        has_addiction = any(k in str(collected_info) for k in ["成瘾", "吸毒", "毒品"])
        has_injection = any(k in str(collected_info) for k in ["静脉", "注射"])
        
        return {
            "is_correct": has_addiction and has_injection,
            "reveal_rate": len([k for k in required if k in collected_info]) / max(1, len(required))
        }


class MedicalMultiTurnAgent:
    """
    Multi-turn agent for medical dialogue training.
    
    Implements the agent interface required by OpenRLHF for multi-turn conversations.
    """
    
    def __init__(
        self,
        scenario_config: Dict[str, Any],
        agent_type: str = "doctor"
    ):
        self.scenario_config = scenario_config
        self.agent_type = agent_type
        self.trust_score = 3.0
        self.phase = 0
        self.collected_info = {}
        
        self.reward_func = MedicalDialogueRewardFunction(scenario_config)
    
    def reset(self, states: dict) -> dict:
        """Reset agent state for new episode."""
        self.trust_score = 3.0
        self.phase = 0
        self.collected_info = {}
        
        return {
            "observation": states.get("observation", "")
        }
    
    def step(self, states: dict) -> dict:
        """
        Execute one step of the dialogue.
        
        Args:
            states: Dict containing:
                - observation_text: Current dialogue state
                - action_text: Last agent's response
                - label: Ground truth for evaluation
                
        Returns:
            Dict with rewards, done status, feedback
        """
        doctor_speech = ""
        patient_speech = ""
        
        # Parse states
        if "action_text" in states:
            if self.agent_type == "doctor":
                doctor_speech = states["action_text"]
            else:
                patient_speech = states["action_text"]
        
        # Update collected info
        if patient_speech:
            revealed = self._extract_info(patient_speech)
            self.collected_info.update(revealed)
            
            # Update trust
            trust_delta = self._calculate_trust_change(doctor_speech)
            self.trust_score = max(0.0, min(10.0, self.trust_score + trust_delta))
        
        # Compute reward
        reward = self.reward_func.compute_reward(
            doctor_speech=doctor_speech,
            patient_speech=patient_speech,
            trust_score=self.trust_score,
            phase=self.phase,
            collected_info=self.collected_info,
            is_terminal=states.get("done", False),
            ground_truth=self.scenario_config
        )
        
        # Update phase
        self.phase = min(3, len(self.collected_info) // 2)
        
        # Generate feedback for next turn
        if self.agent_type == "doctor":
            feedback = self._generate_doctor_feedback(patient_speech)
        else:
            feedback = self._generate_patient_feedback(doctor_speech)
        
        return {
            "rewards": reward,
            "scores": reward,
            "environment_feedback": feedback,
            "done": states.get("done", False),
            "extra_logs": {
                "trust_score": self.trust_score,
                "collected_info_keys": list(self.collected_info.keys()),
                "phase": self.phase
            }
        }
    
    def _extract_info(self, speech: str) -> Dict[str, Any]:
        """Extract information from speech."""
        info = {}
        if "吸毒" in speech or "毒品" in speech:
            info["成瘾史"] = True
        if "静脉" in speech or "注射" in speech:
            info["静脉注射史"] = True
        if "发热" in speech or "发烧" in speech:
            info["发热"] = True
        return info
    
    def _calculate_trust_change(self, speech: str) -> float:
        """Calculate trust score change."""
        trust_increase = ["理解", "不用", "保密", "帮助", "谢谢"]
        trust_decrease = ["怎么", "为什么", "你必须"]
        
        delta = 0.0
        for marker in trust_increase:
            if marker in speech:
                delta += 0.5
        for marker in trust_decrease:
            if marker in speech:
                delta -= 0.8
        return delta
    
    def _generate_doctor_feedback(self, patient_speech: str) -> str:
        """Generate feedback for doctor agent."""
        if not patient_speech:
            return ""
        return f"Patient said: {patient_speech[:200]}..."
    
    def _generate_patient_feedback(self, doctor_speech: str) -> str:
        """Generate feedback for patient agent."""
        if not doctor_speech:
            return ""
        return f"Doctor said: {doctor_speech[:200]}..."


def setup_scenarios() -> Dict[str, Dict]:
    """Setup training scenarios."""
    return {
        "default": {
            "clinical_symptoms": "持续性发热（38.5°C）3周，伴胸痛、心悸、活动后呼吸困难。",
            "medical_history": "否认高血压、糖尿病。无手术外伤史。",
            "psychological_state": "害怕被医生发现吸毒史，担心被警察带走。",
            "initial_chief_complaint": "工作压力太大，熬夜导致的。",
            "stigma_levels": {
                "一般症状": 2,
                "个人习惯": 5,
                "敏感行为": 9
            },
            "ground_truth_diagnosis": "感染性心内膜炎",
            "required_info_keys": ["成瘾史", "静脉注射", "发热持续时间", "心脏症状"]
        },
        "depression": {
            "clinical_symptoms": "情绪低落、失眠2个月，伴食欲减退。",
            "medical_history": "近期工作压力大，婚姻出现问题。",
            "psychological_state": "不愿意承认心理问题，担心被歧视。",
            "initial_chief_complaint": "最近睡眠不好，可能是工作太累了。",
            "stigma_levels": {
                "一般症状": 2,
                "工作压力": 3,
                "心理问题": 8
            },
            "ground_truth_diagnosis": "抑郁症",
            "required_info_keys": ["情绪问题", "失眠", "食欲", "自杀观念"]
        }
    }


def generate_openrlhf_command(
    config: TrainingConfig,
    scenario: str = "default",
    output_dir: str = "checkpoints/openrlhf"
) -> str:
    """
    Generate OpenRLHF training command.
    
    Returns the shell command to run training with OpenRLHF.
    """
    scenario_config = setup_scenarios()[scenario]
    
    # Save scenario as JSON for reward function
    scenario_path = Path(output_dir) / "scenario.json"
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scenario_path, "w") as f:
        json.dump(scenario_config, f, ensure_ascii=False, indent=2)
    
    # Generate reward function Python file
    reward_func_path = Path(output_dir) / "medical_reward_func.py"
    reward_code = f'''"""Medical Dialogue Reward Function for OpenRLHF"""
import torch
from typing import Dict, Any

def reward_func(queries, prompts, labels, **kwargs):
    """
    Custom reward function for medical dialogue training.
    
    Args:
        queries: Full text (prompt + response)
        prompts: Original prompts only
        labels: Scenario config or metadata
        
    Returns:
        dict with rewards, scores, extra_logs
    """
    batch_size = len(queries)
    rewards = torch.zeros(batch_size)
    scores = torch.zeros(batch_size)
    
    # Parse scenario config from labels
    scenario = kwargs.get("scenario", {{}})
    required_keys = scenario.get("required_info_keys", [])
    
    for i, query in enumerate(queries):
        # Simple rule-based reward
        reward = 0.0
        
        # Check for sensitive info reveal
        sensitive_markers = ["吸毒", "静脉", "注射", "成瘾", "自杀", "抑郁"]
        if any(marker in query for marker in sensitive_markers):
            reward += 3.0
        
        # Check for trust-building markers
        trust_markers = ["理解", "不用", "保密", "帮助", "谢谢"]
        for marker in trust_markers:
            if marker in query:
                reward += 0.5
        
        # Check for aggressive questioning
        aggressive_markers = ["怎么", "为什么", "你必须", "不说", "撒谎"]
        for marker in aggressive_markers:
            if marker in query:
                reward -= 1.0
        
        rewards[i] = reward
        scores[i] = min(1.0, max(-1.0, reward / 10.0))
    
    return {{
        "rewards": rewards,
        "scores": scores,
        "extra_logs": {{
            "mean_reward": rewards.mean().item(),
        }}
    }}
'''
    with open(reward_func_path, "w") as f:
        f.write(reward_code)
    
    # Build command
    cmd = f"""#!/bin/bash
# Medical Dialogue Training with OpenRLHF
# Scenario: {scenario}

# Start Ray cluster
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --dashboard-host 0.0.0.0

# PPO Training with custom reward function
ray job submit --address="http://127.0.0.1:8265" \\
    --runtime-env-json='{{"working_dir": "/openrlhf"}}' \\
    -- python3 -m openrlhf.cli.train_ppo_ray \\
    --actor.num_nodes 1 \\
    --actor.num_gpus_per_node 8 \\
    --vllm.num_engines 4 \\
    --vllm.tensor_parallel_size 2 \\
    --vllm.gpu_memory_utilization 0.5 \\
    --actor.model_name_or_path {config.model_name} \\
    --reward.remote_url {reward_func_path} \\
    --ckpt.output_dir {output_dir} \\
    --train.batch_size {config.batch_size} \\
    --rollout.batch_size 1024 \\
    --train.max_epochs 1 \\
    --prompt_max_len 1024 \\
    --generate_max_len 1024 \\
    --ds.zero_stage 3 \\
    --ds.param_dtype bf16 \\
    --actor.adam.lr {config.lr} \\
    --algo.kl.init_coef {config.kl_coef} \\
    --data.max_samples {config.num_episodes} \\
    --data.prompt_dataset {output_dir}/prompts.json \\
    --ds.packing_samples \\
    --actor.gradient_checkpointing_enable \\
    --vllm.sync_backend nccl \\
    --logger.wandb.key ${{WANDB_TOKEN:-""}}
"""
    return cmd


def prepare_training_data(
    scenario: str = "default",
    num_samples: int = 1000,
    output_dir: str = "data/openrlhf"
) -> str:
    """
    Prepare training data in OpenRLHF format.
    
    Returns path to prepared data.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate prompts
    prompts = []
    for i in range(num_samples):
        prompts.append({
            "prompt_id": f"med_dialogue_{i}",
            "context_messages": [
                {"role": "user", "content": "大夫，我最近感觉不舒服。"},
                {"role": "assistant", "content": "请描述一下您的症状。"}
            ]
        })
    
    prompts_path = Path(output_dir) / "prompts.json"
    with open(prompts_path, "w") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    
    return str(prompts_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Medical Dialogue Training with OpenRLHF")
    
    # Model config
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                        help="Model name or path")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "reinforce", "grpo", "rloo"],
                        help="RL algorithm")
    
    # Training config
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate")
    
    # Scenario
    parser.add_argument("--scenario", type=str, default="default",
                        choices=["default", "depression"],
                        help="Training scenario")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/openrlhf",
                        help="Output directory")
    parser.add_argument("--prepare_only", action="store_true",
                        help="Only prepare data without training")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_name=args.model,
        algorithm=args.algorithm,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # Prepare data
    print("Preparing training data...")
    prompts_path = prepare_training_data(
        scenario=args.scenario,
        num_samples=args.episodes,
        output_dir=os.path.join(args.output_dir, "data")
    )
    print(f"Data prepared: {prompts_path}")
    
    if args.prepare_only:
        print("Data preparation complete. Exiting.")
        return
    
    # Generate training command
    print("\nGenerating OpenRLHF training command...")
    cmd = generate_openrlhf_command(config, args.scenario, args.output_dir)
    
    cmd_path = Path(args.output_dir) / "train.sh"
    with open(cmd_path, "w") as f:
        f.write(cmd)
    os.chmod(cmd_path, 0o755)
    
    print(f"\nTraining command saved to: {cmd_path}")
    print("\nTo start training, run:")
    print(f"  bash {cmd_path}")
    print("\nOr manually submit to Ray:")
    print(f"  ray job submit --address=\"http://127.0.0.1:8265\" ...")
    
    print("\n" + "="*60)
    print("Training Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Algorithm: {config.algorithm}")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.lr}")
    print(f"  Scenario: {args.scenario}")
    print("="*60)


if __name__ == "__main__":
    main()
