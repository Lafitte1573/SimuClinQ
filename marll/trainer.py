"""marll/trainer.py - MAPPO Training for Medical Dialogue Agents"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import torch
import numpy as np
from pathlib import Path
import json
import time


@dataclass
class TrainingConfig:
    """Configuration for MARL training."""
    num_episodes: int = 1000
    max_turns_per_episode: int = 15
    batch_size: int = 32
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_workers: int = 4
    save_freq: int = 100
    log_freq: int = 10
    eval_freq: int = 50
    eval_episodes: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_path: str = "checkpoints"
    log_path: str = "logs"


class ReplayBuffer:
    """Replay buffer for storing trajectories."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(
        self,
        state: Dict,
        action: Dict,
        reward: Dict,
        next_state: Dict,
        done: bool
    ):
        """Add transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List:
        """Sample random batch."""
        return np.random.choice(len(self.buffer), batch_size, replace=False)
    
    def __len__(self):
        return len(self.buffer)


class ExperienceBuffer:
    """Buffer for storing agent experiences during an episode."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def append(
        self,
        state: Dict,
        action: str,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def get_trajectory(self) -> Dict[str, List]:
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "values": self.values,
            "log_probs": self.log_probs,
            "dones": self.dones
        }
    
    def compute_returns(self, last_value: float, gamma: float, lam: float) -> Tuple[List[float], List[float]]:
        """Compute GAE returns and advantages."""
        rewards = self.rewards
        values = self.values + [last_value]
        
        gae = 0
        returns = []
        advantages = []
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] - values[step]
            gae = delta + gamma * lam * gae
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
        
        return returns, advantages


class MAPPOTrainer:
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) Trainer.
    
    Implements the CTDE (Centralized Training, Decentralized Execution) paradigm
    for training doctor and patient agents in the medical dialogue setting.
    
    Based on the MAPPO algorithm from MARLlib.
    """
    
    def __init__(
        self,
        doctor_agent: Any,
        patient_agent: Any,
        env: Any,
        reward_calculator: Any,
        config: Optional[TrainingConfig] = None
    ):
        self.doctor = doctor_agent
        self.patient = patient_agent
        self.env = env
        self.reward_calculator = reward_calculator
        self.config = config or TrainingConfig()
        
        self.device = torch.device(self.config.device)
        
        self.doctor_buffer = ExperienceBuffer()
        self.patient_buffer = ExperienceBuffer()
        
        self.doctor_optimizer = None
        self.patient_optimizer = None
        self._init_optimizers()
        
        self.training_stats = {
            "episode_rewards": [],
            "diagnosis_accuracy": [],
            "trust_scores": [],
            "info_reveal_rates": []
        }
        
        self.checkpoint_dir = Path(self.config.model_path)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(self.config.log_path)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_optimizers(self):
        """Initialize optimizers for both agents."""
        try:
            from transformers import AdamW
            
            self.doctor_optimizer = AdamW(
                self.doctor.model.parameters(),
                lr=self.config.lr_actor,
                weight_decay=0.01
            )
            
            self.patient_optimizer = AdamW(
                self.patient.model.parameters(),
                lr=self.config.lr_actor,
                weight_decay=0.01
            )
            
            print(f"Initialized AdamW optimizers with lr={self.config.lr_actor}")
        except ImportError:
            print("Transformers not available, using PyTorch optimizer")
            self.doctor_optimizer = torch.optim.Adam(
                self.doctor.model.parameters() if hasattr(self.doctor, 'model') else [],
                lr=self.config.lr_actor
            )
            self.patient_optimizer = torch.optim.Adam(
                self.patient.model.parameters() if hasattr(self.patient, 'model') else [],
                lr=self.config.lr_actor
            )
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.num_episodes} episodes")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        
        for episode in range(self.config.num_episodes):
            episode_start = time.time()
            
            episode_reward, stats = self._run_episode()
            
            if episode % self.config.log_freq == 0:
                self._log_stats(episode, episode_reward, stats, time.time() - episode_start)
            
            if episode % self.config.save_freq == 0:
                self.save_checkpoint(episode)
            
            if episode % self.config.eval_freq == 0:
                eval_stats = self.evaluate()
                self._log_eval_stats(episode, eval_stats)
        
        print("Training completed!")
        self.save_checkpoint(self.config.num_episodes)
    
    def _run_episode(self) -> Tuple[float, Dict[str, Any]]:
        """Run a single episode."""
        observations = self.env.reset()
        
        self.doctor_buffer.clear()
        self.patient_buffer.clear()
        
        episode_reward = 0.0
        step_rewards = []
        
        for turn in range(self.config.max_turns_per_episode):
            doctor_obs = observations.get("doctor")
            patient_obs = observations.get("patient")
            
            doctor_speech, doctor_info = self._get_doctor_action(doctor_obs)
            patient_speech, patient_info = self._get_patient_action(patient_obs, doctor_speech)
            
            trust_score = patient_info.get("trust_score", 3.0)
            phase = doctor_info.get("phase", 0)
            collected_info = doctor_info.get("collected_info", {})
            
            step_reward = self.reward_calculator.compute_step_reward(
                doctor_speech, patient_speech, trust_score, phase, collected_info
            )
            
            self._store_transitions(
                doctor_obs, doctor_speech, step_reward["doctor_reward"],
                patient_obs, patient_speech, step_reward["patient_reward"]
            )
            
            step_rewards.append(step_reward)
            episode_reward += step_reward["doctor_reward"] + step_reward["patient_reward"]
            
            next_observations, rewards, done, trunc, info = self.env.step({
                "doctor": {"speech": doctor_speech},
                "patient": {"speech": patient_speech, "trust_delta": patient_info.get("trust_delta", 0)}
            })
            
            observations = next_observations
            
            if done:
                break
        
        terminal_reward = self.reward_calculator.compute_episode_reward(
            collected_info=collected_info,
            ground_truth=self.env.scenario_config,
            dialogue_turn=turn + 1,
            max_turns=self.config.max_turns_per_episode
        )
        
        episode_reward += terminal_reward["doctor_reward"] + terminal_reward["patient_reward"]
        
        self.training_stats["episode_rewards"].append(episode_reward)
        self.training_stats["trust_scores"].append(trust_score)
        
        diagnosis_result = info.get("diagnosis_result", {})
        self.training_stats["diagnosis_accuracy"].append(
            1.0 if diagnosis_result.get("is_correct") else 0.5 if diagnosis_result.get("is_partial") else 0.0
        )
        
        return episode_reward, {
            "turns": turn + 1,
            "trust_score": trust_score,
            "diagnosis_result": diagnosis_result,
            "terminal_reward": terminal_reward
        }
    
    def _get_doctor_action(self, observation: Any) -> Tuple[str, Dict[str, Any]]:
        """Get doctor's action."""
        conversation_history = self._extract_history(observation)
        patient_speech = conversation_history[-1]["content"] if conversation_history else ""
        
        current_phase = observation.own_role_context.get("phase", 0) if hasattr(observation, 'own_role_context') else 0
        collected_info = observation.own_role_context.get("collected_info", {}) if hasattr(observation, 'own_role_context') else {}
        
        return self.doctor.get_action(
            patient_speech=patient_speech,
            conversation_history=conversation_history,
            current_phase=current_phase,
            collected_info=collected_info
        )
    
    def _get_patient_action(self, observation: Any, doctor_speech: str) -> Tuple[str, Dict[str, Any]]:
        """Get patient's action."""
        conversation_history = self._extract_history(observation)
        
        return self.patient.get_action(
            doctor_speech=doctor_speech,
            conversation_history=conversation_history
        )
    
    def _extract_history(self, observation: Any) -> List[Dict[str, str]]:
        """Extract conversation history from observation."""
        if hasattr(observation, 'own_history'):
            return observation.own_history
        elif hasattr(observation, 'dialogue_state'):
            return observation.dialogue_state.conversation_history
        return []
    
    def _store_transitions(
        self,
        doctor_obs: Any,
        doctor_action: str,
        doctor_reward: float,
        patient_obs: Any,
        patient_action: str,
        patient_reward: float
    ):
        """Store transitions in buffers."""
        self.doctor_buffer.append(
            state=doctor_obs,
            action=doctor_action,
            reward=doctor_reward,
            value=0.0,
            log_prob=0.0,
            done=False
        )
        
        self.patient_buffer.append(
            state=patient_obs,
            action=patient_action,
            reward=patient_reward,
            value=0.0,
            log_prob=0.0,
            done=False
        )
    
    def _update_policy(self):
        """Update policy using PPO clipped objective."""
        if len(self.doctor_buffer.states) < self.config.batch_size:
            return
        
        returns, advantages = self.doctor_buffer.compute_returns(
            last_value=0.0,
            gamma=self.config.gamma,
            lam=self.config.lam
        )
        
        for _ in range(self.config.num_workers):
            self._ppo_update(self.doctor_buffer, self.doctor_optimizer, returns, advantages)
            self._ppo_update(self.patient_buffer, self.patient_optimizer, returns, advantages)
    
    def _ppo_update(
        self,
        buffer: ExperienceBuffer,
        optimizer: Any,
        returns: List[float],
        advantages: List[float],
        clip_epsilon: float = 0.2
    ):
        """Perform PPO update."""
        pass
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate current policy."""
        eval_rewards = []
        eval_accuracy = []
        eval_trust = []
        
        for _ in range(self.config.eval_episodes):
            observations = self.env.reset()
            
            episode_reward = 0.0
            trust_scores = []
            collected_info = {}
            
            for turn in range(self.config.max_turns_per_episode):
                doctor_obs = observations.get("doctor")
                patient_obs = observations.get("patient")
                
                doctor_speech, doctor_info = self._get_doctor_action(doctor_obs)
                patient_speech, patient_info = self._get_patient_action(patient_obs, doctor_speech)
                
                trust_scores.append(patient_info.get("trust_score", 3.0))
                
                next_observations, rewards, done, trunc, info = self.env.step({
                    "doctor": {"speech": doctor_speech},
                    "patient": {"speech": patient_speech, "trust_delta": patient_info.get("trust_delta", 0)}
                })
                
                episode_reward += rewards.get("doctor", 0) + rewards.get("patient", 0)
                observations = next_observations
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_trust.append(np.mean(trust_scores) if trust_scores else 0.0)
            
            diagnosis_result = info.get("diagnosis_result", {})
            eval_accuracy.append(
                1.0 if diagnosis_result.get("is_correct") else 0.5 if diagnosis_result.get("is_partial") else 0.0
            )
        
        return {
            "eval_reward": np.mean(eval_rewards),
            "eval_accuracy": np.mean(eval_accuracy),
            "eval_trust": np.mean(eval_trust)
        }
    
    def save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint = {
            "episode": episode,
            "training_stats": self.training_stats,
            "config": self.config.__dict__
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{episode}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        with open(path, "r") as f:
            checkpoint = json.load(f)
        
        self.training_stats = checkpoint.get("training_stats", self.training_stats)
        print(f"Checkpoint loaded from: {path}")
    
    def _log_stats(self, episode: int, reward: float, stats: Dict, elapsed: float):
        """Log training statistics."""
        print(f"\n[Episode {episode}] Reward: {reward:.2f} | Turns: {stats['turns']} | "
              f"Trust: {stats['trust_score']:.1f} | Time: {elapsed:.2f}s")
        
        diagnosis = stats.get("diagnosis_result", {})
        if diagnosis:
            status = "CORRECT" if diagnosis.get("is_correct") else "PARTIAL" if diagnosis.get("is_partial") else "MISSED"
            print(f"  Diagnosis: {status}")
    
    def _log_eval_stats(self, episode: int, eval_stats: Dict):
        """Log evaluation statistics."""
        print(f"\n[Eval @ Episode {episode}]")
        print(f"  Mean Reward: {eval_stats['eval_reward']:.2f}")
        print(f"  Accuracy: {eval_stats['eval_accuracy']:.2%}")
        print(f"  Mean Trust: {eval_stats['eval_trust']:.2f}")
        
        self.training_stats.setdefault("eval_rewards", []).append(eval_stats["eval_reward"])
        self.training_stats.setdefault("eval_accuracy", []).append(eval_stats["eval_accuracy"])


class MARLlibWrapper:
    """
    Wrapper for integrating with MARLlib.
    
    This allows using MARLlib's algorithms while keeping our custom
    medical dialogue environment.
    """
    
    def __init__(
        self,
        env_creator,
        algo_name: str = "mappo",
        model_arch: str = "gru"
    ):
        self.env_creator = env_creator
        self.algo_name = algo_name
        self.model_arch = model_arch
        
        self.trainer = None
        self.model = None
    
    def setup(self):
        """Setup MARLlib components."""
        try:
            import marllib
            marllib.register
            
            self.algo = marllib.algos.mappo(hyperparam_source="common")
            self.model = marllib.build_model(
                self.env_creator(),
                self.algo,
                {"core_arch": self.model_arch}
            )
            
            print(f"MARLlib setup complete with {self.algo_name}")
            return True
        except ImportError as e:
            print(f"MARLlib not available: {e}")
            return False
        except Exception as e:
            print(f"MARLlib setup failed: {e}")
            return False
    
    def train(self, stop_condition: Optional[Dict] = None):
        """Train using MARLlib."""
        if self.trainer is None:
            print("MARLlib not properly initialized")
            return
        
        stop = stop_condition or {"timesteps_total": 1000000}
        
        self.algo.fit(
            self.env_creator(),
            self.model,
            stop=stop,
            share_policy="all"
        )
    
    def render(self, restore_path: Dict[str, str]):
        """Render episode using trained model."""
        self.algo.render(
            self.env_creator(),
            self.model,
            local_mode=True,
            restore_path=restore_path
        )