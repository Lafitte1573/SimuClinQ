"""marll/reward.py - Reward calculation for medical dialogue MARL"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re


@dataclass
class RewardComponents:
    """Breakdown of reward components for analysis."""
    doctor_info_reveal: float = 0.0
    doctor_sensitive_reveal: float = 0.0
    doctor_trust_gain: float = 0.0
    doctor_aggressive_penalty: float = 0.0
    doctor_diagnosis_bonus: float = 0.0
    doctor_efficiency_bonus: float = 0.0
    
    patient_honesty: float = 0.0
    patient_comfort: float = 0.0
    patient_privacy_penalty: float = 0.0
    
    episode_bonus: float = 0.0
    episode_penalty: float = 0.0


class RewardCalculator:
    """
    Reward calculator for medical dialogue MARL.
    
    Designs appropriate reward signals for both doctor and patient agents
    to encourage desired behaviors in the medical consultation setting.
    """
    
    def __init__(self, config: Optional[Dict[str, float]] = None):
        """
        Initialize reward calculator.
        
        Args:
            config: Reward configuration parameters
        """
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict[str, float]:
        """Default reward configuration."""
        return {
            # Doctor rewards
            "doctor_info_reveal": 1.0,
            "doctor_sensitive_reveal": 3.0,
            "doctor_trust_gain": 0.5,
            "doctor_aggressive_penalty": -1.0,
            "doctor_aggressive_threshold": 0.5,
            "doctor_diagnosis_correct": 10.0,
            "doctor_diagnosis_partial": 5.0,
            "doctor_efficiency_bonus": 0.2,
            
            # Patient rewards
            "patient_honesty_trust_high": 2.0,
            "patient_honesty_trust_low": -0.5,
            "patient_comfort_trust_gain": 1.0,
            "patient_privacy_breach_penalty": -2.0,
            "patient_defensive_honest": 0.5,
            "patient_unnecessary_lie_penalty": -0.5,
            
            # Episode rewards
            "episode_success_bonus": 5.0,
            "episode_timeout_penalty": -5.0,
            "episode_partial_success": 2.0
        }
    
    def compute_step_reward(
        self,
        doctor_speech: str,
        patient_speech: str,
        trust_score: float,
        phase: int,
        collected_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute reward for a single dialogue step.
        
        Args:
            doctor_speech: Doctor's current speech
            patient_speech: Patient's current speech
            trust_score: Current trust score (0-10)
            phase: Current consultation phase
            collected_info: Information collected so far
            
        Returns:
            Dict with doctor_reward and patient_reward
        """
        components = RewardCalculator._calculate_components(
            doctor_speech, patient_speech, trust_score, phase, collected_info, self.config
        )
        
        doctor_reward = (
            components.doctor_info_reveal +
            components.doctor_sensitive_reveal +
            components.doctor_trust_gain +
            components.doctor_aggressive_penalty +
            components.doctor_efficiency_bonus
        )
        
        patient_reward = (
            components.patient_honesty +
            components.patient_comfort +
            components.patient_privacy_penalty +
            components.patient_unnecessary_lie_penalty
        )
        
        return {
            "doctor_reward": doctor_reward,
            "patient_reward": patient_reward,
            "components": components
        }
    
    def compute_episode_reward(
        self,
        collected_info: Dict[str, Any],
        ground_truth: Dict[str, Any],
        dialogue_turn: int,
        max_turns: int
    ) -> Dict[str, float]:
        """
        Compute terminal reward for episode completion.
        
        Args:
            collected_info: Information collected by doctor
            ground_truth: Ground truth information
            dialogue_turn: Number of turns taken
            max_turns: Maximum allowed turns
            
        Returns:
            Dict with terminal rewards
        """
        diagnosis_result = self._evaluate_diagnosis(collected_info, ground_truth)
        
        doctor_reward = 0.0
        patient_reward = 0.0
        
        if diagnosis_result["is_correct"]:
            doctor_reward += self.config["doctor_diagnosis_correct"]
            doctor_reward += self.config["episode_success_bonus"]
            patient_reward += self.config["episode_success_bonus"]
        elif diagnosis_result["is_partial"]:
            doctor_reward += self.config["doctor_diagnosis_partial"]
            doctor_reward += self.config["episode_partial_success"]
        else:
            doctor_reward += self.config["episode_timeout_penalty"]
            patient_reward += self.config["episode_timeout_penalty"]
        
        efficiency_bonus = self.config["doctor_efficiency_bonus"] * (max_turns - dialogue_turn) / max_turns
        doctor_reward += efficiency_bonus
        
        return {
            "doctor_reward": doctor_reward,
            "patient_reward": patient_reward,
            "diagnosis_result": diagnosis_result
        }
    
    @staticmethod
    def _calculate_components(
        doctor_speech: str,
        patient_speech: str,
        trust_score: float,
        phase: int,
        collected_info: Dict[str, Any],
        config: Dict[str, float]
    ) -> RewardComponents:
        """Calculate individual reward components."""
        components = RewardComponents()
        
        components.doctor_aggressive_penalty = RewardCalculator._compute_aggressive_penalty(
            doctor_speech, config
        )
        
        if components.doctor_aggressive_penalty == 0:
            components.doctor_trust_gain = RewardCalculator._compute_trust_gain_bonus(
                doctor_speech, config
            )
        
        components.doctor_info_reveal = RewardCalculator._compute_info_reveal_bonus(
            patient_speech, collected_info, config
        )
        
        components.doctor_sensitive_reveal = RewardCalculator._compute_sensitive_reveal_bonus(
            patient_speech, config
        )
        
        sensitive_revealed = RewardCalculator._check_sensitive_info_reveal(patient_speech)
        if sensitive_revealed:
            if trust_score > 5:
                components.patient_honesty = config["patient_honesty_trust_high"]
            else:
                components.patient_honesty = config["patient_defensive_honest"]
        elif trust_score > 5:
            components.patient_honesty = config["patient_honesty_trust_low"]
        
        if components.doctor_trust_gain > 0:
            components.patient_comfort = config["patient_comfort_trust_gain"]
        
        return components
    
    @staticmethod
    def _compute_aggressive_penalty(speech: str, config: Dict[str, float]) -> float:
        """Check for aggressive questioning patterns."""
        aggressive_markers = [
            "怎么", "为什么", "你必须", "不说", "撒谎", "搞清楚",
            "你懂不懂", "你不知道", "你难道", "我问你"
        ]
        
        count = sum(1 for marker in aggressive_markers if marker in speech)
        threshold = config.get("doctor_aggressive_threshold", 0.5)
        
        if count >= 2:
            return config["doctor_aggressive_penalty"] * (count - 1)
        return 0.0
    
    @staticmethod
    def _compute_trust_gain_bonus(speech: str, config: Dict[str, float]) -> float:
        """Compute bonus for trust-building speech."""
        trust_markers = [
            "理解", "不用", "不会", "保密", "帮助", "一起", 
            "谢谢", "慢慢说", "没关系", "正常", "常见",
            "共情", "支持", "耐心", "关心"
        ]
        
        count = sum(1 for marker in trust_markers if marker in speech)
        return config["doctor_trust_gain"] * min(count, 3)
    
    @staticmethod
    def _compute_info_reveal_bonus(
        patient_speech: str,
        collected_info: Dict[str, Any],
        config: Dict[str, float]
    ) -> float:
        """Compute bonus for information reveal."""
        info_markers = {
            "发热": r"发热|发烧|体温",
            "胸痛": r"胸痛|胸口",
            "心悸": r"心悸|心跳",
            "呼吸困难": r"呼吸|气短|喘",
            "吸烟": r"吸烟|抽烟|烟",
            "饮酒": r"饮酒|喝酒|酒"
        }
        
        bonus = 0.0
        for info_key, pattern in info_markers.items():
            if re.search(pattern, patient_speech) and info_key not in collected_info:
                bonus += config["doctor_info_reveal"]
        
        return bonus
    
    @staticmethod
    def _compute_sensitive_reveal_bonus(patient_speech: str, config: Dict[str, float]) -> float:
        """Compute bonus for revealing sensitive information."""
        sensitive_markers = ["吸毒", "静脉", "注射", "成瘾", "不洁", "毒品"]
        
        if any(marker in patient_speech for marker in sensitive_markers):
            return config["doctor_sensitive_reveal"]
        return 0.0
    
    @staticmethod
    def _check_sensitive_info_reveal(speech: str) -> bool:
        """Check if sensitive information was revealed."""
        sensitive_markers = ["吸毒", "静脉", "注射", "成瘾", "不洁", "毒品"]
        return any(marker in speech for marker in sensitive_markers)
    
    def _evaluate_diagnosis(
        self,
        collected_info: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate the correctness of diagnosis."""
        has_addiction = any(kw in str(collected_info) for kw in ["成瘾", "吸毒", "毒品"])
        has_injection = any(kw in str(collected_info) for kw in ["静脉", "注射"])
        
        is_correct = has_addiction and has_injection
        is_partial = bool(collected_info) and not is_correct
        
        return {
            "is_correct": is_correct,
            "is_partial": is_partial,
            "collected_keys": list(collected_info.keys())
        }
    
    def compute_credit_assignment(
        self,
        episode_rewards: List[Dict[str, float]],
        discount_factor: float = 0.99
    ) -> List[Dict[str, float]]:
        """
        Compute discounted returns for credit assignment.
        
        Args:
            episode_rewards: List of step rewards
            discount_factor: Discount factor for future rewards
            
        Returns:
            List of discounted returns for each step
        """
        returns = []
        running_return = 0.0
        
        for reward in reversed(episode_rewards):
            running_return = reward["doctor_reward"] + discount_factor * running_return
            returns.insert(0, {"doctor_return": running_return})
            
            running_return_patient = reward["patient_reward"] + discount_factor * running_return
            returns[0]["patient_return"] = running_return_patient
        
        return returns


class TrustModel:
    """
    Model for tracking and updating patient trust dynamically.
    
    Based on Goffman's stigma theory and interpersonal trust models.
    """
    
    def __init__(self, initial_trust: float = 3.0, max_trust: float = 10.0):
        self.trust_level = initial_trust
        self.max_trust = max_trust
        self.trust_history = [initial_trust]
        
        self.trust_thresholds = {
            "disclosure_easy": 5.0,
            "disclosure_moderate": 7.0,
            "disclosure_hard": 9.0
        }
    
    def update(self, doctor_speech: str, patient_speech: str, action_type: str = "neutral"):
        """
        Update trust based on dialogue.
        
        Args:
            doctor_speech: Doctor's speech content
            patient_speech: Patient's speech content
            action_type: Type of patient action (honest, defensive, deceptive)
        """
        trust_delta = self._calculate_delta(doctor_speech, patient_speech, action_type)
        self.trust_level = max(0.0, min(self.max_trust, self.trust_level + trust_delta))
        self.trust_history.append(self.trust_level)
    
    def _calculate_delta(
        self,
        doctor_speech: str,
        patient_speech: str,
        action_type: str
    ) -> float:
        """Calculate trust change."""
        delta = 0.0
        
        trust_building_markers = ["理解", "不用担心", "保密", "帮助您", "谢谢", "慢慢说"]
        trust_damaging_markers = ["怎么", "为什么", "你必须", "搞清楚", "撒谎"]
        
        for marker in trust_building_markers:
            if marker in doctor_speech:
                delta += 0.5
        
        for marker in trust_damaging_markers:
            if marker in doctor_speech:
                delta -= 0.8
        
        if action_type == "honest" and self.trust_level > 5:
            delta += 0.3
        
        if action_type == "deceptive" and self.trust_level < 5:
            delta -= 0.2
        
        return delta
    
    def should_reveal(self, stigma_level: float) -> bool:
        """Determine if patient should reveal sensitive information."""
        return self.trust_level > stigma_level
    
    def get_disclosure_probability(self, stigma_level: float) -> float:
        """Get probability of disclosure based on trust and stigma."""
        gap = self.trust_level - stigma_level
        
        if gap >= 3:
            return 0.9
        elif gap >= 0:
            return 0.5 + (gap / 3) * 0.4
        elif gap >= -3:
            return 0.1 + ((gap + 3) / 6) * 0.4
        else:
            return 0.1