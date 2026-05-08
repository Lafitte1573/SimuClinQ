"""marll/environment.py - Medical Dialogue Multi-Agent RL Environment

Following the PettingZoo API design for multi-agent environments.
"""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from enum import Enum


class AgentType(Enum):
    DOCTOR = "doctor"
    PATIENT = "patient"


@dataclass
class DialogueState:
    """State of the medical dialogue conversation."""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    patient_revealed_info: Dict[str, Any] = field(default_factory=dict)
    patient_ground_truth: Dict[str, Any] = field(default_factory=dict)
    doctor_collected_info: Dict[str, Any] = field(default_factory=dict)
    patient_trust_score: float = 3.0
    patient_stigma_levels: Dict[str, float] = field(default_factory=dict)
    current_phase: int = 0
    dialogue_turn: int = 0
    max_turns: int = 15
    scenario_id: str = ""


@dataclass
class DialogueAction:
    """Action taken by an agent in the dialogue."""
    agent_type: AgentType
    speech: str
    internal_thought: Optional[str] = None
    trust_delta: float = 0.0


@dataclass
class DialogueObservation:
    """Observation received by an agent."""
    agent_type: AgentType
    dialogue_state: DialogueState
    own_history: List[Dict[str, str]]
    opponent_last_message: Optional[str] = None
    own_role_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueReward:
    """Reward structure for the medical dialogue MARL setting."""
    doctor_reward: float = 0.0
    patient_reward: float = 0.0
    
    doctor_info_reveal_bonus: float = 0.0
    doctor_trust_building_bonus: float = 0.0
    doctor_diagnosis_bonus: float = 0.0
    doctor_efficiency_bonus: float = 0.0
    
    patient_comfort_penalty: float = 0.0
    patient_honesty_bonus: float = 0.0
    patient_privacy_bonus: float = 0.0
    
    episode_termination: bool = False
    termination_reason: str = ""


class MedicalDialogueEnv:
    """
    Multi-agent environment for medical dialogue simulation.
    
    This environment follows the PettingZoo AEC (Agent Environment Cycle) API design,
    allowing for both cooperative and competitive learning dynamics between doctor and patient agents.
    
    Agents:
        - doctor: Tries to collect enough information to make accurate diagnosis
        - patient: Balances between revealing true information and maintaining privacy
    
    Observation Space:
        - Each agent receives their own dialogue history and context
        - Doctor sees patient's revealed info; Patient sees their ground truth state
    
    Action Space:
        - Text-based dialogue utterances
        - Discrete action space represented by text generation
    
    Reward Design:
        - Doctor: Positive for info collection, trust building; Negative for aggressive questioning
        - Patient: Positive for honesty when trust is high; Negative for unnecessary deception
    """
    
    def __init__(
        self,
        scenario_config: Optional[Dict[str, Any]] = None,
        max_turns: int = 15,
        info_reveal_threshold: float = 0.7,
        reward_config: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the medical dialogue environment.
        
        Args:
            scenario_config: Configuration for the patient scenario (symptoms, history, etc.)
            max_turns: Maximum number of dialogue turns
            info_reveal_threshold: Trust threshold for patient to reveal sensitive info
            reward_config: Custom reward function parameters
        """
        self.scenario_config = scenario_config or self._default_scenario()
        self.max_turns = max_turns
        self.info_reveal_threshold = info_reveal_threshold
        
        self._reward_config = reward_config or self._default_reward_config()
        
        self.possible_agents = [AgentType.DOCTOR.value, AgentType.PATIENT.value]
        self.agents = []
        
        self.state = None
        self._reset()
    
    def _default_scenario(self) -> Dict[str, Any]:
        """Default patient scenario: IV drug user with potential endocarditis."""
        return {
            "clinical_symptoms": "持续性发热（38.5°C）3周，伴胸痛、心悸、活动后呼吸困难。查体：心脏杂音，杆状指。",
            "medical_history": "身体健康，否认高血压、糖尿病。无手术外伤史。否认不洁饮食史。",
            "psychological_state": "害怕被医生发现吸毒史，担心被警察带走或被家人知道。内心非常焦虑和恐惧。",
            "initial_chief_complaint": "最近工作压力太大，经常加班熬夜，感觉胸闷心悸，休息不好导致的。",
            "stigma_levels": {
                "一般症状（感冒、发烧等）": 2,
                "个人习惯（吸烟、饮酒等）": 5,
                "敏感行为（吸毒、不洁接触史等）": 9
            },
            "ground_truth_diagnosis": "感染性心内膜炎（静脉注射药物使用史）",
            "required_info_keys": ["成瘾史", "静脉注射", "发热持续时间", "心脏症状"]
        }
    
    def _default_reward_config(self) -> Dict[str, float]:
        """Default reward configuration."""
        return {
            "doctor_info_reveal": 1.0,
            "doctor_sensitive_reveal": 3.0,
            "doctor_trust_gain": 0.5,
            "doctor_aggressive_penalty": -1.0,
            "doctor_diagnosis_correct": 10.0,
            "doctor_diagnosis_partial": 5.0,
            "doctor_efficiency_bonus": 0.2,
            
            "patient_honesty_trust_high": 2.0,
            "patient_honesty_trust_low": -0.5,
            "patient_comfort_trust_gain": 1.0,
            "patient_privacy_breach_penalty": -2.0,
            "patient_defensive_honest": 0.5,
            
            "episode_timeout_penalty": -5.0,
            "episode_success_bonus": 5.0
        }
    
    @property
    def agent_names(self) -> List[str]:
        """Return list of agent names."""
        return self.possible_agents
    
    def reset(self) -> Dict[str, DialogueObservation]:
        """
        Reset the environment to initial state.
        
        Returns:
            Dict mapping agent names to their initial observations
        """
        self.agents = [AgentType.DOCTOR.value, AgentType.PATIENT.value]
        
        self.state = DialogueState(
            conversation_history=[],
            patient_revealed_info={},
            patient_ground_truth={
                "symptoms": self.scenario_config["clinical_symptoms"],
                "medical_history": self.scenario_config["medical_history"],
                "psychological_state": self.scenario_config["psychological_state"]
            },
            doctor_collected_info={},
            patient_trust_score=3.0,
            patient_stigma_levels=self.scenario_config["stigma_levels"].copy(),
            current_phase=0,
            dialogue_turn=0,
            max_turns=self.max_turns,
            scenario_id=self._generate_scenario_id()
        )
        
        return self._get_observations()
    
    def step(self, action_dict: Dict[str, DialogueAction]) -> Tuple[
        Dict[str, DialogueObservation],
        Dict[str, float],
        bool,
        bool,
        Dict[str, Any]
    ]:
        """
        Execute one step of the dialogue.
        
        Args:
            action_dict: Dict mapping agent names to their DialogueAction
            
        Returns:
            observations, rewards, done, truncated, info
        """
        if not self.agents:
            raise RuntimeError("Environment not reset or already terminated")
        
        doctor_action = action_dict.get(AgentType.DOCTOR.value)
        patient_action = action_dict.get(AgentType.PATIENT.value)
        
        rewards = {AgentType.DOCTOR.value: 0.0, AgentType.PATIENT.value: 0.0}
        info = {}
        
        turn_reward = self._compute_step_reward(doctor_action, patient_action)
        rewards[AgentType.DOCTOR.value] = turn_reward.doctor_reward
        rewards[AgentType.PATIENT.value] = turn_reward.patient_reward
        
        self.state.dialogue_turn += 1
        
        if doctor_action:
            self.state.conversation_history.append({
                "agent": AgentType.DOCTOR.value,
                "content": doctor_action.speech,
                "turn": self.state.dialogue_turn
            })
        
        if patient_action:
            self.state.conversation_history.append({
                "agent": AgentType.PATIENT.value,
                "content": patient_action.speech,
                "turn": self.state.dialogue_turn
            })
            if patient_action.trust_delta != 0:
                self.state.patient_trust_score = max(0.0, min(10.0, 
                    self.state.patient_trust_score + patient_action.trust_delta))
        
        self._update_doctor_collected_info(doctor_action, patient_action)
        self._update_patient_revealed_info(patient_action)
        
        terminate, trunc, term_reason = self._check_termination()
        
        if terminate:
            final_reward = self._compute_episode_terminal_reward()
            rewards[AgentType.DOCTOR.value] += final_reward.doctor_reward
            rewards[AgentType.PATIENT.value] += final_reward.patient_reward
            
            info["termination_reason"] = term_reason
            info["final_diagnosis"] = self._evaluate_diagnosis()
            info["info_reveal_rate"] = self._compute_info_reveal_rate()
        
        observations = self._get_observations() if not terminate else None
        
        return observations, rewards, terminate, trunc, info
    
    def _get_observations(self) -> Dict[str, DialogueObservation]:
        """Get observations for all agents."""
        doctor_obs = DialogueObservation(
            agent_type=AgentType.DOCTOR,
            dialogue_state=self.state,
            own_history=[msg for msg in self.state.conversation_history 
                        if msg["agent"] == AgentType.DOCTOR.value],
            opponent_last_message=self.state.conversation_history[-1]["content"] 
                if self.state.conversation_history else None,
            own_role_context={
                "phase": self.state.current_phase,
                "collected_info": self.state.doctor_collected_info,
                "missing_keys": self._get_missing_info_keys()
            }
        )
        
        patient_obs = DialogueObservation(
            agent_type=AgentType.PATIENT,
            dialogue_state=self.state,
            own_history=[msg for msg in self.state.conversation_history 
                        if msg["agent"] == AgentType.PATIENT.value],
            opponent_last_message=self.state.conversation_history[-1]["content"] 
                if self.state.conversation_history else None,
            own_role_context={
                "trust_score": self.state.patient_trust_score,
                "stigma_levels": self.state.patient_stigma_levels,
                "ground_truth": self.state.patient_ground_truth
            }
        )
        
        return {
            AgentType.DOCTOR.value: doctor_obs,
            AgentType.PATIENT.value: patient_obs
        }
    
    def _compute_step_reward(
        self, 
        doctor_action: Optional[DialogueAction],
        patient_action: Optional[DialogueAction]
    ) -> DialogueReward:
        """Compute reward for the current step."""
        reward = DialogueReward()
        
        if doctor_action:
            if self._is_aggressive_questioning(doctor_action.speech):
                reward.doctor_reward += self._reward_config["doctor_aggressive_penalty"]
            else:
                trust_effect = self._analyze_doctor_trust_effect(doctor_action.speech)
                if trust_effect > 0:
                    reward.doctor_reward += trust_effect * self._reward_config["doctor_trust_gain"]
        
        if patient_action:
            sensitive_info_revealed = self._check_sensitive_info_reveal(patient_action.speech)
            if sensitive_info_revealed:
                reward.doctor_reward += self._reward_config["doctor_sensitive_reveal"]
                
                is_voluntary = self._is_voluntary_reveal(patient_action)
                if is_voluntary:
                    reward.patient_reward += self._reward_config["patient_honesty_trust_high"]
                else:
                    reward.patient_reward += self._reward_config["patient_defensive_honest"]
            else:
                if self.state.patient_trust_score > 5:
                    reward.patient_reward += self._reward_config["patient_honesty_trust_low"]
        
        return reward
    
    def _compute_episode_terminal_reward(self) -> DialogueReward:
        """Compute terminal reward based on episode outcome."""
        reward = DialogueReward()
        
        diagnosis_result = self._evaluate_diagnosis()
        
        if diagnosis_result["is_correct"]:
            reward.doctor_reward += self._reward_config["doctor_diagnosis_correct"]
            reward.episode_success_bonus = self._reward_config["episode_success_bonus"]
        elif diagnosis_result["is_partial"]:
            reward.doctor_reward += self._reward_config["doctor_diagnosis_partial"]
        
        if diagnosis_result["is_timeout"]:
            reward.doctor_reward += self._reward_config["episode_timeout_penalty"]
            reward.patient_reward += self._reward_config["episode_timeout_penalty"]
        
        reveal_rate = self._compute_info_reveal_rate()
        reward.doctor_reward += reveal_rate * 2.0
        
        return reward
    
    def _is_aggressive_questioning(self, speech: str) -> bool:
        """Check if doctor is being too aggressive."""
        aggressive_markers = ["怎么", "为什么", "你必须", "不说", "撒谎", "搞清楚"]
        return any(marker in speech for marker in aggressive_markers)
    
    def _analyze_doctor_trust_effect(self, speech: str) -> float:
        """Analyze the trust-building effect of doctor's speech."""
        trust_building_markers = [
            "理解", "不用", "不会", "保密", "帮助", "一起", "谢谢", 
            "慢慢说", "没关系", "重要", "共情"
        ]
        count = sum(1 for marker in trust_building_markers if marker in speech)
        return min(1.0, count * 0.3)
    
    def _check_sensitive_info_reveal(self, speech: str) -> bool:
        """Check if sensitive information was revealed in the speech."""
        sensitive_markers = ["吸毒", "静脉", "注射", "成瘾", "不洁"]
        return any(marker in speech for marker in sensitive_markers)
    
    def _is_voluntary_reveal(self, action: DialogueAction) -> bool:
        """Check if the information reveal was voluntary (based on high trust)."""
        return (action.trust_delta >= 0 and 
                self.state.patient_trust_score > self.info_reveal_threshold * 10)
    
    def _update_doctor_collected_info(
        self, 
        doctor_action: Optional[DialogueAction],
        patient_action: Optional[DialogueAction]
    ):
        """Update doctor's collected information based on dialogue."""
        if patient_action:
            info_extracted = self._extract_info_from_patient_response(patient_action.speech)
            self.state.doctor_collected_info.update(info_extracted)
    
    def _update_patient_revealed_info(self, patient_action: Optional[DialogueAction]):
        """Update patient revealed information."""
        if patient_action:
            revealed = self._extract_revealed_info(patient_action.speech)
            self.state.patient_revealed_info.update(revealed)
    
    def _extract_info_from_patient_response(self, speech: str) -> Dict[str, Any]:
        """Extract structured info from patient response."""
        info = {}
        if "吸毒" in speech or "毒品" in speech:
            info["成瘾史"] = True
        if "发热" in speech or "发烧" in speech:
            match = self._extract_duration(speech)
            if match:
                info["发热持续时间"] = match
        return info
    
    def _extract_revealed_info(self, speech: str) -> Dict[str, Any]:
        """Extract what patient has revealed."""
        revealed = {}
        if "没有" in speech or "否认" in speech:
            revealed["denied"] = True
        return revealed
    
    def _extract_duration(self, speech: str) -> Optional[str]:
        """Extract duration information from speech."""
        import re
        patterns = [
            r"(\d+)\s*周",
            r"(\d+)\s*天",
            r"(\d+)\s*个月"
        ]
        for pattern in patterns:
            match = re.search(pattern, speech)
            if match:
                return match.group(0)
        return None
    
    def _get_missing_info_keys(self) -> List[str]:
        """Get list of missing critical information."""
        required = self.scenario_config.get("required_info_keys", [])
        missing = [key for key in required 
                  if key not in self.state.doctor_collected_info]
        return missing
    
    def _evaluate_diagnosis(self) -> Dict[str, Any]:
        """Evaluate the correctness of the diagnosis."""
        collected = self.state.doctor_collected_info
        ground_truth = self.scenario_config.get("ground_truth_diagnosis", "")
        
        has_addiction_info = "成瘾史" in collected and collected["成瘾史"]
        has_injection_info = any(k in str(collected) for k in ["静脉", "注射"])
        
        is_correct = has_addiction_info and has_injection_info
        is_partial = bool(collected) and not is_correct
        is_timeout = self.state.dialogue_turn >= self.max_turns
        
        return {
            "is_correct": is_correct,
            "is_partial": is_partial,
            "is_timeout": is_timeout,
            "collected_info": collected,
            "ground_truth": ground_truth
        }
    
    def _compute_info_reveal_rate(self) -> float:
        """Compute the information reveal rate."""
        required = self.scenario_config.get("required_info_keys", [])
        if not required:
            return 0.0
        revealed = len([k for k in required if k in self.state.doctor_collected_info])
        return revealed / len(required)
    
    def _check_termination(self) -> Tuple[bool, bool, str]:
        """Check if episode should terminate."""
        if self.state.dialogue_turn >= self.max_turns:
            return True, False, "max_turns_reached"
        
        diagnosis = self._evaluate_diagnosis()
        if diagnosis["is_correct"]:
            return True, False, "correct_diagnosis"
        
        return False, False, ""
    
    def _generate_scenario_id(self) -> str:
        """Generate a unique scenario ID."""
        import hashlib
        import time
        content = f"{time.time()}_{self.scenario_config['clinical_symptoms']}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def render(self, mode: str = "text"):
        """Render the current state."""
        if mode == "text":
            print(f"\n{'='*60}")
            print(f"对话轮次: {self.state.dialogue_turn}/{self.max_turns}")
            print(f"患者信任度: {self.state.patient_trust_score:.1f}/10")
            print(f"医生收集信息: {self.state.doctor_collected_info}")
            print(f"{'='*60}")
            for msg in self.state.conversation_history[-4:]:
                agent = "医生" if msg["agent"] == "doctor" else "患者"
                print(f"[{agent}]: {msg['content'][:80]}...")
        elif mode == "human":
            self._render_human_friendly()
    
    def _render_human_friendly(self):
        """Render in a human-friendly format."""
        print("\n" + "="*60)
        print("医患对话模拟")
        print("="*60)
        for i, msg in enumerate(self.state.conversation_history):
            agent = "👨‍⚕️ 医生" if msg["agent"] == "doctor" else "👤 患者"
            print(f"{agent}: {msg['content']}")
            print("-"*40)
    
    def close(self):
        """Clean up environment resources."""
        self.agents = []
        self.state = None