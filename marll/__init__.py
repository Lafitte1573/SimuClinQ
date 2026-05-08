"""marll/__init__.py - Medical Agent Learning Library"""
from .environment import MedicalDialogueEnv, DialogueState, DialogueAction, DialogueObservation, DialogueReward, AgentType
from .agents import QwenDoctorAgent, QwenPatientAgent
from .trainer import MAPPOTrainer, TrainingConfig
from .reward import RewardCalculator

__version__ = "0.1.0"

__all__ = [
    "MedicalDialogueEnv",
    "DialogueState",
    "DialogueAction", 
    "DialogueObservation",
    "DialogueReward",
    "AgentType",
    "QwenDoctorAgent",
    "QwenPatientAgent",
    "MAPPOTrainer",
    "TrainingConfig",
    "RewardCalculator"
]