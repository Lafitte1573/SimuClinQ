"""
Agent module for medical dialogue simulation.
"""
from .base import Agent
from .patient import PatientAgent
from .doctor import DoctorAgent
from .config import AgentConfig, APIConfig, ModelConfig, GenerationConfig

__all__ = [
    "Agent", 
    "PatientAgent", 
    "DoctorAgent",
    "AgentConfig",
    "APIConfig", 
    "ModelConfig", 
    "GenerationConfig"
]