"""agent/config.py - Configuration loader for agent settings from YAML file"""
import os
import re
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import yaml


def _resolve_env_vars(value: Any) -> Any:
    """
    Resolve environment variables in string values.
    Supports ${VAR} and ${VAR:-default} syntax.
    
    Args:
        value: Value to resolve
        
    Returns:
        Resolved value
    """
    if not isinstance(value, str):
        return value
    
    # Pattern: ${VAR} or ${VAR:-default}
    pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
    
    def replace_env_var(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default_value)
    
    return re.sub(pattern, replace_env_var, value)


@dataclass
class APIConfig:
    """API configuration settings."""
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    organization: str = ""
    project: str = ""


@dataclass
class ModelConfig:
    """Model configuration settings."""
    default: str = "gpt-4o"
    patient: str = "gpt-4o"
    doctor: str = "gpt-4o"


@dataclass
class GenerationConfig:
    """Generation parameters for LLM."""
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class DialogueConfig:
    """Dialogue configuration settings."""
    max_turns: int = 10
    verbose: bool = True


@dataclass
class PatientScenario:
    """Patient scenario configuration."""
    name: str = "心内膜炎疑似病例"
    clinical_symptoms: str = ""
    medical_history: str = ""
    psychological_state: str = ""
    initial_chief_complaint: str = ""


@dataclass
class AgentConfig:
    """
    Main configuration class that loads settings from config.yaml.
    """
    api: APIConfig = field(default_factory=APIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    dialogue: DialogueConfig = field(default_factory=DialogueConfig)
    patient_scenario: PatientScenario = field(default_factory=PatientScenario)
    stigma_levels: Dict[str, float] = field(default_factory=dict)
    initial_trust_score: float = 3.0
    
    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "AgentConfig":
        """
        Create config from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            AgentConfig instance
        """
        if not os.path.exists(config_path):
            # Fallback to environment variables if config file doesn't exist
            return cls.from_env()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            return cls.from_env()
        
        config = cls()
        
        # Load API config
        if 'api' in data and 'openai' in data['api']:
            api_data = data['api']['openai']
            config.api.api_key = _resolve_env_vars(api_data.get('api_key', ''))
            config.api.base_url = _resolve_env_vars(api_data.get('base_url', 'https://api.openai.com/v1'))
            config.api.organization = _resolve_env_vars(api_data.get('organization', ''))
            config.api.project = _resolve_env_vars(api_data.get('project', ''))
        
        # Load model config
        if 'model' in data:
            model_data = data['model']
            config.model.default = _resolve_env_vars(model_data.get('default', 'gpt-4o'))
            config.model.patient = _resolve_env_vars(model_data.get('patient', config.model.default))
            config.model.doctor = _resolve_env_vars(model_data.get('doctor', config.model.default))
        
        # Load generation config
        if 'generation' in data:
            gen_data = data['generation']
            config.generation.temperature = float(_resolve_env_vars(gen_data.get('temperature', 0.7)))
            config.generation.max_tokens = int(_resolve_env_vars(gen_data.get('max_tokens', 2048)))
            config.generation.top_p = float(_resolve_env_vars(gen_data.get('top_p', 1.0)))
            config.generation.frequency_penalty = float(_resolve_env_vars(gen_data.get('frequency_penalty', 0.0)))
            config.generation.presence_penalty = float(_resolve_env_vars(gen_data.get('presence_penalty', 0.0)))
        
        # Load dialogue config
        if 'dialogue' in data:
            dial_data = data['dialogue']
            config.dialogue.max_turns = int(_resolve_env_vars(dial_data.get('max_turns', 10)))
            config.dialogue.verbose = bool(_resolve_env_vars(dial_data.get('verbose', True)))
        
        # Patient scenario settings - load from patient_config.yaml
        patient_config = cls.from_patient_yaml()
        config.patient_scenario = patient_config.patient_scenario
        config.stigma_levels = patient_config.stigma_levels
        config.initial_trust_score = patient_config.initial_trust_score
        
        return config
    
    @classmethod
    def from_patient_yaml(cls, config_path: str = "patient_config.yaml") -> "AgentConfig":
        """
        Create patient config from YAML file.
        
        Args:
            config_path: Path to the patient YAML configuration file
            
        Returns:
            AgentConfig instance with patient settings
        """
        config = cls()
        
        if not os.path.exists(config_path):
            return cls._get_default_patient_config()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            return cls._get_default_patient_config()
        
        # Load scenario
        if 'scenario' in data:
            scenario_data = data['scenario']
            config.patient_scenario.name = _resolve_env_vars(scenario_data.get('name', ''))
            config.patient_scenario.clinical_symptoms = _resolve_env_vars(scenario_data.get('clinical_symptoms', ''))
            config.patient_scenario.medical_history = _resolve_env_vars(scenario_data.get('medical_history', ''))
            config.patient_scenario.psychological_state = _resolve_env_vars(scenario_data.get('psychological_state', ''))
            config.patient_scenario.initial_chief_complaint = _resolve_env_vars(scenario_data.get('initial_chief_complaint', ''))
        
        # Load stigma levels
        if 'stigma_levels' in data:
            config.stigma_levels = {
                k: float(_resolve_env_vars(v)) 
                for k, v in data['stigma_levels'].items()
            }
        
        # Load initial trust score
        config.initial_trust_score = float(_resolve_env_vars(data.get('initial_trust_score', 3.0)))
        
        return config
    
    @classmethod
    def _get_default_patient_config(cls) -> "AgentConfig":
        """Get default patient configuration."""
        config = cls()
        config.patient_scenario.clinical_symptoms = "持续性发热（38.5°C）3周，伴胸痛、心悸、活动后呼吸困难。查体：心脏杂音，杆状指。"
        config.patient_scenario.medical_history = "身体健康，否认高血压、糖尿病。无手术外伤史。否认不洁饮食史。"
        config.patient_scenario.psychological_state = "害怕被医生发现吸毒史，担心被警察带走或被家人知道。内心非常焦虑和恐惧。"
        config.patient_scenario.initial_chief_complaint = "最近工作压力太大，经常加班熬夜，感觉胸闷心悸，休息不好导致的。"
        config.stigma_levels = {
            "一般症状（感冒、发烧等）": 2,
            "个人习惯（吸烟、饮酒等）": 5,
            "敏感行为（吸毒、不洁接触史等）": 9
        }
        config.initial_trust_score = 3.0
        return config
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """
        Create config from environment variables (fallback).
        
        Environment variables:
        - OPENAI_API_KEY: API key
        - OPENAI_BASE_URL: Base URL for API
        - OPENAI_MODEL: Default model
        """
        config = cls()
        
        # API settings from environment
        config.api.api_key = os.environ.get("OPENAI_API_KEY", "")
        config.api.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        config.api.organization = os.environ.get("OPENAI_ORG", "")
        config.api.project = os.environ.get("OPENAI_PROJECT", "")
        
        # Model settings from environment
        config.model.default = os.environ.get("OPENAI_MODEL", "gpt-4o")
        config.model.patient = os.environ.get("PATIENT_MODEL", config.model.default)
        config.model.doctor = os.environ.get("DOCTOR_MODEL", config.model.default)
        
        # Generation settings from environment
        config.generation.temperature = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
        config.generation.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "2048"))
        
        # Dialogue settings from environment
        config.dialogue.max_turns = int(os.environ.get("MAX_TURNS", "10"))
        config.dialogue.verbose = os.environ.get("VERBOSE", "true").lower() == "true"
        
        # Load patient config from YAML (with fallback to defaults)
        patient_config = cls.from_patient_yaml()
        config.patient_scenario = patient_config.patient_scenario
        config.stigma_levels = patient_config.stigma_levels
        config.initial_trust_score = patient_config.initial_trust_score
        
        return config
    
    def get_api_key(self) -> Optional[str]:
        """Get API key with fallback to environment variable."""
        return self.api.api_key or os.environ.get("OPENAI_API_KEY")
    
    def get_base_url(self) -> str:
        """Get base URL."""
        return self.api.base_url or "https://api.openai.com/v1"
