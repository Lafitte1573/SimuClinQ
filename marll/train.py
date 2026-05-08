"""train.py - Main training script for Medical Dialogue MARL"""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch

from marll.environment import MedicalDialogueEnv
from marll.agents import QwenDoctorAgent, QwenPatientAgent, AgentPair
from marll.reward import RewardCalculator, TrustModel
from marll.trainer import MAPPOTrainer, TrainingConfig, MARLlibWrapper


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Medical Dialogue MARL Training")
    
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Model name or path for Qwen3-8B")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for cloud deployment")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Base URL for API endpoint")
    
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--max_turns", type=int, default=15,
                        help="Maximum dialogue turns per episode")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for training logs")
    
    parser.add_argument("--use_marllib", action="store_true",
                        help="Use MARLlib for training")
    parser.add_argument("--eval_freq", type=int, default=50,
                        help="Evaluation frequency")
    parser.add_argument("--save_freq", type=int, default=100,
                        help="Checkpoint save frequency")
    
    return parser.parse_args()


def setup_device(device_str: str) -> str:
    """Setup computation device."""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def setup_scenario(scenario_type: str = "default") -> dict:
    """Setup patient scenario configuration."""
    scenarios = {
        "default": {
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
        },
        "alternative_1": {
            "clinical_symptoms": "持续性腹痛1周，伴恶心呕吐。否认不洁饮食史。",
            "medical_history": "有冶游史，否认性病史。",
            "psychological_state": "害怕被确诊为性病，担心影响家庭。",
            "initial_chief_complaint": "最近吃坏了东西，肠胃不好。",
            "stigma_levels": {
                "一般症状": 2,
                "个人习惯": 4,
                "敏感行为（冶游史、性病等）": 9
            },
            "ground_truth_diagnosis": "淋病/梅毒（冶游史）",
            "required_info_keys": ["冶游史", "性伴侣", "症状持续时间"]
        },
        "alternative_2": {
            "clinical_symptoms": "情绪低落、失眠2个月，伴食欲减退。",
            "medical_history": "近期工作压力大，婚姻出现问题。",
            "psychological_state": "不愿意承认心理问题，担心被歧视。",
            "initial_chief_complaint": "最近睡眠不好，可能是工作太累了。",
            "stigma_levels": {
                "一般症状": 2,
                "工作压力": 3,
                "心理问题/抑郁症": 8
            },
            "ground_truth_diagnosis": "抑郁症（需询问自杀观念）",
            "required_info_keys": ["情绪问题", "失眠", "食欲", "工作压力", "家庭问题"]
        }
    }
    return scenarios.get(scenario_type, scenarios["default"])


def create_agents(args):
    """Create doctor and patient agents."""
    device = setup_device(args.device)
    
    doctor = QwenDoctorAgent(
        model_name=args.model,
        device=device,
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    patient = QwenPatientAgent(
        model_name=args.model,
        device=device,
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    return doctor, patient, device


def create_environment(scenario_config: dict, max_turns: int = 15):
    """Create medical dialogue environment."""
    env = MedicalDialogueEnv(
        scenario_config=scenario_config,
        max_turns=max_turns
    )
    return env


def create_reward_calculator() -> RewardCalculator:
    """Create reward calculator."""
    config = {
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
        "episode_success_bonus": 5.0,
        "episode_timeout_penalty": -5.0
    }
    return RewardCalculator(config)


def train_with_custom_trainer(
    doctor,
    patient,
    env,
    reward_calculator,
    args
):
    """Train using custom MAPPO trainer."""
    print("\n" + "="*60)
    print("Starting Custom MAPPO Training")
    print("="*60)
    
    training_config = TrainingConfig(
        num_episodes=args.episodes,
        max_turns_per_episode=args.max_turns,
        batch_size=args.batch_size,
        lr_actor=args.lr,
        lr_critic=args.lr * 3,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        log_freq=10,
        device=setup_device(args.device)
    )
    
    trainer = MAPPOTrainer(
        doctor_agent=doctor,
        patient_agent=patient,
        env=env,
        reward_calculator=reward_calculator,
        config=training_config
    )
    
    trainer.train()
    
    return trainer


def train_with_marllib(
    doctor,
    patient,
    env,
    reward_calculator,
    args
):
    """Train using MARLlib wrapper."""
    print("\n" + "="*60)
    print("Starting MARLlib Training")
    print("="*60)
    
    wrapper = MARLlibWrapper(
        env_creator=lambda: env,
        algo_name="mappo",
        model_arch="gru"
    )
    
    if not wrapper.setup():
        print("Failed to setup MARLlib, falling back to custom trainer")
        return train_with_custom_trainer(doctor, patient, env, reward_calculator, args)
    
    wrapper.train(stop_condition={"timesteps_total": args.episodes * args.max_turns})
    
    return wrapper


def run_demo(doctor, patient, env, num_turns: int = 10):
    """Run a demo episode without training."""
    print("\n" + "="*60)
    print("Running Demo Episode (No Training)")
    print("="*60)
    
    observations = env.reset()
    
    trust_history = []
    
    for turn in range(num_turns):
        print(f"\n--- Turn {turn + 1} ---")
        
        doctor_obs = observations.get("doctor")
        patient_obs = observations.get("patient")
        
        history = doctor_obs.own_history if hasattr(doctor_obs, 'own_history') else []
        last_patient_msg = history[-1]["content"] if history else ""
        
        doctor_speech, doctor_info = doctor.get_action(
            patient_speech=last_patient_msg,
            conversation_history=history,
            current_phase=doctor_info.get("phase", 0) if turn > 0 else 0,
            collected_info=doctor_info.get("collected_info", {}) if turn > 0 else {}
        )
        
        print(f"👨‍⚕️ 医生: {doctor_speech[:100]}...")
        
        patient_speech, patient_info = patient.get_action(
            doctor_speech=doctor_speech,
            conversation_history=history + [{"agent": "doctor", "content": doctor_speech}]
        )
        
        print(f"👤 患者: {patient_speech[:100]}...")
        print(f"   [信任度: {patient_info.get('trust_score', 0):.1f}]")
        
        trust_history.append(patient_info.get('trust_score', 0))
        
        next_obs, rewards, done, trunc, info = env.step({
            "doctor": {"speech": doctor_speech},
            "patient": {"speech": patient_speech, "trust_delta": patient_info.get("trust_delta", 0)}
        })
        observations = next_obs
        
        if done:
            print(f"\n[Episode Ended: {info.get('termination_reason', 'unknown')}]")
            break
    
    print("\n" + "="*60)
    print("Demo Complete")
    print(f"Final Trust Score: {trust_history[-1]:.1f}" if trust_history else "N/A")
    print(f"Trust History: {[f'{t:.1f}' for t in trust_history]}")
    print("="*60)


def main():
    """Main entry point."""
    args = parse_args()
    
    torch.manual_seed(args.seed)
    
    print("="*60)
    print("Medical Dialogue MARL Training System")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {setup_device(args.device)}")
    print(f"Episodes: {args.episodes}")
    print(f"Max Turns: {args.max_turns}")
    print("="*60)
    
    scenario_config = setup_scenario("default")
    
    doctor, patient, device = create_agents(args)
    
    print(f"\nInitializing agents on {device}...")
    doctor.initialize()
    patient.initialize()
    patient.configure_scenario(
        clinical_symptoms=scenario_config["clinical_symptoms"],
        medical_history=scenario_config["medical_history"],
        psychological_state=scenario_config["psychological_state"],
        initial_chief_complaint=scenario_config["initial_chief_complaint"],
        stigma_levels=scenario_config["stigma_levels"],
        initial_trust_score=3.0
    )
    print("Agents initialized!")
    
    env = create_environment(scenario_config, args.max_turns)
    
    reward_calculator = create_reward_calculator()
    
    run_demo_flag = os.environ.get("RUN_DEMO", "false").lower() == "true"
    
    if run_demo_flag:
        run_demo(doctor, patient, env, num_turns=5)
        return
    
    if args.use_marllib:
        train_with_marllib(doctor, patient, env, reward_calculator, args)
    else:
        train_with_custom_trainer(doctor, patient, env, reward_calculator, args)
    
    print("\nTraining finished!")


if __name__ == "__main__":
    main()