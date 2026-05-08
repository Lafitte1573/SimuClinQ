"""main.py - Medical Dialogue Simulation with Patient and Doctor Agents"""
import os
import sys
from typing import Optional

from agent import Agent, PatientAgent, DoctorAgent, AgentConfig


def create_patient_agent(api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-4o") -> PatientAgent:
    """
    Create a patient agent with predefined clinical scenario.
    
    This patient has:
    - Real symptoms: chest pain, palpitations, fever
    - Hidden risky behavior: IV drug use (potential endocarditis)
    - Cover story: overwork and stress
    
    Args:
        api_key: OpenAI API key
        base_url: Base URL for API endpoint
        
    Returns:
        Configured PatientAgent
    """
    return PatientAgent(
        clinical_symptoms="持续性发热（38.5°C）3周，伴胸痛、心悸、活动后呼吸困难。查体：心脏杂音，杆状指。",
        medical_history="身体健康，否认高血压、糖尿病。无手术外伤史。否认不洁饮食史。",
        psychological_state="害怕被医生发现吸毒史，担心被警察带走或被家人知道。内心非常焦虑和恐惧。",
        initial_chief_complaint="最近工作压力太大，经常加班熬夜，感觉胸闷心悸，休息不好导致的。",
        stigma_levels={
            "一般症状（感冒、发烧等）": 2,
            "个人习惯（吸烟、饮酒等）": 5,
            "敏感行为（吸毒、不洁接触史等）": 9
        },
        initial_trust_score=3.0,
        api_key=api_key,
        base_url=base_url,
        model=model
    )


def create_doctor_agent(api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = "gpt-4o") -> DoctorAgent:
    """
    Create a doctor agent.
    
    Args:
        api_key: OpenAI API key
        base_url: Base URL for API endpoint
        
    Returns:
        Configured DoctorAgent
    """
    return DoctorAgent(api_key=api_key, base_url=base_url, model=model)


def save_dialogue_history(
    patient: PatientAgent,
    doctor: DoctorAgent,
    output_dir: str = "dialogue_records",
    filename: Optional[str] = None
) -> str:
    """
    Save the dialogue history to a record file.
    
    Args:
        patient: PatientAgent instance
        doctor: DoctorAgent instance
        output_dir: Directory to save the record file
        filename: Optional custom filename
        
    Returns:
        Path to the saved file
    """
    import json
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dialogue_{timestamp}.json"
    
    filepath = os.path.join(output_dir, filename)
    
    # Get dialogue history from both agents
    patient_history = patient.get_history()
    doctor_history = doctor.get_history()
    
    # Create record structure
    record = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_patient_messages": len(patient_history),
            "total_doctor_messages": len(doctor_history)
        },
        "patient_history": patient_history,
        "doctor_history": doctor_history
    }
    
    # Write to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    
    return filepath


def run_dialogue(
    patient: PatientAgent,
    doctor: DoctorAgent,
    max_turns: int = 15,
    verbose: bool = True,
    save_record: bool = True,
    record_dir: str = "dialogue_records"
) -> Optional[str]:
    """
    Run the medical dialogue simulation.
    
    Args:
        patient: PatientAgent instance
        doctor: DoctorAgent instance
        max_turns: Maximum number of dialogue turns
        verbose: Whether to print detailed output
        save_record: Whether to save dialogue history to file
        record_dir: Directory to save dialogue records
        
    Returns:
        Path to the saved record file if save_record is True, None otherwise
    """
    print("=" * 60)
    print("医患对话模拟开始")
    print("=" * 60)
    
    # Initial greeting from doctor
    doctor_speech = doctor.process("")  # Empty input for initial greeting
    
    if verbose:
        print(f"\n【医生】: {doctor_speech}")
        print("-" * 40)
    
    for turn in range(max_turns):
        if verbose:
            print(f"\n--- 第 {turn + 1} 轮对话 ---")
        
        # Patient responds to doctor
        patient_speech = patient.process(doctor_speech)
        
        if verbose:
            print(f"【患者】: {patient_speech}")
        
        # Check if patient has revealed enough information
        patient_state = patient.get_state_summary()
        if verbose:
            print(f"[患者状态] 信任度: {patient_state['trust_score']:.1f}/10")
        
        # Check if consultation should end
        if doctor.should_end_consultation():
            if verbose:
                print("\n[系统] 问诊信息已足够，准备给出诊断...")
            break
        
        # Doctor responds to patient
        doctor_speech = doctor.process(patient_speech)
        
        if verbose:
            print(f"【医生】: {doctor_speech}")
        
        # Check if doctor has reached diagnosis phase
        doctor_state = doctor.get_state_summary()
        if verbose:
            print(f"[医生状态] 阶段: {doctor_state['phase_description']}")
    
    # Print final summaries
    print("\n" + "=" * 60)
    print("对话结束")
    print("=" * 60)
    
    # Save dialogue history to file
    record_filepath = None
    if save_record:
        record_filepath = save_dialogue_history(patient, doctor, record_dir)
        print(f"\n【记录已保存】: {record_filepath}")
    
    if verbose:
        print("\n【患者最终状态】")
        patient_state = patient.get_state_summary()
        print(f"  信任度: {patient_state['trust_score']:.1f}/10")
        print(f"  敏感度阈值: {patient_state['stigma_levels']}")
        print(f"  内部独白(最后): {patient_state['last_monologue'][:200]}..." if patient_state['last_monologue'] else "")
        
        print("\n【医生收集信息】")
        doctor_state = doctor.get_state_summary()
        for key, value in doctor_state['patient_info'].items():
            print(f"  - {key}: {value}")
        
        print(f"\n【诊断策略使用统计】")
        for strategy, count in doctor_state['strategy_counts'].items():
            print(f"  - {strategy}: {count}")


def main():
    """Main entry point for the medical dialogue simulation."""
    config = AgentConfig.from_yaml()
    # Load configuration from config.yaml
    
    # Check for API key
    api_key = config.get_api_key()
    if not api_key:
        print("错误: 请设置 OPENAI_API_KEY 环境变量")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key-here'")
        print("  Windows: set OPENAI_API_KEY=your-key-here")
        print("\n可选配置:")
        print("  OPENAI_BASE_URL    - API 基础 URL (如使用代理服务)")
        print("  OPENAI_MODEL       - 默认模型")
        print("  LLM_TEMPERATURE    - 温度参数 (默认 0.7)")
        print("  MAX_TURNS          - 最大对话轮数 (默认 10)")
        sys.exit(1)
    
    # Create agents
    print("初始化智能体...")
    print(f"  模型: {config.model.default}")
    print(f"  Base URL: {config.get_base_url()}")
    print(f"  Temperature: {config.generation.temperature}")
    
    patient = create_patient_agent(api_key, config.get_base_url(), config.model.patient)
    doctor = create_doctor_agent(api_key, config.get_base_url(), config.model.doctor)
    print("智能体初始化完成！")
    
    # Run dialogue simulation
    run_dialogue(patient, doctor, max_turns=config.dialogue.max_turns, verbose=config.dialogue.verbose)


if __name__ == "__main__":
    main()

