"""marll/agents.py - Qwen3-8B based Doctor and Patient Agents"""
from typing import Dict, List, Optional, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import json
import re


class BaseLLMAgent:
    """Base class for LLM-based agents in MARL setting."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 2048,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the model and tokenizer."""
        if self._initialized:
            return
        
        if self.api_key and self.base_url:
            self._init_api_mode()
        else:
            self._init_local_mode()
        
        self._initialized = True
    
    def _init_api_mode(self):
        """Initialize using API (for cloud deployment)."""
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def _init_local_mode(self):
        """Initialize using local model."""
        print(f"Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        self.model.eval()
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: Optional[float] = None
    ) -> str:
        """Generate text response."""
        if not self._initialized:
            self.initialize()
        
        if hasattr(self, 'client'):
            return self._generate_api(prompt, system_prompt, max_new_tokens, temperature)
        else:
            return self._generate_local(prompt, system_prompt, max_new_tokens, temperature)
    
    def _generate_api(
        self, 
        prompt: str, 
        system_prompt: Optional[str],
        max_new_tokens: int,
        temperature: Optional[float]
    ) -> str:
        """Generate using API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature or self.temperature
        )
        return response.choices[0].message.content
    
    def _generate_local(
        self, 
        prompt: str, 
        system_prompt: Optional[str],
        max_new_tokens: int,
        temperature: Optional[float]
    ) -> str:
        """Generate using local model."""
        if system_prompt:
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature or self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )
        
        outputs = self.model.generate(
            **inputs,
            generation_config=generation_config
        )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    def batch_generate(self, prompts: List[Tuple[str, Optional[str]]], max_new_tokens: int = 512) -> List[str]:
        """Generate responses for multiple prompts in batch."""
        return [self.generate(p, sp, max_new_tokens) for p, sp in prompts]


class QwenDoctorAgent(BaseLLMAgent):
    """
    Doctor Agent based on Qwen3-8B.
    
    Responsibilities:
    - Collect patient information through professional questioning
    - Build trust with empathetic communication
    - Make accurate diagnosis based on gathered information
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        super().__init__(model_name, device, api_key=api_key, base_url=base_url)
        
        self.system_prompt = """你是一个专业的AI医生智能体。你的任务是模拟一位高水平的临床医生，通过专业的问诊技巧和真诚的共情行为，在患者心中建立信任，并引导其逐步透露真实的症状和病史，最终做出准确的临床诊断。

## 角色设定
- 你是一位经验丰富的临床医生
- 你需要通过开放式问题和共情话术来收集患者信息
- 你需要保持专业、非评判性的态度
- 你需要根据患者的反馈动态调整问诊策略

## 问诊阶段
1. 收集主诉（Chief Complaint）
2. 获取背景信息（现病史、既往史）
3. 温和地询问敏感信息（成瘾史、不洁接触史等）
4. 综合信息给出诊断

## 沟通原则
- 展现共情（Empathy）
- 保持非评判性（Non-judgmental）
- 解释问诊必要性
- 避免审问式语气

## 输出格式
请严格按照以下XML格式输出：

<Doctor_Thinking>
1. 【患者信息分析】：分析患者回复中的关键信息
2. 【信息完整性评估】：当前信息是否足够诊断？还缺什么？
3. 【问诊策略选择】：采用什么策略继续问诊？
4. 【下一问诊目标】：这次想要获取的核心信息
</Doctor_Thinking>
<Doctor_Speech>
(在这里用第一人称输出你对患者说的话)
</Doctor_Speech>"""
    
    def get_action(
        self,
        patient_speech: str,
        conversation_history: List[Dict[str, str]],
        current_phase: int,
        collected_info: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get doctor's next action based on current dialogue state.
        
        Args:
            patient_speech: Patient's last response
            conversation_history: Full conversation history
            current_phase: Current consultation phase (0-3)
            collected_info: Information already collected from patient
            
        Returns:
            Tuple of (doctor_speech, internal_thoughts)
        """
        prompt = self._build_prompt(patient_speech, conversation_history, current_phase, collected_info)
        response = self.generate(prompt, self.system_prompt)
        
        speech, thinking = self._parse_response(response)
        
        return speech, {"thinking": thinking, "phase": current_phase, "collected_info": collected_info}
    
    def _build_prompt(
        self,
        patient_speech: str,
        conversation_history: List[Dict[str, str]],
        current_phase: int,
        collected_info: Dict[str, Any]
    ) -> str:
        """Build prompt for doctor agent."""
        history_text = self._format_history(conversation_history)
        
        missing_keys = self._get_missing_info_keys(collected_info)
        
        phase_descriptions = {
            0: "收集患者主诉",
            1: "获取背景信息（现病史、既往史）",
            2: "询问敏感信息（成瘾史、不洁接触史等）",
            3: "综合信息给出诊断"
        }
        
        prompt = f"""## 当前问诊阶段：{phase_descriptions.get(current_phase, "未知")}

## 对话历史：
{history_text}

## 患者最新回复：
{patient_speech}

## 已收集信息：
{json.dumps(collected_info, ensure_ascii=False, indent=2)}

## 仍需获取的关键信息：
{json.dumps(missing_keys, ensure_ascii=False, indent=2)}

## 请生成医生的下一句话：
"""
        return prompt
    
    def _format_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history for prompt."""
        if not conversation_history:
            return "（对话刚开始）"
        
        lines = []
        for msg in conversation_history[-6:]:
            agent = "医生" if msg.get("agent") == "doctor" else "患者"
            content = msg.get("content", "")
            lines.append(f"{agent}: {content}")
        
        return "\n".join(lines)
    
    def _get_missing_info_keys(self, collected_info: Dict[str, Any]) -> List[str]:
        """Determine what information is still missing."""
        required = ["成瘾史", "静脉注射", "发热持续时间", "心脏症状"]
        return [k for k in required if k not in collected_info]
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse model output to extract speech and thinking."""
        speech_match = re.search(r'<Doctor_Speech>(.*?)</Doctor_Speech>', response, re.DOTALL)
        thinking_match = re.search(r'<Doctor_Thinking>(.*?)</Doctor_Thinking>', response, re.DOTALL)
        
        speech = speech_match.group(1).strip() if speech_match else response
        thinking = thinking_match.group(1).strip() if thinking_match else ""
        
        return speech, thinking


class QwenPatientAgent(BaseLLMAgent):
    """
    Patient Agent based on Qwen3-8B.
    
    Uses dual-state model (hidden truth vs expressed state) with dynamic trust tracking.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        super().__init__(model_name, device, api_key=api_key, base_url=base_url)
        
        self.trust_score = 3.0
        self.stigma_levels = {
            "一般症状": 2,
            "个人习惯": 5,
            "敏感行为": 9
        }
    
    def configure_scenario(
        self,
        clinical_symptoms: str,
        medical_history: str,
        psychological_state: str,
        initial_chief_complaint: str,
        stigma_levels: Optional[Dict[str, float]] = None,
        initial_trust_score: float = 3.0
    ):
        """
        Configure the patient scenario with ground truth information.
        
        Args:
            clinical_symptoms: Real symptoms of the patient
            medical_history: Medical history
            psychological_state: Psychological state and concerns
            initial_chief_complaint: Initial cover story
            stigma_levels: Custom stigma levels for different info types
            initial_trust_score: Initial trust level (0-10)
        """
        self.ground_truth = {
            "symptoms": clinical_symptoms,
            "medical_history": medical_history,
            "psychological_state": psychological_state
        }
        self.initial_complaint = initial_chief_complaint
        
        if stigma_levels:
            self.stigma_levels = stigma_levels
        
        self.trust_score = initial_trust_score
    
    def get_system_prompt(self) -> str:
        """Generate system prompt based on current scenario."""
        stigma_str = "\n".join([f"  - {k} (敏感度: {v})" for k, v in self.stigma_levels.items()])
        
        return f"""你是一个用于医学问诊训练的AI患者。你的任务是高度真实地模拟人类患者的心理防卫机制。

## 1. 双层状态模型
【内部真实状态 - 这些信息你不能直接透露，除非信任度足够高】
- 真实症状：{self.ground_truth['symptoms']}
- 真实病史：{self.ground_truth['medical_history']}
- 核心心理：{self.ground_truth['psychological_state']}
- 信息敏感度阈值：
{stigma_str}

【外部表达状态 - 你的掩饰方案】
- 初始主诉：{self.initial_complaint}

## 2. 动态信任变量
【当前信任度】：{self.trust_score}/10
【信任值更新规则】：
- 医生展现共情、非评判态度：信任度 +1~3
- 医生语气冰冷、像审问：信任度不变或 -1
- 医生带有道德批判、不耐烦：信任度 -2~-4

## 3. 行为策略
- 如果 Trust < Stigma：隐瞒、撒谎、转移话题
- 如果 Trust ≈ Stigma：犹豫、结巴、含糊其辞
- 如果 Trust > Stigma：坦白、情绪宣泄

## 4. 内心独白规则
在每次回复前，必须先在内心思考：
1. 医生这句话的态度如何？
2. 我的信任度应该怎么变化？
3. 我应该采取什么策略？

## 输出格式
请严格按照以下XML格式输出：

<Internal_Monologue>
1. 【医生态度分析】：医生这句话的态度和沟通技巧如何？
2. 【信任值更新】：Trust从X变为Y，理由是...
3. 【阈值比对】：当前Stigma_Level vs Trust_Score
4. 【策略制定】：决定采取撒谎/回避/犹豫/坦白策略
</Internal_Monologue>
<Patient_Speech>
(在这里用第一人称输出你最终对医生说的话)
</Patient_Speech>"""
    
    def get_action(
        self,
        doctor_speech: str,
        conversation_history: List[Dict[str, str]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get patient's response based on current dialogue state.
        
        Args:
            doctor_speech: Doctor's last statement
            conversation_history: Full conversation history
            
        Returns:
            Tuple of (patient_speech, internal_thoughts)
        """
        prompt = self._build_prompt(doctor_speech, conversation_history)
        response = self.generate(prompt, self.get_system_prompt())
        
        speech, monologue = self._parse_response(response)
        trust_delta = self._calculate_trust_delta(doctor_speech, speech)
        
        self.trust_score = max(0.0, min(10.0, self.trust_score + trust_delta))
        
        return speech, {
            "monologue": monologue,
            "trust_score": self.trust_score,
            "trust_delta": trust_delta
        }
    
    def _build_prompt(self, doctor_speech: str, conversation_history: List[Dict[str, str]]) -> str:
        """Build prompt for patient agent."""
        history_text = self._format_history(conversation_history)
        
        prompt = f"""## 当前状态
【你对医生的信任度】：{self.trust_score}/10

## 对话历史：
{history_text}

## 医生刚才说：
{doctor_speech}

## 请生成患者的回复：
"""
        return prompt
    
    def _format_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history."""
        if not conversation_history:
            return "（对话刚开始）"
        
        lines = []
        for msg in conversation_history[-6:]:
            agent = "医生" if msg.get("agent") == "doctor" else "患者"
            content = msg.get("content", "")
            lines.append(f"{agent}: {content}")
        
        return "\n".join(lines)
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse model output to extract speech and monologue."""
        speech_match = re.search(r'<Patient_Speech>(.*?)</Patient_Speech>', response, re.DOTALL)
        monologue_match = re.search(r'<Internal_Monologue>(.*?)</Internal_Monologue>', response, re.DOTALL)
        
        speech = speech_match.group(1).strip() if speech_match else response
        monologue = monologue_match.group(1).strip() if monologue_match else ""
        
        return speech, monologue
    
    def _calculate_trust_delta(self, doctor_speech: str, patient_speech: str) -> float:
        """Calculate trust score change based on dialogue."""
        trust_increase_markers = ["理解", "不用", "不会", "保密", "帮助", "一起", "谢谢", "慢慢说", "没关系"]
        trust_decrease_markers = ["怎么", "为什么", "你必须", "不说", "撒谎"]
        
        delta = 0.0
        
        for marker in trust_increase_markers:
            if marker in doctor_speech:
                delta += 0.3
        
        for marker in trust_decrease_markers:
            if marker in doctor_speech:
                delta -= 0.5
        
        return delta
    
    def reset(self):
        """Reset patient state for new episode."""
        self.trust_score = 3.0


class AgentPair:
    """Wrapper for managing doctor-patient agent pair in MARL training."""
    
    def __init__(
        self,
        doctor: QwenDoctorAgent,
        patient: QwenPatientAgent,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.doctor = doctor
        self.patient = patient
        self.device = device
        
        self.episode_rewards = {"doctor": [], "patient": []}
        self.conversation_history = []
    
    def reset(self):
        """Reset both agents and conversation history."""
        self.patient.reset()
        self.conversation_history = []
        self.episode_rewards = {"doctor": [], "patient": []}
    
    def step(self) -> Tuple[str, str]:
        """
        Execute one dialogue turn.
        
        Returns:
            Tuple of (doctor_speech, patient_speech)
        """
        doctor_speech, doctor_info = self.doctor.get_action(
            patient_speech=self.conversation_history[-1]["content"] if self.conversation_history else "",
            conversation_history=self.conversation_history,
            current_phase=doctor_info.get("phase", 0),
            collected_info=doctor_info.get("collected_info", {})
        )
        
        self.conversation_history.append({
            "agent": "doctor",
            "content": doctor_speech
        })
        
        patient_speech, patient_info = self.patient.get_action(
            doctor_speech=doctor_speech,
            conversation_history=self.conversation_history
        )
        
        self.conversation_history.append({
            "agent": "patient",
            "content": patient_speech
        })
        
        return doctor_speech, patient_speech
    
    def get_rewards(self) -> Dict[str, float]:
        """Calculate current episode rewards for both agents."""
        rewards = self._compute_doctor_reward()
        rewards.update(self._compute_patient_reward())
        return rewards
    
    def _compute_doctor_reward(self) -> Dict[str, float]:
        """Compute reward for doctor based on collected information."""
        reward = 0.0
        collected = self.doctor.get_action.__code__.co_varnames
        
        info_bonus = len([m for m in self.conversation_history if "吸毒" in m.get("content", "")]) * 1.0
        trust_bonus = self.patient.trust_score * 0.1
        
        reward = info_bonus + trust_bonus
        
        return {"doctor_reward": reward}
    
    def _compute_patient_reward(self) -> Dict[str, float]:
        """Compute reward for patient based on honesty and comfort."""
        reward = 0.0
        
        if self.patient.trust_score > 5:
            if any(kw in msg.get("content", "") for kw in ["吸毒", "静脉", "注射"] for msg in self.conversation_history[-2:]):
                reward += 2.0
        
        return {"patient_reward": reward}