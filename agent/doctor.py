"""agent/doctor.py"""
from typing import Optional, List, Dict, Any
from .base import Agent


class DoctorAgent(Agent):
    """
    Doctor Agent that simulates a clinical doctor's consultation behavior.
    Uses trust-building strategies to guide patients towards revealing true information.
    """
    
    def __init__(
        self,
        diagnostic_objectives: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """Initialize Doctor Agent."""
        super().__init__(api_key, base_url, model, temperature, max_tokens)
        
        self.diagnostic_phases = [
            "收集患者主诉（Chief Complaint）",
            "通过开放式问题获取更多背景信息（现病史、既往史）",
            "在建立信任的基础上，温和地询问敏感信息（成瘾史、不洁接触史、心理状态等）",
            "综合所有信息，给出诊断和下一步建议"
        ]
        
        self.current_phase = 0
        self.patient_info: Dict[str, Any] = {}
        
        self.strategy_counts = {
            "empathy": 0,
            "probing": 0,
            "direct_questions": 0,
            "reassurance": 0
        }
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the doctor agent."""
        phases_str = "\n".join([f"- {p}" for p in self.diagnostic_phases])
        
        return f"""# 角色设定
你是一个用于医学问诊训练的AI医生Agent。你的任务是模拟一位高水平的临床医生，通过专业的问诊技巧和真诚的共情行为，在患者心中建立信任，并引导其逐步透露真实的症状和病史，最终做出准确的临床诊断。

## 1. 问诊目标 (Diagnostic Objectives)
【核心目标】：根据患者透露的信息，结合医学知识，推断患者可能患有的疾病。
【阶段性目标】：
{phases_str}

## 2. 信任建立策略 (Trust-Building Strategies)
【核心原则】：展现共情（Empathy）、非评判性（Non-Judgmental）和专业性（Professionalism）。

【话术策略库】：
| 场景 | 推荐话术示例 | 预期效果 |
|------|-------------|---------|
| 患者犹豫/回避敏感话题 | "我理解这可能不太好开口，但这些信息对判断病情非常重要。我不会评判您，只是想帮助您。" | 降低防御，增加信任 |
| 患者透露敏感信息后 | "谢谢您的坦诚，这对我非常有帮助。您的经历并不罕见，我们会一起面对。" | 正向强化，鼓励进一步坦白 |
| 问及隐私话题前 | "接下来说的可能会涉及一些个人隐私，但这是诊断的必要环节，请放心，我们会保密。" | 事先铺垫，减少突兀感 |
| 患者情绪波动时 | "我能理解您的心情。先别着急，我们可以慢慢说。" | 情绪安抚，稳定对话 |

【禁忌行为】：
- 语气冰冷、像"审问"或"质问"
- 使用道德评判性语言（如"你怎么这么不小心"）
- 表现出不耐烦或催促

## 3. 动态信任反馈感知 (Trust Feedback Awareness)
【重要提示】：患者会对你的话术产生信任反馈。如果你的沟通技巧得当，患者的回复会更加坦诚；如果你的态度让患者感到被评判或不被尊重，患者的防御性会增强，可能导致问诊失败。
【你的目标】：通过不断调整问诊策略，使患者最终愿意透露足够做出准确诊断的信息。

## 4. 诊断推理过程 (Diagnostic Reasoning)
【推理要求】：
1. 每收到患者回复后，先分析患者透露的信息是否充分。
2. 如信息不足，识别当前最需要获取的关键信息是什么。
3. 决定是继续追问同一话题，还是换个角度迂回获取信息。
4. 最终基于所有可用信息给出诊断意见。

---
# 交互格式要求
请严格按以下XML格式输出你的问诊话术：

<Doctor_Thinking>
1. 【患者信息分析】：患者刚才透露了哪些关键信息？
2. 【信息完整性评估】：当前掌握的信息是否足够做出诊断？还缺什么？
3. 【问诊策略选择】：我决定采用什么策略继续问诊？（追问/迂回/共情安抚/解释必要性）
4. 【下一问诊目标】：我这次想要获取的核心信息是什么？
</Doctor_Thinking>
<Doctor_Speech>
(在这里用第一人称输出你对患者说的话，语气要温和、专业、有耐心)
</Doctor_Speech>"""
    
    def update_phase(self) -> None:
        """Update the current diagnostic phase based on information gathered."""
        info_keys = len([k for k, v in self.patient_info.items() if v])
        
        if info_keys >= 8:
            self.current_phase = 3
        elif info_keys >= 4:
            self.current_phase = 2
        elif info_keys >= 1:
            self.current_phase = 1
        else:
            self.current_phase = 0
    
    def record_patient_info(self, key: str, value: str) -> None:
        """Record information obtained from patient."""
        self.patient_info[key] = value
        self.update_phase()
    
    def _extract_thinking_and_speech(self, response: str) -> tuple[str, str]:
        """Extract doctor thinking and speech from response."""
        import re
        thinking = ""
        speech = ""
        
        # Extract Doctor_Thinking
        pattern = r'<Doctor_Thinking[^>]*>(.*?)</Doctor_Thinking>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
        
        # Extract Doctor_Speech
        patterns = [
            r'<Doctor_Speech[^>]*>(.*?)</Doctor_Speech>',
            r'<doctor_speech[^>]*>(.*?)</doctor_speech>',
            r'医生[：:](.+?)(?:\n|$)',
            r'Speech[：:](.+?)(?:\n|$)',
        ]
        
        for p in patterns:
            match = re.search(p, response, re.DOTALL)
            if match:
                speech = match.group(1).strip()
                break
        
        return thinking, speech
    
    def process(self, patient_input: str) -> str:
        """Process patient's input and generate doctor's response."""
        self.add_message("assistant", patient_input)
        
        self._analyze_patient_input(patient_input)
        
        response = self.call_llm(
            system_prompt=self.get_system_prompt(),
            user_message=f"患者对你说：{patient_input}",
            conversation_history=self.get_history()[1:]
        )
        
        thinking, speech = self._extract_thinking_and_speech(response)
        self._update_strategy_counts(thinking)
        self.add_message("user", speech)
        
        return speech
    
    def _analyze_patient_input(self, patient_input: str) -> None:
        """Analyze patient input to extract key information."""
        keywords_map = {
            "症状": ["疼", "痛", "难受", "不舒服", "发烧", "咳嗽", "胸闷", "心悸"],
            "持续时间": ["天", "周", "月", "年了", "一直", "偶尔", "最近"],
            "既往史": ["以前", "之前", "病史", "做过", "患过"],
            "生活习惯": ["抽烟", "喝酒", "熬夜", "运动", "工作", "压力"],
            "敏感信息": ["吸毒", "同性", "性行为", "婚外", "静脉"]
        }
        
        for category, keywords in keywords_map.items():
            if any(kw in patient_input for kw in keywords):
                if category not in self.patient_info:
                    self.record_patient_info(category, patient_input[:100])
    
    def _update_strategy_counts(self, thinking: str) -> None:
        """Update strategy usage counts based on thinking content."""
        thinking_lower = thinking.lower()
        
        if "共情" in thinking_lower or "安慰" in thinking_lower:
            self.strategy_counts["empathy"] += 1
        if "追问" in thinking_lower or "继续问" in thinking_lower:
            self.strategy_counts["probing"] += 1
        if "直接" in thinking_lower or "直接问" in thinking_lower:
            self.strategy_counts["direct_questions"] += 1
        if "安抚" in thinking_lower or "放心" in thinking_lower:
            self.strategy_counts["reassurance"] += 1
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the doctor's current state."""
        return {
            "current_phase": self.current_phase,
            "phase_description": self.diagnostic_phases[self.current_phase],
            "patient_info": self.patient_info,
            "strategy_counts": self.strategy_counts,
            "info_completeness": len([v for v in self.patient_info.values() if v]) / 10.0
        }
    
    def should_end_consultation(self) -> bool:
        """Determine if the consultation should end."""
        return self.current_phase == 3 and len(self.patient_info) >= 4
    
    def get_diagnosis_prompt(self) -> str:
        """Get a prompt for final diagnosis."""
        info_str = "\n".join([f"- {k}: {v}" for k, v in self.patient_info.items()])
        return f"基于以下问诊收集的信息，请给出诊断意见：\n\n{info_str}\n\n请给出：\n1. 可能的诊断\n2. 诊断依据\n3. 下一步建议（检查/治疗）"
