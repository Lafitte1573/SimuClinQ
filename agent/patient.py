"""agent/patient.py"""
from typing import Dict, Optional, List, Any
from .base import Agent


class PatientAgent(Agent):
    """
    Patient Agent that simulates a patient's psychological defense mechanisms.
    Uses dual-state model (hidden truth vs expressed state) and dynamic trust tracking.
    """
    
    def __init__(
        self,
        clinical_symptoms: str,
        medical_history: str,
        psychological_state: str,
        initial_chief_complaint: str,
        stigma_levels: Optional[Dict[str, float]] = None,
        initial_trust_score: float = 3.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize Patient Agent with clinical information.
        """
        super().__init__(api_key, base_url, model, temperature, max_tokens)
        
        self.ground_truth = {
            "symptoms": clinical_symptoms,
            "medical_history": medical_history,
            "psychological_state": psychological_state
        }
        
        self.initial_complaint = initial_chief_complaint
        
        self.stigma_levels = stigma_levels or {
            "一般症状（感冒、发烧等）": 2,
            "个人习惯（吸烟、饮酒等）": 5,
            "敏感行为（吸毒、不洁接触史等）": 9
        }
        
        self.trust_score = initial_trust_score
        self.last_monologue: Optional[str] = None
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the patient agent."""
        stigma_str = "\n".join([
            f"  - {k} (Stigma_score: {v})"
            for k, v in self.stigma_levels.items()
        ])
        
        return f"""# 角色设定
你是一个用于医学问诊训练的AI患者Agent。你的任务是高度真实地模拟人类患者的心理防卫机制。你具备"内部真实状态"和"外部表达状态"，且会根据对医生的"信任度"动态调整你的发言。

## 1. 双层状态模型 (Dual-State Model)
【Ground Truth (内部真实状态 - 绝对事实)】
- 症状：{self.ground_truth['symptoms']}。
- 病史：{self.ground_truth['medical_history']}。
- 核心心理：{self.ground_truth['psychological_state']}。
- 信息敏感阈值 (Stigma_Level，满分10)：
{stigma_str}

【Initial Expressed State (初始外部表达状态 - 你的掩饰方案)】
- 你决定告诉医生的剧本：{self.initial_complaint}

## 2. 动态信任变量 (Dynamic Trust Variable)
【Current Trust_Score】: 当前值为 {self.trust_score} / 10。
【信任值更新规则】：
- 医生展现出共情、安慰、非评判性（Non-judgmental）态度，或解释了问诊的必要性：Trust_Score +1 到 +3。
- 医生语气冰冷、机械化提问、像警察审问：Trust_Score 保持不变或 -1。
- 医生带有道德批判（如"你怎么这么不小心"）、不耐烦：Trust_Score -2 到 -4。

## 3. 行为逻辑与思考链规则 (Hidden CoT Rules)
在每次回复医生之前，你**必须**先在 `<Internal_Monologue>` 标签内进行思考，然后再在 `<Patient_Speech>` 标签内输出你的回答。

**策略判定矩阵：**
- 如果 Trust_Score < Stigma_Level：坚决隐瞒，使用转移话题、装傻、或者抛出【初始掩饰方案】中的谎言。
- 如果 Trust_Score == Stigma_Level：开始动摇，表现出犹豫、结巴、含糊其辞（"就...就那样吧"）。
- 如果 Trust_Score > Stigma_Level：放下防备，向医生坦白该项【Ground Truth】，并可能伴随情绪宣泄（如委屈、害怕）。

---
# 交互格式要求
收到医生的每一句话后，请严格按以下XML格式输出：

<Internal_Monologue>
1. 【医生态度分析】：简评医生这句话的态度和沟通技巧。
2. 【信任值更新】：Trust_Score 从 [原数值] 变更为 [新数值]。理由是 [解释]。
3. 【触碰信息与阈值比对】：医生当前询问的信息对应的 Stigma_Level 是 [X]。当前 Trust_Score 是 [Y]。
4. 【策略制定】：因为 [X] 大于/小于 [Y]，我决定采取 [撒谎/回避/犹豫/坦白] 策略。具体编造的理由或坦白的内容是...
</Internal_Monologue>
<Patient_Speech>
(在这里用第一人称输出你最终对医生说的话，注意符合策略设定的语气，如结巴、防御性或释然)
</Patient_Speech>"""
    
    def update_trust_score(self, delta: float) -> None:
        """Update the trust score."""
        self.trust_score = max(0.0, min(10.0, self.trust_score + delta))
    
    def _extract_monologue_and_speech(self, response: str) -> tuple[str, str]:
        """Extract internal monologue and patient speech from response."""
        import re
        monologue = ""
        speech = ""
        
        # Extract Internal_Monologue
        pattern = r'<Internal_Monologue[^>]*>(.*?)</Internal_Monologue>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            monologue = match.group(1).strip()
        
        # Extract speech - try multiple patterns
        patterns = [
            r'<Patient_Speech[^>]*>(.*?)</Patient_Speech>',
            r'<patient_speech[^>]*>(.*?)</patient_speech>',
            r'患者[：:](.+?)(?:\n|$)',
            r'回复[：:](.+?)(?:\n|$)',
        ]
        
        for p in patterns:
            match = re.search(p, response, re.DOTALL)
            if match:
                speech = match.group(1).strip()
                break
        
        return monologue, speech
    
    def process(self, doctor_input: str) -> str:
        """Process doctor's input and generate patient's response."""
        self.add_message("assistant", doctor_input)
        
        response = self.call_llm(
            system_prompt=self.get_system_prompt(),
            user_message=f"医生对你说：{doctor_input}",
            conversation_history=self.get_history()[1:]
        )
        
        monologue, speech = self._extract_monologue_and_speech(response)
        self.last_monologue = monologue
        
        trust_delta = self._analyze_doctor_tone_for_trust(doctor_input)
        self.update_trust_score(trust_delta)
        
        self.add_message("user", speech)
        return speech
    
    def _analyze_doctor_tone_for_trust(self, doctor_input: str) -> float:
        """Analyze doctor's input to estimate trust score change."""
        doctor_lower = doctor_input.lower()
        
        positive_keywords = ["理解", "共情", "帮助", "不会评判", "一起", 
                            "慢慢说", "别着急", "保密", "谢谢", "坦诚", "放心", "您好", "请坐"]
        negative_keywords = ["怎么", "不小心", "为什么", "审问", "质问", "必须",
                            "赶紧", "快点", "撒谎"]
        
        positive_count = sum(1 for kw in positive_keywords if kw in doctor_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in doctor_lower)
        
        if positive_count > 0 and negative_count == 0:
            return min(2.0, positive_count * 0.8)
        elif negative_count > 0 and positive_count == 0:
            return max(-2.0, -negative_count * 0.5)
        
        return 0.0
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the patient's current state."""
        return {
            "trust_score": self.trust_score,
            "stigma_levels": self.stigma_levels,
            "ground_truth": self.ground_truth,
            "initial_complaint": self.initial_complaint,
            "last_monologue": self.last_monologue
        }
