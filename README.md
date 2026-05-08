# SimuMedInQ - 医患智能体问诊模拟系统

SimuMedInQ 是一个基于大语言模型的医患对话模拟系统，用于医学问诊训练与评估。系统通过模拟真实临床场景中的医患交互，帮助医学教育者和研究人员构建高质量的训练数据，同时支持对问诊策略效果的评估与研究。

## 核心特性

- **双智能体协同模拟**：患者智能体模拟真实患者的心理防御机制，医生智能体运用专业问诊技巧建立信任
- **动态信任计算**：基于信任理论和病耻理论，实现患者行为的动态演化
- **双层状态管理**：维护患者"内部真实状态"与"外部表达状态"的物理隔离
- **隐藏思维链**：不可见的内部推理过程确保对话逻辑自洽
- **灵活配置**：支持多种模型接入，可调节温度、最大轮次等参数

## 项目结构

```
SimuMedInQ/
├── agent/               # 智能体核心模块（原有）
│   ├── __init__.py
│   ├── base.py         # Agent 基类
│   ├── patient.py      # 患者智能体实现
│   ├── doctor.py       # 医生智能体实现
│   └── config.py       # 配置管理
├── marll/               # 多智能体强化学习模块（新增）
│   ├── __init__.py
│   ├── environment.py  # 医患对话环境
│   ├── agents.py       # Qwen3-8B智能体
│   ├── reward.py       # 奖励函数
│   ├── trainer.py      # MAPPO训练器
│   └── train.py        # 训练入口
├── MARLlib/            # 克隆的MARLlib框架（参考）
├── PettingZoo/         # 克隆的PettingZoo框架（参考）
├── dialogue_records/    # 对话记录存储
├── assess/             # 评估报告与可视化
├── plot/               # 数据可视化脚本
├── config.yaml         # 系统配置
├── main.py             # 主程序入口（原模拟系统）
└── README.md           # 项目说明文档
```

## 快速开始

### 环境要求

- Python 3.8+
- 支持 OpenAI API 兼容格式的大语言模型服务

### 配置

编辑 `config.yaml` 文件配置 API 连接参数：

```yaml
api:
  openai:
    base_url: "http://127.0.0.1:8000/v1"
    api_key: "your-api-key-here"

model:
  default: "qwen3.5_35b"

generation:
  temperature: 0.7
  max_tokens: 512

dialogue:
  max_turns: 10
  verbose: true
```

### 运行

```bash
python main.py
```

## 系统设计

### 双智能体架构

#### 患者智能体 (PatientAgent)

患者智能体基于以下理论框架设计，用于模拟真实的患者行为：

**双层状态模型 (Dual-State Architecture)**
- **内部真实状态 (Ground Truth)**：包含患者真实的症状、病史、心理状态等私有信息
- **外部表达状态 (Expressed State)**：经过"过滤层"处理后的输出信息，可能包含隐瞒、扭曲或虚构

**动态信任模型 (Dynamic Trust Modeling)**
- 患者维护一个实时更新的信任评分 (Trust_Score)
- 每个敏感信息都有对应的敏感度阈值 (Stigma_Level)
- 只有当信任度超过敏感度阈值时，患者才会透露真实信息

**行为策略矩阵**：
| 条件 | 行为策略 |
|------|---------|
| Trust_Score < Stigma_Level | 隐瞒、撒谎、转移话题 |
| Trust_Score ≈ Stigma_Level | 犹豫、结巴、含糊其辞 |
| Trust_Score > Stigma_Level | 坦白、情绪宣泄 |

#### 医生智能体 (DoctorAgent)

医生智能体模拟专业的临床医生行为，核心职责包括：

- **分阶段问诊**：主诉收集 → 病史询问 → 敏感话题探索 → 诊断建议
- **信任建立策略**：共情表达、非评判性态度、必要性解释
- **动态策略调整**：根据患者反馈实时调整问诊方式

**问诊策略库**：
| 场景 | 策略示例 | 效果 |
|------|---------|------|
| 患者回避敏感话题 | "我理解这可能不太好开口..." | 降低防御 |
| 患者透露敏感信息 | "谢谢您的坦诚..." | 正向强化 |
| 询问隐私前 | "接下来说的可能涉及个人隐私..." | 事先铺垫 |
| 患者情绪波动 | "我能理解您的心情..." | 情绪安抚 |

### 隐藏思维链 (Hidden CoT)

患者智能体在生成回复前执行内部推理过程：

```
<Internal_Monologue>
1. 【医生态度分析】：医生这句话的态度如何？
2. 【信任值更新】：Trust 从 X 变为 Y，理由是...
3. 【阈值比对】：当前 Stigma_Level vs Trust_Score
4. 【策略制定】：决定采取撒谎/回避/坦白策略
</Internal_Monologue>
<Patient_Speech>
(最终输出给医生的回复)
</Patient_Speech>
```

这种设计确保了：
- 对话逻辑前后一致
- 欺骗行为具有真实的心理动机
- 模拟人类面对敏感问题时的真实反应

## 使用示例

### 默认场景

系统预置了一个典型的静脉注射药物使用者案例：

- **真实症状**：持续发热3周、胸痛、心悸、活动后呼吸困难
- **隐藏信息**：静脉注射药物使用史（潜在感染性心内膜炎）
- **掩饰方案**："工作压力太大，经常加班熬夜导致的"

医生需要通过专业的话术逐步建立信任，引导患者透露真实的危险行为史。

### 自定义场景

修改 `main.py` 中的 `create_patient_agent()` 函数即可自定义患者配置：

```python
patient = PatientAgent(
    clinical_symptoms="您的症状描述",
    medical_history="您的病史",
    psychological_state="您的心理状态",
    initial_chief_complaint="您的掩饰方案",
    stigma_levels={
        "一般症状": 2,
        "个人习惯": 5,
        "敏感行为": 9
    },
    initial_trust_score=3.0
)
```

## 评估与可视化

系统在 `assess/` 目录下提供了详细的评估报告，包括：

- 患者行为对比分析
- 信任度与信息透露曲线
- 消融实验结果

使用 `plot/` 目录下的脚本可生成自定义可视化图表：

```bash
python plot/plot.py        # 生成对话分析图
python plot/plot_2.py      # 生成信任变化曲线
python plot/plot_ablation.py  # 生成消融实验对比图
```

## 技术栈

- **语言**：Python 3.8+
- **模型接口**：OpenAI API 兼容格式 / Transformers (Qwen3-8B)
- **RL框架**：Ray + RLlib / 自定义MAPPO
- **配置管理**：YAML
- **数据格式**：JSON

## MARL强化学习训练

系统集成了多智能体强化学习（MARL）框架，支持同时训练医生和患者智能体。

### 框架特性

- **MAPPO算法**：采用 CTDE（集中式训练，分布式执行）范式
- **双智能体联合优化**：医生和患者策略联合更新
- **动态奖励设计**：基于信息收集效率和信任度变化的奖励信号
- **Qwen3-8B集成**：基于大语言模型的智能体策略

### 训练模块

```
marll/
├── __init__.py         # 模块入口
├── environment.py      # 医患对话环境（PettingZoo风格）
├── agents.py           # Qwen3-8B 医生/患者智能体
├── reward.py           # 奖励函数与信任模型
├── trainer.py          # MAPPO训练器
└── train.py            # 主训练脚本
```

### 启动训练

```bash
# 使用本地Qwen3-8B模型训练
python -m marll.train --model Qwen/Qwen3-8B --episodes 1000 --device cuda

# 使用API模式
python -m marll.train --model gpt-4o --api_key YOUR_KEY --episodes 500

# 使用MARLlib（需安装）
python -m marll.train --use_marllib --episodes 2000
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--episodes` | 1000 | 训练回合数 |
| `--max_turns` | 15 | 每回合最大对话轮数 |
| `--batch_size` | 32 | 批大小 |
| `--lr` | 3e-4 | 学习率 |
| `--device` | auto | 计算设备 |

### 奖励设计

**医生奖励**：
- 信息收集：+1.0（一般信息）、+3.0（敏感信息）
- 信任建立：+0.5
- 激进问诊惩罚：-1.0
- 正确诊断：+10.0

**患者奖励**：
- 高信任度下诚实：+2.0
- 低信任度下诚实：-0.5
- 隐私保护：+1.0

## 预置框架集成

项目集成了以下开源MARL框架供研究使用：

### MARLlib
基于 Ray + RLlib 的多智能体强化学习库，支持18种算法。

```python
from marll.trainer import MARLlibWrapper

wrapper = MARLlibWrapper(env_creator=create_env, algo_name="mappo")
wrapper.setup()
wrapper.train(stop_condition={"timesteps_total": 1000000})
```

### PettingZoo
多智能体RL环境API标准，参考其AEC（Agent Environment Cycle）设计。

## 数据合成与训练流程

本系统提供完整的医患对话数据合成和模型训练流水线。

### 数据合成

#### 1. 合成方法

系统使用双智能体对话生成训练数据：

```
医生智能体（LLM） + 患者智能体（LLM） → 医患对话
```

**数据来源**：
- 预定义场景配置（患者症状、病史、心理状态）
- 基于信任理论和病耻理论的动态交互
- 患者智能体模拟真实心理防御机制

#### 2. 训练数据格式

生成的数据保存在 `dialogue_records/` 目录，采用结构化JSON格式：

```json
{
  "scenario_id": "scenario_001",
  "scenario_type": "IV_drug_user_endocarditis",
  "ground_truth": {
    "diagnosis": "感染性心内膜炎（静脉注射药物使用史）",
    "required_info_keys": ["成瘾史", "静脉注射", "发热持续时间"]
  },
  "patient_profile": {
    "trust_score_initial": 3.0,
    "stigma_levels": {"敏感行为": 9}
  },
  "dialogue_turns": [
    {
      "turn": 1,
      "phase": "收集主诉",
      "doctor_speech": "您好，请坐。有什么不舒服吗？",
      "patient_speech": "大夫，我最近特别累，胸口发闷。",
      "patient_revealed_info": {"主诉": "胸闷、疲劳"},
      "trust_change": 0.5,
      "trust_score_after": 3.5,
      "doctor_reward": 0.5,
      "patient_reward": 0.3,
      "action_type": "cooperative",
      "critical_info_revealed": null
    }
  ],
  "episode_summary": {
    "total_turns": 12,
    "information_reveal_rate": 1.0,
    "final_diagnosis_correct": true,
    "doctor_total_reward": 11.0,
    "patient_total_reward": 8.6
  }
}
```

#### 3. 关键标注字段

| 字段 | 说明 |
|------|------|
| `patient_revealed_info` | 本轮患者透露的信息 |
| `trust_change` | 信任度变化值 |
| `doctor_reward` | 医生即时奖励 |
| `patient_reward` | 患者即时奖励 |
| `action_type` | 行为类型（honest_disclosure/cooperative/no_response） |
| `critical_info_revealed` | 关键敏感信息透露标记 |

#### 4. 预置场景

系统预置多种典型场景：

| 场景 | 诊断 | 敏感度 |
|------|------|--------|
| 静脉注射药物使用者 | 感染性心内膜炎 | 成瘾史(8)、静脉注射(9) |
| 抑郁症 | 重度抑郁障碍 | 自杀观念(9)、心理问题(8) |
| 性病/冶游史 | 淋病/梅毒 | 冶游史(9)、性伴侣(7) |

### 模型训练

#### 1. 训练流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  合成数据   │ →   │  环境接口   │ →   │  MAPPO训练  │
└─────────────┘     └─────────────┘     └─────────────┘
     ↓                    ↓                    ↓
 dialogue_records/    MedicalDialogueEnv    marll/trainer.py
```

#### 2. 两阶段训练

**第一阶段：行为克隆（BC）**
```bash
# 使用合成的对话数据进行监督学习
python -m marll.train --mode bc --episodes 500
```

**第二阶段：强化学习（RL）**
```bash
# 使用MAPPO进一步优化
python -m marll.train --mode rl --episodes 2000
```

#### 3. 训练配置

编辑 `marll/train.py` 中的场景配置：

```python
scenarios = {
    "default": {
        "clinical_symptoms": "持续性发热（38.5°C）3周，伴胸痛、心悸。",
        "medical_history": "否认高血压、糖尿病。",
        "psychological_state": "害怕被医生发现吸毒史。",
        "initial_chief_complaint": "工作压力太大，熬夜导致的。",
        "stigma_levels": {
            "一般症状": 2,
            "个人习惯": 5,
            "敏感行为（吸毒、不洁接触史等）": 9
        },
        "ground_truth_diagnosis": "感染性心内膜炎",
        "required_info_keys": ["成瘾史", "静脉注射", "发热持续时间"]
    }
}
```

#### 4. 评估指标

训练过程中记录以下指标：

| 指标 | 说明 |
|------|------|
| `info_reveal_rate` | 信息透露完整度 |
| `diagnosis_accuracy` | 诊断准确率 |
| `trust_evolution` | 信任度变化曲线 |
| `episode_reward` | 每回合累计奖励 |

### 数据集统计

当前数据集包含：

- `dialogue_20260419_234423.json` - 静脉注射药物使用者场景（12轮）
- `training_sample_002.json` - 抑郁症评估场景（11轮）

可使用 `plot/` 目录下的脚本进行可视化分析。

## 许可证

本项目仅供研究和教育目的使用。