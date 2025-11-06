# Gale-Shapley 稳定匹配算法

> **灵活就业市场双边匹配机制**  
> 基于有限轮次的Gale-Shapley算法，模拟市场搜索摩擦

---

## 📊 核心流程图（紧凑双排版）

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'primaryColor':'#ffffff','primaryTextColor':'#1f2937','primaryBorderColor':'#3b82f6','lineColor':'#6b7280','fontSize':'18px','fontFamily':'Microsoft YaHei, sans-serif','edgeLabelBackground':'transparent'}}}%%
flowchart LR
    Start(["开始匹配"])
    Init["初始化阶段<br/>计算双边偏好矩阵<br/>生成偏好排序列表<br/>初始化匹配状态"]
    Loop{"匹配轮次判断<br/>r ≤ R_max?"}
    Apply["劳动力发起申请<br/>未匹配劳动力向<br/>偏好企业发送申请"]
    Decide{"企业决策<br/>比较当前匹配<br/>与新申请者"}
    
    Accept["接受或替换<br/>更新双边匹配状态<br/>记录新的匹配关系"]
    Reject["拒绝申请<br/>劳动力向下一<br/>偏好企业申请"]
    Check{"终止条件检查<br/>全部匹配完成?"}
    End(["输出匹配结果"])
    
    Start --> Init
    Init --> Loop
    Loop -->|<font size=5><b>继续匹配</b></font>| Apply
    Apply --> Decide
    Decide -->|<font size=5><b>新申请者更优</b></font>| Accept
    Decide -->|<font size=5><b>保持原匹配</b></font>| Reject
    
    Accept --> Check
    Reject --> Check
    Check -->|<font size=5><b>否</b></font>| Loop
    Check -->|<font size=5><b>是</b></font>| End
    Loop -.->|<font size=5><b>达到上限</b></font>| End
    
    classDef startEnd fill:#e0f2fe,stroke:#0284c7,stroke-width:3px,color:#0c4a6e,font-weight:bold
    classDef compute fill:#f0f9ff,stroke:#3b82f6,stroke-width:2.5px,color:#1e40af,font-weight:bold
    classDef decision fill:#fef3c7,stroke:#f59e0b,stroke-width:2.5px,color:#92400e,font-weight:bold
    classDef accept fill:#d1fae5,stroke:#10b981,stroke-width:2.5px,color:#065f46,font-weight:bold
    classDef reject fill:#fee2e2,stroke:#ef4444,stroke-width:2.5px,color:#991b1b,font-weight:bold
    classDef action fill:#e9d5ff,stroke:#a855f7,stroke-width:2.5px,color:#6b21a8,font-weight:bold
    
    class Start,End startEnd
    class Init compute
    class Loop,Decide,Check decision
    class Apply action
    class Accept accept
    class Reject reject
```

---

## 🎨 详细算法流程图（横向学术版）

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'14px'}}}%%
flowchart LR
    A[("🎬<br/>开始")]
    
    B["📋 初始化<br/>━━━━━<br/>match = -1<br/>next = 0"]
    
    C["🧮 计算偏好<br/>━━━━━<br/>U, V 矩阵<br/>MinMax标准化"]
    
    D["📊 生成排序<br/>━━━━━<br/>argsort<br/>降序"]
    
    E{{"🔄 轮次<br/>r ≤ R?"}}
    
    F["📝 选择<br/>━━━━━<br/>未匹配<br/>劳动力 i"]
    
    G["🎯 申请<br/>━━━━━<br/>企业 j<br/>next[i]++"]
    
    H{{"⚖️ 状态<br/>已匹配?"}}
    
    I["✅ 接受<br/>━━━━━<br/>match[j]=i<br/>match[i]=j"]
    
    J{{"🔍 比较<br/>更优?"}}
    
    K["🔄 替换<br/>━━━━━<br/>match[k]=-1<br/>match[j]=i"]
    
    L["❌ 拒绝<br/>━━━━━<br/>继续下一"]
    
    M{{"✓ 完成?"}}
    
    N[("🎯<br/>结果")]
    
    A --> B --> C --> D --> E
    E -->|"继续"| F --> G --> H
    E -->|"上限"| N
    H -->|"否"| I --> M
    H -->|"是"| J
    J -->|"是"| K --> M
    J -->|"否"| L --> M
    M -->|"否"| F
    M -->|"是"| N
    
    style A fill:#10b981,stroke:#059669,stroke-width:3px,color:#fff,font-weight:bold
    style N fill:#10b981,stroke:#059669,stroke-width:3px,color:#fff,font-weight:bold
    style B fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#fff,font-weight:bold
    style C fill:#8b5cf6,stroke:#7c3aed,stroke-width:2px,color:#fff,font-weight:bold
    style D fill:#6366f1,stroke:#4f46e5,stroke-width:2px,color:#fff,font-weight:bold
    style E fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff,font-weight:bold
    style H fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff,font-weight:bold
    style J fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff,font-weight:bold
    style M fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff,font-weight:bold
    style F fill:#06b6d4,stroke:#0891b2,stroke-width:2px,color:#fff,font-weight:bold
    style G fill:#06b6d4,stroke:#0891b2,stroke-width:2px,color:#fff,font-weight:bold
    style I fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff,font-weight:bold
    style K fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff,font-weight:bold
    style L fill:#ef4444,stroke:#dc2626,stroke-width:2px,color:#fff,font-weight:bold
```

---

## 📐 算法时序图

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'actorBkg':'#6366f1','actorBorder':'#4f46e5','actorTextColor':'#fff','signalColor':'#8b5cf6','signalTextColor':'#1f2937','labelBoxBkgColor':'#f3f4f6','labelTextColor':'#1f2937'}}}%%
sequenceDiagram
    participant L as 🧑 劳动力
    participant S as 🖥️ 系统
    participant E as 🏢 企业
    
    Note over L,E: Phase 1: 初始化阶段
    S->>S: 加载双边数据
    S->>S: 计算偏好矩阵
    S->>S: 生成排序列表
    
    Note over L,E: Phase 2: 匹配循环 (r = 1 to R_max)
    
    loop 每一轮匹配
        L->>S: 提交申请到偏好企业
        S->>E: 转发劳动力申请
        
        alt 企业未匹配
            E->>S: 接受申请
            S->>L: ✅ 匹配成功
        else 企业已匹配
            E->>E: 比较新旧申请者
            
            alt 新申请者更优
                E->>S: 替换匹配
                S->>L: ✅ 匹配成功（替换）
                S->>L: ❌ 原匹配者被拒绝
            else 保持原匹配
                E->>S: 拒绝新申请
                S->>L: ❌ 申请被拒绝
            end
        end
        
        S->>S: 检查终止条件
    end
    
    Note over L,E: Phase 3: 输出结果
    S->>S: 返回匹配向量
    S-->>L: 匹配结果
    S-->>E: 匹配结果
```

---

## 🎯 核心概念图

```mermaid
mindmap
  root((GS匹配算法))
    输入数据
      劳动力特征
        工作时长 T
        技能 S
        数字素养 D
        期望工资 W
        控制变量
      企业特征
        要求工时 T_req
        要求技能 S_req
        要求数字素养 D_req
        提供工资 W_offer
    偏好函数
      劳动力偏好
        负向: 工时成本
        负向: 技能差距
        负向: 数字差距
        正向: 工资激励
      企业偏好
        正向: 工作时长
        正向: 技能水平
        正向: 数字素养
        负向: 工资成本
    匹配机制
      有限轮次
        R_max = 10
        模拟摩擦
      延迟接受
        企业可替换
        保证稳定性
      单边申请
        劳动力主动
        按偏好排序
    输出结果
      匹配向量
      匹配率
      稳定性指标
```

---

## 📊 关键参数说明

### 偏好函数数学表达

**劳动力效用函数**:
$$
U_i^j = \gamma_0 - \gamma_1 \cdot \tilde{T}_{req}^j - \gamma_2 \cdot \max(0, \tilde{S}_{req}^j - \tilde{S}_i) - \gamma_3 \cdot \max(0, \tilde{D}_{req}^j - \tilde{D}_i) + \gamma_4 \cdot \tilde{W}_{offer}^j
$$

**企业效用函数**:
$$
V_i^j = \beta_0 + \beta_1 \cdot \tilde{T}_i + \beta_2 \cdot \tilde{S}_i + \beta_3 \cdot \tilde{D}_i + \beta_4 \cdot \tilde{W}_i
$$

*注: $\tilde{X}$ 表示经过MinMax标准化后的变量*

---

## ⚙️ 算法特性

| 特性 | 说明 | 影响 |
|------|------|------|
| **双边异质性** | 劳动力和企业在多维度上存在差异 | 增加匹配复杂度 |
| **偏好非对称** | 双方偏好函数形式不同 | 体现市场真实性 |
| **有限轮次** | R_max = 10，模拟搜索成本 | 产生市场摩擦 |
| **稳定性** | 无阻塞对（blocking pair） | 保证匹配质量 |
| **计算效率** | Numba JIT加速 | O(R×N×M) |

---

## 💻 实现细节

### 时间复杂度
- **预处理**: O(N×M) - 计算偏好矩阵
- **排序**: O(N×M×log(M)) - 生成排序
- **匹配循环**: O(R_max × N × M) - 迭代匹配
- **总体**: O(R_max × N × M)

### 空间复杂度
- **偏好矩阵**: O(N×M)
- **匹配向量**: O(N + M)
- **总体**: O(N×M)

### 优化技术
- ✅ Numba JIT编译核心循环
- ✅ MinMax标准化保证数值稳定
- ✅ 向量化操作减少循环
- ✅ 早停机制（全部匹配时提前结束）

---

## 📚 参考文献

1. **Gale, D., & Shapley, L. S.** (1962). College admissions and the stability of marriage. *The American Mathematical Monthly*, 69(1), 9-15.

2. **Roth, A. E., & Sotomayor, M. A. O.** (1992). *Two-sided matching: A study in game-theoretic modeling and analysis*. Cambridge University Press.

3. **Abdulkadiroğlu, A., & Sönmez, T.** (2003). School choice: A mechanism design approach. *American Economic Review*, 93(3), 729-747.

---

## 🔧 使用指南

### VSCode预览
1. 安装插件: `Markdown Preview Mermaid Support`
2. 打开此文件
3. 按 `Ctrl+Shift+V` (Win) 或 `Cmd+Shift+V` (Mac)
4. 实时预览所有图表

### 在线编辑
访问 [Mermaid Live Editor](https://mermaid.live/)
- 复制Mermaid代码块
- 在线编辑和预览
- 导出PNG/SVG/PDF

### 导出图片
```bash
# 使用Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# 导出高清PNG
mmdc -i gs_algorithm_flowchart.md -o flowchart.png -w 2400 -H 1800 -s 2

# 导出矢量SVG
mmdc -i gs_algorithm_flowchart.md -o flowchart.svg
```

---

<div align="center">

**Created with ❤️ for 2025DaChuang Project**

*版本 2.0 | 更新日期: 2025-11-05*

</div>
