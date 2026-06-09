I now have strong coverage across all five directions. Compiling the structured literature review.

---

# 文献调研结果:欺诈通话检测 + LLM 鲁棒性评测

说明:以下文献均通过 WebSearch/WebFetch 实际检索得到。会议/期刊与年份基于检索到的元数据(arXiv、ACL Anthology、AAAI、MDPI 等)。标注 "arXiv preprint" 的为预印本,正式版发表信息以括注为准。引用字段已按可生成 BibTeX 的格式整理。

---

## 方向 E:Fraud-R1(核心参考论文)

### [E1] Fraud-R1: A Multi-Round Benchmark for Assessing the Robustness of LLM Against Augmented Fraud and Phishing Inducements
- 作者:Shu Yang, Shenzhe Zhu, Zeyu Wu, Keyu Wang, Junchi Yao, Junchao Wu, Lijie Hu, Mengdi Li, Derek F. Wong, Di Wang
- 发表:ACL 2025 Findings(arXiv:2502.12904)
- 核心方法一句话:提出一个多轮评测流水线,沿"信任建立(credibility building)→紧迫感制造(urgency creation)→情感操纵(emotional manipulation)"三阶段,在助手模式与角色扮演两种情境下评估 LLM 对 8,564 条欺诈/钓鱼诱导样本的抵抗鲁棒性(覆盖 5 类欺诈)。
- 优点:首个系统化的多轮欺诈诱导鲁棒性基准;社工策略分阶段建模,贴近真实诈骗话术;中英双语,可对比跨语言鲁棒性差异。
- 局限:聚焦"LLM 是否被诱导/受骗"(生成式防御视角),而非"分类器识别欺诈文本"的判别式任务;角色扮演场景下评测主观性较强;数据构造依赖 LLM 生成,可能存在分布偏置。
- 引用:Yang, S., Zhu, S., Wu, Z., Wang, K., Yao, J., Wu, J., Hu, L., Li, M., Wong, D. F., & Wang, D. (2025). *Fraud-R1: A Multi-Round Benchmark for Assessing the Robustness of LLM Against Augmented Fraud and Phishing Inducements*. Findings of ACL 2025. arXiv:2502.12904.
- 链接:https://arxiv.org/abs/2502.12904 ; 代码 https://github.com/kaustpradalab/Fraud-R1

---

## 方向 A:诈骗/钓鱼/欺诈文本检测方法

### [A1] Exploring Machine Learning and Transformer-based Approaches for Deceptive Text Classification
- 作者:(arXiv 2308.05476,作者见原文)
- 发表:arXiv preprint, 2023
- 核心方法一句话:系统对比传统 ML(SVM、逻辑回归、随机森林等)与 Transformer 模型在欺骗性文本(deceptive text)分类上的表现。
- 优点:横向覆盖传统特征工程与预训练模型,适合做"方法谱系"对照;实验设置清晰。
- 局限:数据集与诈骗电话场景不完全对齐;Transformer 与传统模型对比的特征预处理不完全可比。
- 引用:*Exploring Machine Learning and Transformer-based Approaches for Deceptive Text Classification*. (2023). arXiv:2308.05476.
- 链接:https://arxiv.org/abs/2308.05476

### [A2] Deep Learning Approaches for Multi-Class Classification of Phishing Text Messages
- 发表:Journal of Cybersecurity and Privacy (MDPI), 2025, 5(4):102
- 核心方法一句话:用深度学习对钓鱼短信进行多类别(而非二分类)细粒度分类。
- 优点:多类别设定贴近真实诈骗类型划分;短信(SMS)文本与诈骗通话文本性质接近,迁移价值高。
- 局限:期刊论文非顶会;多类别标注的类别不平衡问题处理有限。
- 引用:*Deep Learning Approaches for Multi-Class Classification of Phishing Text Messages*. (2025). Journal of Cybersecurity and Privacy, 5(4), 102. MDPI.
- 链接:https://www.mdpi.com/2624-800X/5/4/102

### [A3] SMS Spam Detection and Classification to Combat Abuse in Telephone Networks Using Natural Language Processing
- 发表:arXiv preprint, 2024 (arXiv:2406.06578)
- 核心方法一句话:基于 NLP 流水线对电话网络中的垃圾/诈骗短信进行检测与分类。
- 优点:直接面向电信网络滥用场景;包含完整 NLP 预处理与分类对比。
- 局限:英文短信为主,中文场景需重新验证;预印本未经会议同行评审。
- 引用:*SMS Spam Detection and Classification to Combat Abuse in Telephone Networks Using Natural Language Processing*. (2024). arXiv:2406.06578.
- 链接:https://arxiv.org/abs/2406.06578

### [A4] Phishing Email Detection Using BERT and RoBERTa
- 发表:Computation (MDPI), 2026, 14(2):46
- 核心方法一句话:微调 BERT 与 RoBERTa 进行钓鱼邮件二分类并比较两者效果。
- 优点:直接验证预训练 Transformer 在钓鱼检测上的有效性,可作为你 BERT 基线的对照。
- 局限:任务为邮件而非通话;模型创新有限,主要是应用性微调。
- 引用:*Phishing Email Detection Using BERT and RoBERTa*. (2026). Computation, 14(2), 46. MDPI.
- 链接:https://www.mdpi.com/2079-3197/14/2/46

---

## 方向 B:机器学习/深度学习诈骗识别(SVM / CNN / LSTM / BERT)

### [B1] Enhanced Phishing Detection Using LSTM, CNN, and SVM Techniques
- 发表:Springer (会议论文集), 2025 (LNCS/LNNS 系列, 978-981-96-4148-2)
- 核心方法一句话:并行评估 LSTM、CNN、SVM 三类经典模型在钓鱼检测中的性能。
- 优点:正好覆盖你论文中 SVM/CNN/LSTM 三条基线,可直接作为方法对照引用。
- 局限:模型为标准实现,缺少架构创新;数据规模与领域细节披露有限。
- 引用:*Enhanced Phishing Detection Using LSTM, CNN, and SVM Techniques*. (2025). In LNNS, Springer. DOI:10.1007/978-981-96-4148-2_16.
- 链接:https://link.springer.com/chapter/10.1007/978-981-96-4148-2_16

### [B2] A Text Classification Model Combining Adversarial Training with Pre-trained Language Model and Neural Networks
- 发表:arXiv preprint, 2024 (arXiv:2411.06772)
- 核心方法一句话:将对抗训练(adversarial training)与预训练语言模型 + 神经网络结合以提升文本分类鲁棒性。
- 优点:把"对抗鲁棒性"与"文本分类"结合,与你论文鲁棒性主题高度契合;可作为防御方法引用。
- 局限:预印本;对抗训练对真实社工改写攻击的泛化性未充分验证。
- 引用:*A Text Classification Model Combining Adversarial Training with Pre-trained Language Model and Neural Networks*. (2024). arXiv:2411.06772.
- 链接:https://www.arxiv.org/abs/2411.06772

### [B3] Spam Detection Over Call Transcript Using Deep Learning
- 发表:Springer (会议论文集), 2021 (978-3-030-89880-9_10)
- 核心方法一句话:在电话通话转写文本(call transcript)上用深度学习做垃圾/骚扰检测。
- 优点:与"欺诈通话文本检测"任务几乎完全对齐,是直接相关工作;验证了通话转写文本的可分类性。
- 局限:数据集规模与语种受限;深度模型相对早期,未用预训练大模型。
- 引用:*Spam Detection Over Call Transcript Using Deep Learning*. (2021). In LNNS, Springer. DOI:10.1007/978-3-030-89880-9_10.
- 链接:https://link.springer.com/chapter/10.1007/978-3-030-89880-9_10

### [B4] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- 作者:Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
- 发表:NAACL-HLT 2019, pp. 4171–4186 (arXiv:1810.04805)
- 核心方法一句话:提出基于掩码语言建模的深度双向 Transformer 预训练范式,奠定下游微调分类的基础。
- 优点:你 BERT 分类器的根基性引用;迁移学习范式的奠基工作。
- 局限:原生模型对中文/领域文本需进一步预训练;对对抗扰动鲁棒性不足(见方向 D)。
- 引用:Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019, 4171–4186.
- 链接:https://aclanthology.org/N19-1423/

### [B5] RoBERTa: A Robustly Optimized BERT Pretraining Approach
- 作者:Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov
- 发表:arXiv preprint, 2019 (arXiv:1907.11692)
- 核心方法一句话:通过更大数据、更长训练、去除 NSP 等优化策略改进 BERT 预训练。
- 优点:常作为中文/英文诈骗文本分类的强基线;比 BERT 更稳健。
- 局限:计算成本高;同样存在对抗脆弱性。
- 引用:Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. arXiv:1907.11692.
- 链接:https://arxiv.org/abs/1907.11692

---

## 方向 C:LLM 在安全评测 / 对抗鲁棒性中的应用

### [C1] PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts
- 作者:Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, et al.
- 发表:arXiv 2023 (arXiv:2306.04528),后扩展为统一评测库(arXiv:2312.07910)
- 核心方法一句话:构建跨字符/词/句/语义多层级的对抗提示扰动,系统评估 LLM 在对抗提示下的鲁棒性。
- 优点:多层级扰动设计可借鉴到你的"语义改写攻击"评测框架;开源工具便于复现。
- 局限:针对通用任务提示扰动,非专门面向欺诈语义改写;评测以英文为主。
- 引用:Zhu, K., Wang, J., Zhou, J., Wang, Z., Chen, H., Wang, Y., et al. (2023). *PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts*. arXiv:2306.04528.
- 链接:https://arxiv.org/abs/2306.04528

### [C2] AdvGLUE: Adversarial GLUE — A Multi-Task Benchmark for Robustness Evaluation of Language Models
- 作者:Boxin Wang, Chejian Xu, Shuohang Wang, Zhe Gan, Yu Cheng, Jianfeng Gao, Ahmed Hassan Awadallah, Bo Li
- 发表:NeurIPS 2021 Datasets and Benchmarks Track (arXiv:2111.02840)
- 核心方法一句话:构建覆盖多种文本对抗攻击类型的多任务鲁棒性评测基准 AdvGLUE。
- 优点:对抗鲁棒性评测的标准基准之一;方法学上为"对测试集施加扰动评测鲁棒性"提供范式支撑(与你的实验设计同源)。
- 局限:基于通用 GLUE 任务,非诈骗领域;扰动多为词级而非社工语义级。
- 引用:Wang, B., Xu, C., Wang, S., Gan, Z., Cheng, Y., Gao, J., Awadallah, A. H., & Li, B. (2021). *Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models*. NeurIPS 2021 Datasets and Benchmarks. arXiv:2111.02840.
- 链接:https://arxiv.org/abs/2111.02840

### [C3] Evaluating the Efficacy of Large Language Models in Identifying Phishing Attempts
- 发表:arXiv preprint, 2024 (arXiv:2404.15485)
- 核心方法一句话:评估多个 LLM 直接作为零样本/少样本钓鱼识别器的有效性。
- 优点:验证 LLM 作判别器的可行性与边界,可作为你 LLM 分类基线引用;与你"LLM + 诈骗检测"主线直接相关。
- 局限:以提示工程为主,缺乏鲁棒性对抗评测;结果对提示敏感。
- 引用:*Evaluating the Efficacy of Large Language Models in Identifying Phishing Attempts*. (2024). arXiv:2404.15485.
- 链接:https://arxiv.org/abs/2404.15485

### [C4] Zero-Shot Spam Email Classification Using Pre-trained Large Language Models
- 发表:arXiv preprint, 2024 (arXiv:2405.15936)
- 核心方法一句话:利用预训练 LLM 在无监督标注下进行零样本垃圾/诈骗邮件分类。
- 优点:展示 LLM 零样本能力,可作为与监督 BERT 对照的另一类基线。
- 局限:零样本性能通常低于微调模型;领域适配性有限。
- 引用:*Zero-Shot Spam Email Classification Using Pre-trained Large Language Models*. (2024). arXiv:2405.15936.
- 链接:https://arxiv.org/html/2405.15936

### [C5] Robust ML-based Detection of Conventional, LLM-Generated, and Adversarial Phishing Emails Using Advanced Text Preprocessing
- 发表:arXiv preprint, 2025 (arXiv:2510.11915)
- 核心方法一句话:针对常规、LLM 生成、以及对抗性钓鱼邮件三类,设计鲁棒的 ML 检测与文本预处理流程。
- 优点:明确区分"对抗性/LLM 生成"钓鱼文本,与你的鲁棒性评测主题高度一致;提供对抗场景下的防御思路。
- 局限:预印本;预处理增强对真实社工改写的覆盖度需进一步验证。
- 引用:*Robust ML-based Detection of Conventional, LLM-Generated, and Adversarial Phishing Emails Using Advanced Text Preprocessing*. (2025). arXiv:2510.11915.
- 链接:https://arxiv.org/html/2510.11915

---

## 方向 D:社会工程攻击 / 文本对抗样本 / 改写攻击

### [D1] Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment (TextFooler)
- 作者:Di Jin, Zhijing Jin, Joey Tianyi Zhou, Peter Szolovits
- 发表:AAAI 2020 (arXiv:1907.11932)
- 核心方法一句话:提出 TextFooler,通过重要词定位 + 同义词替换生成保持语义的对抗样本,显著攻陷 BERT 等分类器。
- 优点:词级对抗攻击的奠基与强基线;直接支撑"分类器鲁棒性脆弱"的论点;可用作你改写攻击的对照方法。
- 局限:同义词替换偏低层扰动,语义/句法自然度有限;与社工语义改写(高层策略)不同层级。
- 引用:Jin, D., Jin, Z., Zhou, J. T., & Szolovits, P. (2020). *Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment*. AAAI 2020. arXiv:1907.11932.
- 链接:https://ojs.aaai.org/index.php/AAAI/article/view/6311

### [D2] BERT-ATTACK: Adversarial Attack Against BERT Using BERT
- 发表:EMNLP 2020 (arXiv:2004.01970)
- 核心方法一句话:用预训练 BERT 的掩码语言模型生成上下文感知的替换词,对 BERT 分类器实施高质量对抗攻击。
- 优点:对抗样本流畅度高于同义词替换;说明强模型亦可被自身机制攻击,鲁棒性论证有力。
- 局限:仍为词级替换;计算开销大;非社工策略层面的改写。
- 引用:Li, L., Ma, R., Guo, Q., Xue, X., & Qiu, X. (2020). *BERT-ATTACK: Adversarial Attack Against BERT Using BERT*. EMNLP 2020. arXiv:2004.01970.
- 链接:https://ar5iv.labs.arxiv.org/html/2004.01970

### [D3] TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP
- 作者:John X. Morris, Eli Lifland, Jin Yong Yoo, Jake Grigsby, Di Jin, Yanjun Qi
- 发表:EMNLP 2020 (System Demonstrations)
- 核心方法一句话:提供统一的 NLP 对抗攻击/数据增强/对抗训练框架,集成多种攻击配方。
- 优点:可直接用于复现 TextFooler/BERT-Attack 等攻击,工程价值高;支持构造你的鲁棒性评测对照实验。
- 局限:内置攻击多为词/字符级,缺少社工语义级改写策略(需自行扩展)。
- 引用:Morris, J. X., Lifland, E., Yoo, J. Y., Grigsby, J., Jin, D., & Qi, Y. (2020). *TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP*. EMNLP 2020 (Demo).
- 链接:https://aclanthology.org/2020.emnlp-demos.16/

### [D4] How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs
- 作者:Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi
- 发表:ACL 2024 (Long Paper) (arXiv:2401.06373)
- 核心方法一句话:从社会科学说服理论出发,构建 40 类说服技巧(persuasion taxonomy),将有害请求改写为"有说服力的对抗提示"以越狱 LLM。
- 优点:把"社会工程/说服策略"系统化为可操作的改写分类法,是 Fraud-R1 类社工诱导的理论与方法源头;与你的"信任建立/紧迫感/情感操纵"改写策略直接同源。
- 局限:面向越狱安全而非欺诈检测;说服策略的自动化生成质量参差。
- 引用:Zeng, Y., Lin, H., Zhang, J., Yang, D., Jia, R., & Shi, W. (2024). *How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs*. ACL 2024. arXiv:2401.06373.
- 链接:https://aclanthology.org/2024.acl-long.773

### [D5] Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text
- 发表:arXiv preprint, 2025 (arXiv:2506.07001)
- 核心方法一句话:提出通用的对抗改写(paraphrasing)攻击,通过语义保持的改写规避检测器。
- 优点:聚焦"改写攻击"这一你论文核心操作,论证语义级改写对检测器的普适威胁。
- 局限:面向 AI 文本检测而非诈骗分类;预印本。
- 引用:*Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text*. (2025). arXiv:2506.07001.
- 链接:https://arxiv.org/html/2506.07001

### [D6] Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models
- 发表:arXiv preprint, 2024 (arXiv:2410.20940)
- 核心方法一句话:用语言模型生成对抗样本来攻击虚假信息检测分类器。
- 优点:与你"用 LLM 改写测试集攻击分类器"的实验范式高度一致,方法学可直接借鉴。
- 局限:领域为 misinformation 而非电信诈骗;预印本。
- 引用:*Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models*. (2024). arXiv:2410.20940.
- 链接:https://arxiv.org/html/2410.20940v2

---

## 补充:电信诈骗专门数据集/方法(强相关,建议纳入相关工作)

### [F1] TeleAntiFraud-28k: An Audio-Text Slow-Thinking Dataset for Telecom Fraud Detection
- 发表:arXiv preprint, 2025 (arXiv:2503.24115)
- 核心方法一句话:构建首个面向电信诈骗检测的音频-文本"慢思考(slow-thinking)"数据集(约 28k),支持多模态推理式检测。
- 优点:电信诈骗领域稀缺的专门数据集,与你的数据集主题直接对齐;含推理链标注。
- 局限:中文音频转写质量与领域偏置;预印本。
- 引用:*TeleAntiFraud-28k: An Audio-Text Slow-Thinking Dataset for Telecom Fraud Detection*. (2025). arXiv:2503.24115.
- 链接:https://arxiv.org/html/2503.24115v2

### [F2] Telecom Fraud Detection Based on Large Language Models: A Multi-Role, Multi-Layer Prompting Strategy
- 发表:Applied Sciences (MDPI), 2026, 16(1):544
- 核心方法一句话:设计多角色、多层级提示策略,驱动 LLM 进行电信诈骗检测。
- 优点:直接结合"LLM + 电信诈骗检测",提示策略可借鉴;与你 LLM 主线契合。
- 局限:提示工程依赖性强;缺乏对抗改写鲁棒性测试。
- 引用:*Telecom Fraud Detection Based on Large Language Models: A Multi-Role, Multi-Layer Prompting Strategy*. (2026). Applied Sciences, 16(1), 544. MDPI.
- 链接:https://www.mdpi.com/2076-3417/16/1/544/htm

### [F3] Exploring LLM-based Real-time Detection of Phone Scams
- 发表:arXiv preprint, 2025 (arXiv:2502.03964)
- 核心方法一句话:探索基于 LLM 的实时电话诈骗检测系统设计。
- 优点:聚焦"诈骗电话"实时检测场景,与你课题最贴近;讨论部署可行性。
- 局限:系统性评测有限;预印本。
- 引用:*Exploring LLM-based Real-time Detection of Phone Scams*. (2025). arXiv:2502.03964.
- 链接:https://arxiv.org/html/2502.03964v1

### [F4] System Report for CCL23-Eval Task 6: A Method for Telecom Network Fraud Case Classification Based on Two-stage Training Framework and Within-task Pretraining
- 发表:CCL 2023 (Evaluation Track), ACL Anthology 2023.ccl-3.23
- 核心方法一句话:面向中文电信网络诈骗案件分类,采用两阶段训练 + 任务内预训练框架。
- 优点:中文电信诈骗分类的评测任务正式论文,与你中文场景直接对齐;方法可复现。
- 局限:任务为案件类型分类而非诱导鲁棒性;领域窄。
- 引用:*System Report for CCL23-Eval Task 6: A Method for Telecom Network Fraud Case Classification Based on Two-stage Training Framework and Within-task Pretraining*. (2023). CCL 2023. ACL Anthology 2023.ccl-3.23.
- 链接:https://aclanthology.org/2023.ccl-3.23

---

## 方向覆盖小结(供主 agent 组织"相关工作"章节)

- 共 21 篇真实可考文献,覆盖 A–E 全部方向 + 电信诈骗专门补充(F)。
- 顶会/正式发表:BERT (NAACL'19)、TextFooler (AAAI'20)、BERT-ATTACK (EMNLP'20)、TextAttack (EMNLP'20)、AdvGLUE (NeurIPS'21)、Zeng et al. 说服越狱 (ACL'24)、Fraud-R1 (ACL'25 Findings)、CCL23-Eval Task6 (CCL'23)。
- 建议论文叙事线:监督检测基线(B4/B5/A 系列)→ 电信诈骗专门工作(F 系列,凸显数据稀缺与场景特性)→ 分类器对抗脆弱性(D1/D2/D3)→ 社工/说服策略改写攻击(D4/D5/D6 + E1)→ LLM 鲁棒性评测范式(C1/C2/C5)。Fraud-R1(E1)与 Zeng et al.(D4)是你"社工诱导改写"方法论的两大直接源头,建议重点对比并指出差异:你做的是判别式分类器鲁棒性(改写测试集),而 Fraud-R1 测的是生成式 LLM 抗诱导能力。
- 需提醒用户:标注 "arXiv preprint" 的若用于正式参考文献,建议二次核实是否已被会议/期刊收录,并补全作者全名(部分检索结果未返回完整作者列表,BibTeX 生成前需补齐 A1/A2/A3/A4/B1/B2/B3/C3/C4/C5/D5/D6/F1/F2/F3 的作者字段)。

Sources:
- https://arxiv.org/abs/2502.12904
- https://github.com/kaustpradalab/Fraud-R1
- https://arxiv.org/abs/2308.05476
- https://www.mdpi.com/2624-800X/5/4/102
- https://arxiv.org/abs/2406.06578
- https://www.mdpi.com/2079-3197/14/2/46
- https://link.springer.com/chapter/10.1007/978-981-96-4148-2_16
- https://www.arxiv.org/abs/2411.06772
- https://link.springer.com/chapter/10.1007/978-3-030-89880-9_10
- https://aclanthology.org/N19-1423/
- https://arxiv.org/abs/1907.11692
- https://arxiv.org/abs/2306.04528
- https://arxiv.org/abs/2111.02840
- https://arxiv.org/abs/2404.15485
- https://arxiv.org/html/2405.15936
- https://arxiv.org/html/2510.11915
- https://ojs.aaai.org/index.php/AAAI/article/view/6311
- https://ar5iv.labs.arxiv.org/html/2004.01970
- https://aclanthology.org/2020.emnlp-demos.16/
- https://aclanthology.org/2024.acl-long.773
- https://arxiv.org/html/2506.07001
- https://arxiv.org/html/2410.20940v2
- https://arxiv.org/html/2503.24115v2
- https://www.mdpi.com/2076-3417/16/1/544/htm
- https://arxiv.org/html/2502.03964v1
- https://aclanthology.org/2023.ccl-3.23