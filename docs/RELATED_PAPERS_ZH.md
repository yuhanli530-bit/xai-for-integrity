# XAI for Image Integrity：相关论文与阅读路线

检索日期：2026-11-06。范围：图像生成/篡改取证、视觉解释忠实性、视觉 SAE 与概念解释。条目按论文官方页面、摘要及必要的项目页面核对；这是定向初筛，不是全面系统综述，未逐篇复现实验。预印本以 arXiv 标识，发表状态以链接页面为准。

## 优先级 P0：直接影响选题与方法定位

1. **Kang et al., LEGION: Learning to Ground and Explain for Synthetic Image Detection**，2025，[论文](https://arxiv.org/abs/2503.15264)，[项目](https://opendatalab.github.io/LEGION/)。提出 SynthScars 和联合异常检测、分割、文本解释的 LEGION。我们目前的训练代码、数据与 G/P 方法均基于它；先读数据构造、解释与 mask 对应、损失和评测，再讨论新增贡献。原文的性能不是我们的结果。
2. **Xu et al., FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models**，ICLR 2025，[论文](https://arxiv.org/abs/2410.02761)。同时输出鉴伪依据与篡改位置，使用文本指导定位，提出 MMTD-Set；是解释和定位联合建模的直接相关工作。其 GPT-4o 辅助描述与 SynthScars 人工专家标注来源不同，比较时需区别样本域和标注。
3. **Ding et al., EDCT-Bench: Uncovering Faithfulness Gaps in VLMs via Explanation-Driven Counterfactual Testing**，2026 预印本，[论文](https://arxiv.org/abs/2609.17953)。提取解释中的视觉概念，对其进行验证过的最小编辑，再检查答案及解释变化。它与我们计划的“验证解释依赖其提及证据”高度接近。不能把“解释引导干预”本身写成新贡献；需逐项比较任务、干预有效性标注、对照组、指标和模型训练方式。较早的 [EDCT 2025 版本](https://arxiv.org/abs/2510.00047) 同样应对照。
4. **Gao et al., Toward Generalizable Forgery Detection and Reasoning**，IEEE TIP，2026 页面标注已接收，[论文](https://arxiv.org/abs/2503.21210)。FakeReasoning 使用 CLIP 与 DINO 双分支、伪造感知特征融合和分类概率映射，连接鉴伪与推理。是针对图像取证表征与泛化的重要方法对照；并不自动等价于验证解释忠实性。
5. **Xu et al., Explainable Deepfake Detection with Feature-robust Augmentation and Evidence-grounded Explanation Optimization**，2026 预印本，[论文](https://arxiv.org/abs/2608.20913)。用涉及证据遗漏或无关信息的解释偏好对来优化解释，是“证据约束解释”非常近的竞争工作；需要对照其训练目标、标注和评估，不能只以使用证据监督声称新颖。

## P1：第二条方法线与非 SAE 选择

6. **Pach et al., Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models**，NeurIPS 2025，[论文](https://arxiv.org/abs/2504.02821)，[代码](https://github.com/ExplainableML/sae-for-vlm)。研究 CLIP 视觉特征中的 SAE 特征，并干预视觉编码器以影响下游多模态模型输出。是我们 SAE 干预路线的直接技术参考，但不证明其特征是取证证据或兼容当前 encoder。
7. **Lim et al., Sparse autoencoders reveal selective remapping of visual concepts during adaptation**，ICLR 2025，[论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3d5b603d631d595f56bc36b373458b27-Abstract-Conference.html)，[代码](https://github.com/dynamical-inference/patchsae)。提出 PatchSAE，为 CLIP ViT 提供 patch 级稀疏概念与空间归因。可参考如何把稀疏特征映射到位置；空间激活不等于忠实的伪造证据。
8. **Stevens et al., Interpretable and Testable Vision Features via Sparse Autoencoders**，2025 预印本 v2，[论文](https://arxiv.org/abs/2502.06755)，[SAE-V](https://osu-nlp-group.github.io/SAE-V/)。讨论如何用视觉 SAE 进行可证伪的 patch 级特征干预。v1 标题为 *Sparse Autoencoders for Scientifically Rigorous Interpretation of Vision Models*；可参考其重建和干预验证。
9. **Kravets et al., Interpretability Transfer from Language to Vision via Sparse Autoencoders**，ICML 2026，[论文](https://proceedings.mlr.press/v306/kravets26a.html)。VISTA 将视觉 token 投影到已有语言 SAE 空间。可作为视觉 SAE 之外的跨模态备选；是否适合我们的投影层和伪造细节仍待检查，不能假设现成特征可直接迁移。
10. **Koh et al., Concept Bottleneck Models**，ICML 2020，[论文](https://proceedings.mlr.press/v119/koh20a.html)。先预测人可理解概念，再用概念做决策并允许人工干预。对于“不用 SAE”的路线，可用取证概念标签做概念探针或瓶颈，与稀疏特征和直接区域干预比较；需评估标签成本、概念遗漏及信息泄漏。

## P1：外部评测与数据

11. **Ye et al., LOKI: A Comprehensive Synthetic Data Detection Benchmark using Large Multimodal Models**，ICLR 2025，[论文](https://arxiv.org/abs/2410.09732)，[项目](https://opendatalab.github.io/LOKI/)。覆盖多模态合成数据的真伪判断与细粒度解释任务，可评估外部图像子集；论文摘要不证明该子集具备本项目所需的像素 mask、来源独立或可用许可，使用前需逐项核实。
12. **Narang et al., XPlainVerse: A Million-Scale Benchmark for Explainable Deepfake Detection**，2026 预印本，[论文](https://arxiv.org/abs/2607.03562)，[数据卡](https://huggingface.co/datasets/Abhijeet8901/XPlainVerse)。提供技术和简化两种解释，提出 EntityScore/EvidenceScore；其评估实体与证据描述，不应直接视为模型因果忠实性的证明。下载前核对协议、来源重叠、数据字段和访问条件；公开挑战的实际发布规模应以其数据卡为准。

## 建议阅读顺序与实验含义

- **先读 LEGION → FakeShield → FakeReasoning**：画出各自的图像、mask、文本、模型路径和监督来源，准确说清我们的 G 与既有联合定位方法差在哪里。
- **再读 EDCT-Bench 与证据偏好优化论文**：逐项做对照表，确认干预目标、是否有不相关区域对照、输入有效性、人工标注、测量单位以及跨数据集表现。当前项目的三输入设计是计划中的候选差异，不是已证实的新颖性或效果。
- **最后读 Pach → PatchSAE → SAE-V → VISTA/CBM**：先核实当前模型的视觉编码器版本、hook 层、patch 网格与重建误差；再比较 SAE、概念探针、直接区域/Token 干预和匹配随机特征，避免仅凭特征可视化下结论。
- 做实验时保留三种结论的区别：定位是否准确、解释是否依赖证据、解释是否帮助人审查。现有单 seed 已使用过的测试集结果只能作为探索性证据。

## 待深化，不应当作已完成

对最相关论文逐段核对训练/评测细节和代码、核实数据许可证及数据源重叠、补完整系统检索与引用追踪；冻结外部评测和对照方案后才能评估具体差异是否足以构成贡献。尚无本项目 SAE、独立数据或解释忠实性的完成结果。
