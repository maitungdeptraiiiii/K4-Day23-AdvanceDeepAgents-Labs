# Reinforcement Learning for LLM Reasoning: A Comprehensive Survey

## TL;DR
- DeepSeek-R1 demonstrated that pure reinforcement learning without supervised fine-tuning can elicit reasoning capabilities, achieving 71.0% on AIME 2024 (pass@1) and matching OpenAI o1 performance [1].
- Group Relative Policy Optimization (GRPO) has emerged as the dominant RL algorithm for reasoning tasks, eliminating the need for a critic model while theoretically amplifying success probability through its U-statistic policy gradient [2][3].
- Verifiable outcome rewards (rule-based correctness checks) fundamentally differ from learned human-preference rewards, avoiding reward hacking but creating challenges like length bias and spurious correlations [4][3].
- Cross-domain transfer is highly asymmetric: math-trained models generalize well to code and logic via RL, but SFT-induced representation drift causes catastrophic forgetting on general capabilities [5].
- Test-time compute scaling — where models learn to allocate inference budget adaptively — can be 4× more efficient than best-of-N sampling, with generative verifiers trained alongside reasoners enabling 8–32× more efficient scaling [6][7].

## Background

Reinforcement learning for large language model reasoning represents a paradigm shift from supervised learning toward reward-driven optimization. Traditional LLM training relied on next-token prediction over static datasets, producing models that excel at pattern completion but struggle with multi-step reasoning tasks requiring deliberate thought. The transition began with RLHF (Reinforcement Learning from Human Feedback), which used PPO to align models with human preferences via learned reward models [8]. However, RLHF was primarily designed for dialogue alignment rather than reasoning enhancement.

The field underwent a fundamental transformation in September 2024 when OpenAI released o1, demonstrating that large-scale RL could teach models to "think before answering" by producing long internal chains of thought [9]. This established a new scaling paradigm where both train-time compute (RL training duration) and test-time compute (time spent generating reasoning traces) independently improve performance. By December 2024, the OpenAI o1 System Card documented emergent behaviors including error correction, strategy switching, and breaking down complex steps into simpler ones [10].

The pivotal moment arrived in January 2025 with DeepSeek-R1-Zero, which proved that reasoning capabilities emerge purely through RL without any intermediate supervised fine-tuning step [1]. Using Group Relative Policy Optimization with rule-based rewards, the model naturally developed self-reflection, verification, and dynamic strategy adaptation — behaviors not explicitly programmed but discovered during training. This finding validated that RL, not SFT, is the mechanism that elicits genuine reasoning patterns.

## Algorithm Families

### Policy Gradient Methods

Policy gradient approaches dominate current RL-for-reasoning practice. PPO, the original workhorse of RLHF, requires four components: a policy model, a value function (critic), a reference model, and a reward model. For reasoning tasks, this architecture becomes prohibitively expensive because long chain-of-thought outputs demand massive GPU memory and slow rollout speeds. Truncated PPO addresses this by reducing the computational overhead inherent in PPO's on-policy nature, particularly critical as response lengths increase [11].

GRPO (Group Relative Policy Optimization), introduced in DeepSeekMath [2], eliminates the critic entirely. It samples multiple outputs per prompt (typically G=64) and uses their average reward as the baseline, computing advantages from group-relative scores with whitening normalization. The KL divergence penalty is added directly to the loss rather than the reward, simplifying the optimization landscape. Recent theoretical analysis establishes that GRPO's policy gradient is inherently a U-statistic, making it asymptotically equivalent to an oracle policy gradient algorithm with access to a true value function [12]. This derivation also yields a universal scaling law for optimal group size depending only on training data and model architecture.

With verifiable rewards, GRPO can be reformulated as a KL-regularized contrastive loss between old-policy samples. Theoretical analysis proves that GRPO amplifies the probability of success: the fixed point p* exceeds p_ref under mild assumptions, explaining empirical gains such as increasing GSM8K pass rates from 21% to 37.5% after GRPO training on Qwen2.5-0.5B [3].

ReMax [13] offers another critic-free alternative built on REINFORCE. It exploits three properties unique to LLM alignment: fast simulation, deterministic transitions, and trajectory-level rewards. Using greedy baseline value for variance reduction, ReMax requires only 6 lines of code versus PPO's 30+, saves ~46% GPU memory on 7B models, and runs ~1.6× faster while achieving competitive AlpacaEval win rates.

RLOO (REINFORCE Leave-One-Out) treats the entire model completion as a single action rather than token-by-token actions [14]. Its leave-one-out advantage construction ensures statistical independence between each sample's advantage estimate and the baseline. RLOO requires only 3 model copies in memory (policy, reference, reward) and integrates KL divergence into the modified reward. Cohere's implementation showed a 6.9B checkpoint achieving 78.7% preferred rate with GPT-4-as-judge evaluation.

### Preference Optimization Methods

Direct Preference Optimization (DPO) [8] transforms the KL-constrained reward maximization problem into a binary cross-entropy classification loss through an analytical mapping between reward functions and optimal policies. While DPO eliminates explicit reward modeling, it relies on pre-collected preference pairs rather than online interaction, limiting its applicability to reasoning where exploration is essential. Nevertheless, DPO provides a useful theoretical foundation — viewing RFT, DPO, PPO, and GRPO as different levels of direct or simplified RL techniques [2].

Preference optimization methods face a fundamental limitation for reasoning: they require curated preference data that captures nuanced reasoning quality, whereas reasoning rewards are often binary (correct/incorrect final answer). This makes online RL with verifiable rewards more practical for reasoning-specific training.

### Reward Modeling Approaches

Two fundamentally different reward paradigms exist for reasoning RL:

**Verifiable (sparse) rewards** use rule-based correctness checks against ground-truth answers. These provide precise feedback for mathematical, coding, and logical domains where solutions can be automatically verified [1]. They avoid reward hacking but create challenges: length bias (longer CoTs get more opportunities to contain correct sub-steps), spurious correlations, and sparse signals that make credit assignment difficult across multi-step reasoning.

**Learned dense rewards** employ Process-supervised Reward Models (PRMs) that score individual reasoning steps. While theoretically providing finer-grained credit assignment, PRMs suffer severe reward hacking: LLMs exploit the reward model by generating numerous correct but unnecessary reasoning steps, achieving excessively high returns through repetition of simple steps [4]. Clip and Delta mechanisms were proposed to upper-bound process rewards and stabilize training. Outcome-supervised Reward Models (ORMs) show limited improvement over sparse rewards alone in online RL settings.

**Generative verifiers**, as explored in RLV, jointly train the LLM as both reasoner and verifier using RL-generated data, achieving over 20% boost in MATH accuracy with parallel sampling and enabling 8–32× more efficient test-time compute scaling [7]. This approach occupies a middle ground between task-agnostic prompting and costly dedicated verifiers.

### Tree Search and Self-Play Integration

Monte Carlo Tree Search (MCTS) integration with RL creates powerful reasoning loops. rStar [12] decouples reasoning into a self-play mutual generation-discrimination process between two small language models. The target SLM augments MCTS with human-like reasoning actions to construct higher-quality trajectories, while a discriminator SLM verifies each trajectory. Mutually agreed-upon trajectories provide training signals without external reward models.

RLSP (Reinforcement Learning via Self-Play) frames reasoning as guided search, combining supervised fine-tuning with synthetic data followed by self-play RL where the model competes against itself [15]. This draws connections to self-consistency, PRMs, and AlphaZero-style approaches, aiming to uncover the algorithmic framework for search-based reasoning in LRMs.

## Domain Applications

### Mathematics

Mathematical reasoning represents the most mature application domain for RL-trained models. The AIME 2024 benchmark shows dramatic improvements: OpenAI o1 achieved 74% pass@1 (single sample), 83% consensus among 64 samples, and 93% re-ranking 1000 samples with a learned scoring function — placing above USA Mathematical Olympiad cutoff [9]. DeepSeek-R1-Zero increased from 15.6% to 71.0% on AIME 2024 using pure RL [1].

The MATH dataset and GSM8K serve as standard benchmarks. DeepSeekMath-RL 7B achieved 88.2% on GSM8K and 51.7% on MATH using only instruction-tuning data [2]. RLV achieves over 20% boost in MATH accuracy with parallel sampling [7]. The CoT-Pass@K metric reveals that RLVR extends reasoning boundaries beyond mere sampling efficiency: once LLMs establish strong knowledge and logic priors, the GRPO gradient increases probability of generating more correct CoTs even with only answer-correctness rewards [3].

### Code Generation

Competitive programming benchmarks demonstrate RL's impact on code reasoning. OpenAI o3 reaches CodeForces Elo rating of 2724 (99.8th percentile), surpassing o1-ioi's 2214 (98th percentile) [16]. o3 discovers advanced test-time strategies autonomously, writing brute-force solutions to verify outputs against optimized algorithmic implementations as a self-imposed validation mechanism. At IOI 2024, o3 achieves gold medal scores without hand-crafted domain-specific strategies.

DeepSeek-R1-Distill-Qwen-32B achieves CodeForces rating of 1691 and LiveCodeBench pass@1 of 65.9%, outperforming OpenAI o1-mini across benchmarks [1]. The distilled 70B Llama-based variant achieves 70.0% AIME and 94.5% MATH-500, competitive with much larger proprietary models [16]. ARTIST demonstrates up to 22% absolute improvement on multi-turn function calling benchmarks by interleaving tool queries and outputs within reasoning chains using GRPO [17].

### Logical Reasoning and Theorem Proving

Formal proof assistants like Lean provide clear verification signals that enable effective RL training, overcoming the sparse supervision limitation of natural-language-only approaches [18]. Seed-Prover proposes lemma-style whole-proof reasoning that iteratively refines proofs based on Lean feedback and proved lemmas [18]. ProofNet++ combines LLMs with symbolic proof tree supervision, RL loops using verifiers as reward functions, and iterative self-correction, showing significant improvement on miniF2F and HOL Light benchmarks [19].

Mathesis introduces an end-to-end theorem proving pipeline processing informal natural language problem statements, using RL to enhance formalization ability with a novel LeanScore metric [20]. RuleReasoner enhances rule-based reasoning in small models through dynamic domain sampling with RL, achieving superior performance compared to large models [21].

### Cross-Domain Transfer

Transferability depends critically on the fine-tuning paradigm. Controlled studies on Qwen3-14B show that RL-tuned models generalize well to non-math domains (scientific QA, agent planning, instruction-following), while SFT-tuned models often exhibit catastrophic forgetting [5]. Latent-space PCA and token-distribution KL-divergence analyses reveal SFT induces substantial representation and output drift, while RL preserves general-domain structure. Credit assignment focuses updates on task-relevant tokens while negative gradients reduce reinforcing uninformative content.

Domain interactions show asymmetric effects: training on other domains improves math reasoning by ~25% accuracy but yields negligible transfer to logic and puzzle domains [22]. Training order matters significantly — math→science achieves 83%/41% accuracy on math/science respectively, while science→math degrades to 77%/25%. Mixed-domain training uniformly combining all six domains (Math, Code, Science, Logic, Simulation, Tabular) performs on par or better than single-domain RL [23].

## Open-Source Ecosystem and Frameworks

### Model Releases

DeepSeek-R1's paradigm shift — demonstrating pure RL without SFT can elicit reasoning — catalyzed the open-source ecosystem [1]. Distilled variants include Qwen2.5-based models (1.5B, 7B, 8B, 14B, 32B, 70B) and Llama3-based distillations, all licensed under MIT for commercial use [16]. The 32B distill achieves 72.6% on AIME 2024 and 94.3% on MATH-500, setting records among dense open models.

Empirical studies comparing base models and training approaches confirm that RL training and tool manipulation enhance reasoning performance [24]. Anyscale's architecture comparison describes how most libraries standardize around PPO and GRPO algorithms with sharding backends including Hugging Face Trainer, FSDP, DeepSpeed, and Megatron [25].

### Training Frameworks

veRL (Volcano Engine Reinforcement Learning), initiated by ByteDance Seed team, has emerged as the most mature training framework [5]. It supports FSDP, FSDP2, and Megatron-LM for training; vLLM, SGLang, and HF Transformers for rollout generation. veRL implements diverse RL algorithms: PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO, KL_Cov & Clip_Cov. It scales up to 671B models and hundreds of GPUs with expert parallelism and supports multi-GPU LoRA RL to save memory.

A comprehensive survey of 16 open-source RL libraries distinguishes colocated mode (inference + training on same GPUs, simpler but no true overlap) from disaggregated mode (separate GPU pools enabling async overlap where generation and training run in parallel) [23]. Notable libraries include TRL (optimized for simplicity and HF ecosystem integration), Verifiers (multi-turn RL with environments), SkyRL (agentic settings with sync or async pipelining), and slime (opinionated on sglang + megatron for big MoE models).

LoRA reduces trainable parameters by 99%+, halves peak activation memory, and enables adapter-only weight sync (~50 MB vs ~100–500 ms NCCL broadcast); however, MoE LoRA remains an emerging challenge [23]. EfficientRollout addresses rollout generation as a dominant latency bottleneck through system-aware self-speculative decoding, reducing rollout latency by up to 19.6% [26].

### Infrastructure Challenges

Key infrastructure challenges include KV cache management during long rollouts, balancing inference/training resource allocation between colocated and disaggregated deployments, and handling stragglers in long-horizon tasks [23][25]. Ray is widely used as orchestrator for scheduling, communication, fault tolerance, and autoscaling due to its flexibility and heterogeneous computing support [25].

## Trends and Emerging Directions

### Test-Time Scaling and Inference Compute

Test-time compute scaling has become a central research direction. Snell et al. analyze two mechanisms: searching against dense process-based verifier reward models (PRMs), and adaptively updating the model's distribution over responses at test time [6]. The compute-optimal strategy allocates test-time compute adaptively per prompt difficulty, improving efficiency by more than 4× compared to best-of-N baselines. In FLOPs-matched evaluations, test-time compute can outperform a 14× larger model on problems where a smaller base model attains non-trivial success rates.

Multi-round thinking approaches iteratively refine model reasoning by leveraging previous answers as prompts for subsequent rounds [27]. Overclocking LLM Reasoning manipulates progress encoding mechanisms to regulate reasoning depth, improving accuracy and reducing inference time simultaneously [28]. ConciseR addresses the persistent overthinking phenomenon in state-of-the-art reasoning models — excessive redundancy or repetitive thinking in long CoT responses — through staged RL optimization [29].

### Process Rewards and Step-Level Credit Assignment

Process reward models provide feedback at each step of multi-step reasoning traces, potentially improving credit assignment over outcome reward models that only provide feedback at the final step [30]. However, collecting dense per-step human labels does not scale. Automated approaches for designing process verifiers aim to make PRMs practical for scaling RL training without expensive human annotation.

### Agentic Reasoning and Tool Use

ARTIST couples agentic reasoning, RL, and tool integration in a unified framework [17]. Models autonomously decide when, how, and which tools to invoke within multi-turn reasoning chains, using outcome-based RL (GRPO) without step-level supervision. Tool usage is treated as a first-class operation interleaved within reasoning chains.

WMAct presents world-model internalization through efficient interaction and active reasoning, liberating models from structured reasoning to shape thinking through doing [31]. Interaction frequency annealing progressively reduces maximum allowed interaction turns, compelling the model to internalize environmental dynamics. Experiments on Sokoban, Maze, and Taxi show WMAct resolves tasks in a single turn that previously required multiple interactions.

Thinking vs. Doing agents scale test-time interaction for web agent performance, balancing exploration and exploitation without adding per-step compute [32]. Play to Generalize post-trains multimodal LLMs with RL on arcade-like games, enhancing multimodal reasoning abilities without domain-specific data [33].

### Safety and Alignment Trade-offs

Deliberative alignment directly teaches safety specifications and trains models to explicitly recall and reason over them before answering [34]. Applied to o-series models, it achieves Pareto improvement by reducing both under- and overrefusals while substantially improving jailbreak robustness (StrongREJECT goodness@0.1 of 0.88). Both process-supervision during SFT and outcome-based RL play critical roles.

However, the "Safety Tax" trade-off emerges: sequential production pipelines (reasoning training via RL followed by safety alignment) degrade reasoning capability. SafeChain reduces average reasoning accuracy by 7.09%, while DirectRefusal causes 30.91% reduction [35]. Harmful scores are reduced by 59.6% (DirectRefusal) and 29.1% (SafeChain), showing safety can be restored but at a cost to reasoning performance.

### Scaling Laws and Fundamental Questions

Recent work examines whether RL expands an LLM's reasoning boundary or merely reweights its existing reasoning space [36]. Both cross-domain gains and forgetting occur after RL training, with coverage at large sampling budgets increasing on some tasks and decreasing on others. Fundamental scaling laws for RL training of reasoning capabilities remain under investigation.

The "Illusion of Thinking" paper evaluates LRMs across varying task complexities using controllable puzzle environments, finding limitations in exact computation and inconsistent reasoning behaviors [37]. A NIPS 2025 study probes reasoning capability boundaries across diverse model families and finds that while RLVR improves sampling efficiency at small k, base models achieve higher pass@k scores when k is large — suggesting current RLVR methods' reasoning abilities originate from and are bounded by the base model [7].

Distillation offers an alternative pathway: Merge-of-Thought distillation alternates between teacher-specific SFT branches and weight-space merging, unifying multiple teachers' reasoning abilities into one student. With only ~200 high-quality CoT samples, MoT on Qwen3-14B surpasses DeepSeek-R1, Qwen3-30B-A3B, Qwen3-32B, and OpenAI-O1 on competition math benchmarks [38]. Through the Valley highlights that small language models experience significant performance declines when trained on long chain-of-thought data due to error accumulation, impacting downstream RL performance [39].

Emerging directions include cooperative multi-agent reasoning with TRACER's turn-level regret matching [40], Reinforcement Pre-Training applying RL at pre-training scale rather than only post-training [41], Magistral's scalable RL pipeline that does not require existing RL traces [42], and Mathesis's RL-enhanced formalization for automated theorem proving [20].

## References
[1] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arxiv. https://arxiv.org/abs/2501.12948 (2025-01-22)
[2] DeepSeekMath: Pushing the Limits of Mathematical Reasoning. arxiv. https://arxiv.org/abs/2402.03300 (2024-04-27)
[3] Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs. arxiv. https://arxiv.org/abs/2506.14245 (2025-06-18)
[4] On Designing Effective RL Reward at Training Time for LLM Reasoning. arxiv. https://arxiv.org/abs/2410.15115 (2024-10-19)
[5] veRL: Volcano Engine Reinforcement Learning for LLMs. web. https://github.com/verl-project/verl (2025)
[6] Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. arxiv. https://arxiv.org/abs/2408.03314 (2024-08-06)
[7] Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?. web. https://papers.nips.cc/paper_files/paper/2025/hash/537d5aa768c2d534016a4d06f87bc8fb-Abstract-Conference.html (2025)
[8] Direct Preference Optimization: Your Language Model is Secretly a Reward Model. arxiv. https://arxiv.org/abs/2305.18290 (2023-05-29)
[9] Learning to Reason with LLMs — OpenAI o1 Blog Post. web. https://openai.com/index/learning-to-reason-with-llms/ (2024-09-12)
[10] OpenAI o1 System Card. web. https://arxiv.org/html/2412.16720v1 (2024-12-01)
[11] Truncated Proximal Policy Optimization. arxiv. https://arxiv.org/abs/2506.15050 (2025-06-18)
[12] Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers (rStar). arxiv. https://arxiv.org/abs/2408.06195 (2024-08-12)
[13] ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models. arxiv. https://arxiv.org/abs/2310.10505 (2023-10-16)
[14] REINFORCE Leave-One-Out (RLOO) — Hugging Face Blog. web. https://github.com/huggingface/blog/blob/main/putting_rl_back_in_rlhf_with_rloo.md (2024)
[15] On the Emergence of Thinking in LLMs I: Searching for the Right Intuition — RL via Self-Play. arxiv. https://arxiv.org/abs/2502.06773 (2025-02-10)
[16] deepseek-ai/DeepSeek-R1 — GitHub Repository. web. https://github.com/deepseek-ai/DeepSeek-R1/?tab=readme-ov-file (2025-01-20)
[17] ARTIST: Agentic Reasoning and Tool Integration for LLMs. arxiv. https://arxiv.org/abs/2505.01441 (2025-05-01)
[18] Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving. arxiv. https://arxiv.org/abs/2507.23726 (2025-07-31)
[19] ProofNet++: A Neuro-Symbolic System for Formal Proof Verification with Self-Correction. arxiv. https://arxiv.org/abs/2505.24230 (2025-05-30)
[20] Mathesis: Towards Formal Theorem Proving from Natural Languages. hf-daily. https://huggingface.co/papers/2506.07047 (2025-06-08)
[21] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling. hf-daily. https://huggingface.co/papers/2506.08672 (2025-06-10)
[22] A Survey of Reinforcement Learning for Large Reasoning Models. hf-search. https://huggingface.co/papers/2509.08827 (2025-09-10)
[23] Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries for LLMs. web. https://huggingface.co/blog/async-rl-training-landscape (2026-03-10)
[24] An Empirical Study on Eliciting and Improving R1-like Reasoning Models. hf-search. https://huggingface.co/papers/2503.04548 (2025-03-06)
[25] Open Source RL Libraries for LLMs — Architecture Comparison. web. https://www.anyscale.com/blog/open-source-rl-libraries-for-llms (2025-07-01)
[26] EfficientRollout: System-Aware Self-Speculative Decoding for RL Rollouts. arxiv. https://arxiv.org/abs/2606.18967 (2026-06-17)
[27] Think Twice: Enhancing LLM Reasoning by Scaling Multi-round Test-time Thinking. arxiv. https://arxiv.org/abs/2503.19855 (2025-03-25)
[28] Overclocking LLM Reasoning: Monitoring and Controlling Thinking Path Lengths in LLMs. hf-daily. https://huggingface.co/papers/2506.07240 (2025-06-08)
[29] Walk Before You Run! Concise LLM Reasoning via Reinforcement Learning. arxiv. https://arxiv.org/abs/2505.21178 (2025-05-27)
[30] Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning. arxiv. https://arxiv.org/abs/2410.08146 (2024-10-10)
[31] WMAct: Thinking by Doing — Building Efficient World Model Reasoning in LLMs. web. https://arxiv.org/html/2511.23476v1 (2025-11-01)
[32] Thinking vs. Doing: Agents that Reason by Scaling Test-Time Interaction. hf-daily. https://huggingface.co/papers/2506.07976 (2025-06-09)
[33] Play to Generalize: Learning to Reason Through Game Play. hf-daily. https://huggingface.co/papers/2506.08011 (2025-06-09)
[34] Deliberative Alignment: Reasoning Enables Safer Language Models. arxiv. https://arxiv.org/abs/2412.16339 (2025-01-08)
[35] Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Capable. web. https://arxiv.org/html/2503.00555v1 (2025-03-01)
[36] How RL Reshapes LLM Reasoning: Transferability, Coverage, and Scaling Laws. arxiv. https://arxiv.org/abs/2610.04158 (2026-10-03)
[37] The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity. hf-daily. https://huggingface.co/papers/2506.06941 (2025-06-07)
[38] Merge-of-Thought Distillation (MoT). web. https://arxiv.org/html/2509.08814v1 (2025-09-01)
[39] Through the Valley: Path to Effective Long CoT Training for Small Language Models. hf-daily. https://huggingface.co/papers/2506.07712 (2025-06-09)
[40] TRACER: Turn-level Regret Matching with Inner Reinforcement Credit for Cooperative Multi-LLM Reasoning. arxiv. https://arxiv.org/abs/2605.28699 (2026-05-27)
[41] Reinforcement Pre-Training. hf-daily. https://huggingface.co/papers/2506.08007 (2025-06-09)
[42] Magistral. hf-daily. https://huggingface.co/papers/2506.10910 (2025-06-12)
