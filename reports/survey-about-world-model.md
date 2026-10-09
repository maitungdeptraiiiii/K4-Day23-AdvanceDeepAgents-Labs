# World Models in AI: A Comprehensive Survey of Foundations, Approaches, and Applications

## TL;DR

- World models—internal representations that capture environmental dynamics—trace their origins to Craik's 1943 hypothesis about mental models [1] and were formalized in reinforcement learning through Sutton's Dyna architecture [2]. Ha & Schmidhuber (2018) revived the term for modern neural networks, training agents entirely in self-generated "dreams" [3].
- Three major architectural families have emerged: RSSM-based latent imagination (DreamerV3 and variants), vision-space generative models (Genie, Sora, Cosmos), and JEPA-style decoder-free predictive architectures that operate purely in representation space without pixel reconstruction [4][5][6].
- Vision-based world models have scaled dramatically: Genie 3 generates photorealistic interactive worlds at 20–24 fps from text prompts, while V-JEPA 2 achieves 80% zero-shot pick-and-place success on real robot arms using only 62 hours of action-conditioned video [7][8]. Emergent physical understanding (3D consistency, object permanence) appears at 7–14B parameter scale.
- Language-grounded world models enable natural-language steering of agent behavior, with LLM-based approaches showing significant gains in embodied planning tasks—ReflAct improves ReAct by 27.7%, and language-guided world model agents achieve 3–4× higher rewards than observational baselines [9][10]. However, all tested LLMs still lag human performance on basic world knowledge evaluation [11].
- Major open problems include scalability/training costs ($10K+/month GPU runs), inference speed bottlenecks (V-JEPA 2 takes ~16 seconds per action vs. 100× faster needed), error accumulation over long horizons, underspecified action spaces, and the lack of standardized safety/alignment evaluation [12]. Multiple new benchmarks (WorldModelBench, PhyGround, WBench) are emerging to address evaluation gaps [13][14][15].

## Background

The concept of a "world model" originates from Kenneth Craik's seminal 1943 hypothesis that organisms carry "small-scale models" of external reality within their heads, enabling them to try out alternatives, conclude which is best, and react to future situations before they arise [1]. This idea was later formalized in artificial intelligence through symbolic frame representations (Minsky, 1960s) and, crucially, through Richard Sutton's Dyna architecture (1991), which integrated model-learning, planning on simulated experience, and direct reinforcement learning into a unified loop where an agent learns a forward model of its actions' effects and uses it to generate simulated experience for planning [2].

In modern machine learning, the term was popularized by Ha & Schmidhuber (2018), who trained an agent whose memory component used an RNN to predict compressed latent codes of future observations, and whose controller decided actions based on these representations—all learned and executed inside the agent's own self-generated simulated "dream" [3]. This approach demonstrated that agents could learn entirely from imagined experience rather than real environment interaction.

A key definitional tension persists in the field: whether world models primarily serve *understanding* (compressing sensory data into stable internal representations exposing entities and relations) or *prediction* (generating candidate futures for planning and foresight) [7]. LeCun (2022) proposed a more structured framework—the Joint Embedding Predictive Architecture (JEPA)—which defines a modular system with six components: Configurator, Perception, World Model (predicts future states), Cost function, Actor (optimizes action sequences), and Short-term Memory [4]. LeCun distinguishes Mode-1 (reactive perception→policy→action, no world model) from Mode-2 (deliberative reasoning involving world model rollouts and model-predictive control). The JEPA framework predicts in abstract representation space rather than pixel/token space, arguing that predicting raw pixels is inefficient and unnecessary for intelligent behavior [4].

From a neuroscience perspective, Friston's free energy principle and active inference provide a theoretical grounding where perception, action, and learning all minimize variational free energy through generative models of the world [5][6]. Action fulfills predictions based on inferred states, and belief updating implements Bayesian smoothing with explicit representations of past and counterfactual future states.

## Generative Latent World Models and Dream-Based Planning

The most mature family of world models operates through learned latent dynamics and dream-based planning. At the center is the Recurrent State Space Model (RSSM) architecture, implemented in the Dreamer series. DreamerV3 outperforms specialized RL methods across over 150 diverse tasks with a single fixed configuration, and was the first algorithm to collect diamonds in Minecraft from scratch without human data or curricula [9]. The RSSM encoder maps sensory inputs to stochastic representations, and a sequence model with recurrent state predicts future representations and rewards conditioned on past actions, enabling the actor-critic policy to be trained entirely on imagined latent trajectories rather than real environment interactions.

This paradigm has been extended in multiple directions. DREAMer-VXS applies RSSM-based world modeling to Autonomous Ground Vehicle exploration, using a Convolutional VAE to encode partial LiDAR observations into stochastic latent representations and performing policy improvement in latent-space imagination [11]. ELVIS addresses the compounding model errors that make deep imagination brittle under visual occlusions, replacing standard unimodal MPPI with Gaussian-mixture sampling to handle multi-modal action-value distributions during long-horizon planning [12]. These ensemble-calibrated approaches allow the planner to reason about branching futures where different plausible outcomes exist.

For manipulation tasks, FLIP introduces a flow-centric world model with three modules: a flow generation network (action proposal via CVAE), a flow-conditioned video generation model (dynamics via DiT), and a vision-language representation value module [10]. Trained purely from language-annotated video data, FLIP performs model-based planning by progressively searching successful long-horizon plans on flow and video spaces via tree search, demonstrating zero-shot transfer on LIBERO, FMB, and Bridge-V2 benchmarks.

Navigation World Models (NWM) scales conditional diffusion transformers to 1 billion parameters, trained on diverse egocentric videos from both human and robotic agents [16]. Using an MPC framework with Cross-Entropy Method, NWM optimizes action sequences to reach target goals and can rank trajectories from external policies by measuring LPIPS similarity with goal images. It plans under constraints (e.g., "no left turns") by zeroing out specific actions during optimization.

Tree-Guided Diffusion Planner (TDP) offers a zero-shot test-time planning framework using pretrained diffusion planners via bi-level trajectory-sampling without additional training [12]. It combines diverse parent trajectories via particle guidance for exploration with sub-trajectory refinement through fast conditional denoising guided by task objectives for exploitation.

## Vision-Based and Imagery World Models

Vision-based world models directly predict future images or videos, spanning two major paradigms: pixel-space generative models and feature-space predictive models.

**Pixel-space generative approaches** aim to produce photorealistic future frames. Google DeepMind's Genie (11B parameters) is the first generative interactive environment trained unsupervised from unlabeled Internet videos, comprising a spatiotemporal video tokenizer, autoregressive dynamics model, and latent action model [15]. Genie 2 extends this to 3D environments using autoregressive latent diffusion with classifier-free guidance. Genie 3 achieves real-time operation at 20–24 fps, generating photorealistic worlds from simple text descriptions grounded in Street View data [17]. OpenAI's Sora demonstrates text-conditional diffusion with transformer architecture operating on spacetime patches, generating up to a minute of high-fidelity video with 3D consistency and object permanence [18], though it shows limitations in physics modeling (e.g., inaccurate glass shattering). VideoPoet uses a decoder-only transformer processing multimodal inputs as discrete tokens via MAGVIT V2 tokenizer, demonstrating zero-shot video generation including chaining editing tasks [19]. NVIDIA's Cosmos platform supports Text2World, Image2World, and Video2World modes with auto-regressive inference and ControlNet-style conditioning, achieving up to 30-second video generation [20].

**Feature-space predictive approaches** avoid pixel reconstruction entirely. DINO-WM models visual dynamics on frozen DINOv2 patch embeddings without reconstructing the visual world in pixel space, achieving zero-shot behavioral solutions across maze navigation, push manipulation, and robotic arm control requiring no expert demonstrations, reward modeling, or inverse models [21]. DINO-world trains a predictor on ~60M uncurated web videos in DINOv2's frozen encoder space, vastly more resource-efficient than generative models (<1B vs. 12B parameters for COSMOS) and outperforming previous models on video prediction benchmarks with 6.3% higher mIoU on VSPW segmentation forecasting [22].

Meta's V-JEPA 2 (1.2B parameters) represents the leading decoder-free approach, achieving state-of-the-art visual understanding and zero-shot robot planning [23]. Two-phase training involves actionless pre-training on visual data followed by action-conditioned training on 62 hours of robot data from the DROID dataset. For short-horizon tasks, it uses goal-image specification with model-predictive control; for longer horizon tasks, it specifies series of visual subgoals, achieving 65–80% success rates for pick-and-place of new objects in unseen environments. On real robot arms, V-JEPA 2 achieves 80% zero-shot pick-and-place success [24].

Juno tames predictive latents for vision-language-action (VLA) models by addressing three failures: mismatch with embodiment-specific control, interference with action learning, and teacher miscalibration under distribution shifts [25]. SplitJEPA separates invariant from variant latent factors without reconstruction—a robot pushing a cube should take the same action when camera shifts or lights dim since nothing in the scene has moved [26].

## Language-Grounded and Reasoning World Models

World models are increasingly being applied to language understanding and reasoning, where textual and knowledge-based representations complement sensory ones.

Language-Guided World Models (LWMs) are probabilistic models that simulate environments by reading texts, enabling humans to control agent behaviors via natural verbal communication [27]. Standard Transformer models struggle with compositional generalization in this setting; augmenting with EMMA attention mechanism substantially outperforms baselines. In challenging evaluations, LWM-based agents achieve 3–4× higher average reward than agents using purely observational world models.

LLM-based world models demonstrate decision-making through policy verification and action proposal. Evaluations of GPT-4o across 31 text-based environments show that combining multiple world model functionalities (policy verification, action proposal, policy planning) introduces performance instability that can obscure capability differences between models [28]. GPT-4o significantly outperforms GPT-4o-mini, especially on domain knowledge-intensive tasks.

ReflAct introduces a reasoning backbone that shifts from merely planning next actions to continuously reflecting on the agent's state relative to its goal [29]. Evaluated in ALFWorld, ScienceWorld, and Jericho, ReflAct surpasses ReAct by 27.7% on average, achieving 93.3% success rate in ALFWorld, without relying on additional reflective or memory modules.

Knowledge graph-based approaches like AriGraph construct and update episodic memory graphs integrating semantic and episodic memories during environment exploration [30]. Evaluated in Textworld and NetHack, AriGraph shows significant outperformance over full history, summarization, RAG, Simulacra, and Reflexion baselines. KNOWAGENT enhances LLM planning by incorporating explicit action knowledge to mitigate "planning hallucination"—generating unnecessary or conflicting action sequences—through a knowledgeable self-learning strategy [28].

On evaluation, the Elements of World Knowledge (EWOK) framework tests 20 open-weights LLMs (1.3B–70B parameters) against human performance across 11 world knowledge domains covering social interactions, spatial relations, intuitive physics, and number sense [29]. All tested models perform worse than humans, with performance varying drastically across domains: social interactions highest, physical and spatial relations lowest.

## Applications Across Domains

World models have rapidly expanded from robotics and game playing into healthcare, autonomous driving, finance, and beyond.

**Robotics:** Dreamer 4 learned to collect diamonds in Minecraft with 20,000+ sequential actions from raw pixels using purely offline data with zero environment interaction [24]. Meta's V-JEPA 2 pre-trained on over one million hours of internet video achieves 80% zero-shot pick-and-place on real robot arms [24]. Robotic World Model deploys a dual-autoregressive mechanism with self-supervised training on ANYmal D hardware in zero-shot transfer with minimal sim-to-real loss [31]. The survey of world models for robot learning covers policy learning, planning, and simulation across embodied applications [32].

**Autonomous Driving:** Driving World Models (DWMs) categorize approaches by predicted scene modalities, with surveys reviewing datasets and metrics for autonomous driving world models [33]. WorldLens evaluates driving world models across five aspects: Generation, Reconstruction, Action-Following, Downstream Task, and Human Preference, finding that no existing world model excels universally—those with strong textures often violate physics, while geometry-stable ones lack behavioral fidelity [16]. Behavioral Safety scores remain modest (average '2'~'3' out of '10').

**Healthcare:** Medical World Model (MeWM) visually predicts future disease states based on clinical decisions, improving F1-score in selecting optimal TACE protocol by 13% for interventional physicians [34]. A broader survey identifies four capability levels (L1 temporal prediction through L4 planning/control), with most reviewed systems achieving L1-L2 and fewer implementing L3/L4 [35].

**Finance:** Market-1T provides nearly one trillion one-second quote and trade aggregates across U.S. equities from 2008–2025 for building financial world models analogous to DINO-WM and V-JEPA 2 architectures [36].

**Emergent Physics:** At 7–14B parameters, emergent physical understanding manifests as 3D consistency, object permanence, and realistic physics appearing purely from scale [24]. WorldWeave grows persistent geometric 3D worlds for video generation, decoupling world-state maintenance from visual rendering using continual elevation-map generation [37]. FLEX-WAM supports variable-length contexts and infinite autoregressive generation for long-horizon embodied AI applications [38].

## Trends and Open Problems

**Recent trends (2023–2025):** The field has seen rapid scaling of world model capabilities. Zero-shot manipulation from video pre-training has become feasible—V-JEPA 2 achieves 80% success on real robot arms with only 62 hours of action-conditioned data [24]. Real-time interactive world models like Genie 3 now operate at 20–24 fps generating photorealistic worlds from text [17]. Decoder-free architectures (JEPA variants) have proven competitive with generative approaches while being vastly more compute-efficient [22][26]. Several new evaluation benchmarks have emerged: WorldModelBench evaluates instruction-following and physics-adherence across 7 application domains with 67K human labels [39]; PhyGround focuses specifically on physical reasoning with a 13-law taxonomy [40]; WBench provides multi-turn evaluation across 5 dimensions testing 43 models [41]; WorldExam measures "Inherent Reactivity"—the ability to infer how the world should react and generate plausible consequences not explicitly described [42]; and WorldScore decomposes world generation into controllability, quality, and dynamics assessment [43].

**Open problems:** Scalability remains a critical bottleneck. Training costs run $10K+/month for GPU runs, and serving costs for models like Genie 3 reach $100/hr [24]. Inference speed is inadequate—V-JEPA 2 takes ~16 seconds per action versus the 100× faster rate needed for real-time deployment [24]. Error accumulation over long horizons continues to plague latent-space planning approaches, addressed partially by ensemble calibration (ELVIS) but not fully solved [12]. Tactile sensing gaps persist in embodied applications, limiting world models to vision-dominated representations. Underspecified action spaces and weak interventional validation represent fundamental methodological challenges [35]. The lack of standardized safety and alignment evaluation for world models deployed in real-world systems remains largely unaddressed [35]. Finally, the tension between generative pixel-space models and predictive feature-space models continues unresolved—while JEPA approaches are more efficient, autoregressive observation generation may still be necessary for certain reasoning tasks [10].

## References
[1] Hypothesis on the Nature of Thought (Craik, 1943). web. https://markhuckvale.com/research/hp/Craik_NOE_Chapter_5.pdf (1943-01-01)
[2] Dyna, an Integrated Architecture for Learning, Planning, and Reacting (Sutton, 1991). web. https://dl.acm.org/doi/10.1145/122344.122377 (1991-07-01)
[3] World Models (Ha & Schmidhuber, 2018). web. https://arxiv.org/abs/1803.10122 (2018-03-27)
[4] A Path Towards Autonomous Machine Intelligence (LeCun, 2022). web. https://openreview.net/pdf?id=BZ5a1r-kVsf (2022-06-27)
[5] World Model Learning and Inference (Friston et al., 2021). web. https://www.sciencedirect.com/science/article/pii/S0893608021003610 (2021-12-01)
[6] Active Inference: A Process Theory (Friston et al., 2016). web. https://activeinference.github.io/papers/process_theory.pdf (2016-11-01)
[7] Understanding World or Predicting Future? A Comprehensive Survey of World Models (Ding et al., 2025). web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[8] Understanding World or Predicting Future? A Comprehensive Survey of World Models (Ding et al., 2024). hf-search. https://huggingface.co/papers/2411.14499 (2024-11-21)
[9] Mastering Diverse Domains through World Models (DreamerV3). web. https://arxiv.org/abs/2301.04104 (n.d.)
[10] FLIP: Flow-Centric Generative Planning as General-Purpose Manipulation World Model. web. https://arxiv.org/abs/2412.08261 (2025-02-16)
[11] DREAMer-VXS: Latent World Model for Sample-Efficient AGV Exploration. web. https://arxiv.org/abs/2512.00005 (2025-10-06)
[12] ELVIS: Ensemble-Calibrated Latent Imagination for Long-Horizon Visual MPC. web. https://arxiv.org/abs/2605.04709 (2026-05-06)
[13] LeWAM: JEPA World Action Model with Diffusion-Steering-Based MPC. arxiv. https://arxiv.org/abs/2610.12407 (2026-10-08)
[14] A Generalist Agent (Gato). web. https://arxiv.org/abs/2205.06175 (2022-05-12)
[15] Genie: Generative Interactive Environments (Google DeepMind). web. https://proceedings.mlr.press/v235/bruce24a.html (2024-07-08)
[16] WorldLens: Full-Spectrum Evaluations of Driving World Models in Real World. web. https://worldbench.github.io/assets_common/papers/worldlens.pdf (n.d.)
[17] Genie 3 - Real-time Interactive World Model (Google DeepMind). web. https://deepmind.google/research/publications/60474/ (2025-01-01)
[18] Video generation models as world simulators — Sora (OpenAI). web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[19] VideoPoet: A Large Language Model for Zero-Shot Video Generation (Google). web. https://arxiv.org/abs/2312.14125 (2023-12-19)
[20] Cosmos World Foundation Model Platform (NVIDIA). web. https://huggingface.co/docs/diffusers/main/en/api/pipelines/cosmos (2025-01-01)
[21] DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning. web. https://arxiv.org/abs/2411.04983 (2024-11-04)
[22] DINO-world: Back to the Features — DINO as Foundation for Video World Models. web. https://arxiv.org/html/2507.19468 (2025-07-25)
[23] V-JEPA 2: Video Joint-Embedding Predictive Architecture 2 (Meta/Fair). web. https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/ (2025-06-11)
[24] Can world models unlock general purpose robotics?. web. https://www.bvp.com/atlas/can-world-models-unlock-general-purpose-robotics (2026-03-10)
[25] Juno: Taming Predictive Latents for Vision-Language-Action Models. arxiv. https://arxiv.org/abs/2610.09940 (2026-10-07)
[26] SplitJEPA: Learning Invariant and Variant Latent Worlds without Reconstruction. arxiv. https://arxiv.org/abs/2610.12349 (2026-10-08)
[27] Language-Guided World Models: A Model-Based Approach to AI Control. web. https://arxiv.org/abs/2402.01695 (2024-09-04)
[28] KNOWAGENT: Knowledge-Augmented Planning for LLM-Based Agents. web. https://aclanthology.org/2025.findings-naacl.205.pdf (2025)
[29] Elements of World Knowledge (EWOK): A Cognition-Inspired Framework for Evaluating Basic World Knowledge in Language Models. web. https://aclanthology.org/2025.tacl-1.57.pdf (2025)
[30] AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents. web. https://www.ijcai.org/proceedings/2025/0002.pdf (2025)
[31] Robotic World Model: A Neural Network Simulator for Robust Policy Optimization. web. https://arxiv.org/abs/2501.10100v1 (2025-01-17)
[32] World Model for Robot Learning: A Comprehensive Survey. hf-search. https://huggingface.co/papers/2605.00080 (2026-04-30)
[33] The Role of World Models in Shaping Autonomous Driving: A Comprehensive Survey. hf-search. https://huggingface.co/papers/2502.10498 (2025-02-14)
[34] Medical World Model (MeWM). web. https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_Medical_World_Model_ICCV_2025_paper.pdf (2025)
[35] Beyond Generative AI: World Models for Clinical Prediction, Counterfactuals, and Planning. web. https://arxiv.org/html/2511.16333v1 (2025-11-20)
[36] Towards Financial World Modeling. web. https://arxiv.org/html/2610.09048 (2026-10-06)
[37] WorldWeave: Growing Persistent Geometric Worlds for Video Generation. arxiv. https://arxiv.org/abs/2609.34221 (2026-09-28)
[38] FLEX-WAM: Flexible Block-Causal World-Action Models for Long-Horizon Imagination and Planning. arxiv. https://arxiv.org/abs/2610.05483 (2026-10-04)
[39] WorldModelBench: Judging Video Generation Models As World Models. hf-search. https://huggingface.co/papers/2502.20694 (2025-02-28)
[40] PhyGround: Benchmarking Physical Reasoning in Generative World Models. hf-search. https://huggingface.co/papers/2605.10806 (2026-05-11)
[41] WBench - Interactive World Model Benchmark. web. https://meituan-longcat.github.io/WBench/ (n.d.)
[42] WorldExam: Benchmarking World Models from Apparent Appearance to Inherent Reactivity. web. https://worldexam.github.io/ (n.d.)
[43] WorldScore: A Unified Evaluation Benchmark for World Generation. hf-search. https://huggingface.co/papers/2504.00983 (2025-04-01)
