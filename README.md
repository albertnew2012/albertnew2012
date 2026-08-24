# Zhengzhi Liu

**Embodied AI & Robot Learning** — world models, vision-language-action policies, onboard deployment.

Machine learning engineer with 6+ years building the full **perception → world model → action** stack for
autonomous machines, now focused on humanoid robotics and embodied AI. Deep hands-on experience with
multimodal transformers that fuse camera, LiDAR, and language into a shared latent space, and with learned
world models that forecast future states for closed-loop control. Working knowledge of modern robot
foundation models — OpenVLA / OpenVLA-OFT, π0 (openpi), and NVIDIA Isaac GR00T — including action chunking,
parallel decoding, and imitation-learning fine-tuning. Proven record taking research models to production:
custom CUDA kernels, TensorRT/ONNX deployment, and real-time inference on the NVIDIA Xavier/Orin/Thor
compute humanoid platforms run on.

📍 San Jose, CA · [albertnew2012@gmail.com](mailto:albertnew2012@gmail.com) · [LinkedIn](https://www.linkedin.com/in/zhengzhiliu/) · [Google Scholar](https://scholar.google.com/citations?user=JZgUXF8AAAAJ)

---

## Experience

| Role | Company | |
|---|---|---|
| Senior Machine Learning Engineer | **Lucid Motors** — Newark, CA | 12/2024 – Present |
| Senior Machine Learning Engineer | **SafeAI** (autonomous heavy equipment) — San Jose, CA | 03/2024 – 10/2024 |
| Senior ADAS System Development Engineer | **Audi of America** (Volkswagen Group) — San Jose, CA | 05/2022 – 01/2024 |
| Research Scientist | **RefleXion Medical** — Hayward, CA | 01/2020 – 05/2022 |

At Lucid: a multimodal fusion backbone unifying LiDAR (DSVT), camera (PETR), and language embeddings in a
shared BEV latent space; an end-to-end world model coupling 3D perception with latent scene prediction; an
LLM-agent-orchestrated auto-labeling engine for large multimodal training sets; and ownership of the
research-to-production path through ONNX/TensorRT and custom CUDA plugins onto Orin/Thor embedded compute.

## Education

| | | |
|---|---|---|
| **Postdoctoral Research Fellow**, Deep Learning for Imaging & Perception | Stanford University | 2018 – 2020 |
| **Ph.D., 3D Imaging Theory and Applications** | University of Tennessee, Knoxville | 2014 – 2018 |

## Technical Skills

**Vision-Language-Action (VLA) Policies** — OpenVLA / OpenVLA-OFT, π0 (openpi), NVIDIA Isaac GR00T;
diffusion & flow-matching action experts; action chunking, parallel decoding, continuous action regression;
discrete action tokenization vs. continuous regression; imitation-learning fine-tuning

**Robot Learning & World Models** — World-model–action (WMA) architectures and latent world models;
closed-loop future-state prediction; sim-to-real transfer; teleoperation & demonstration data pipelines

**Generative & Foundation Models** — LLMs, VLMs, multimodal fusion architectures, LoRA / PEFT fine-tuning,
diffusion models, VAE/GAN, agentic multi-model pipelines

**3D Perception & Spatial Reasoning** — Camera & LiDAR 3D detection and tracking, BEV representation learning,
occupancy prediction, LiDAR–camera fusion, sensor calibration, auto-labeling, end-to-end driving models

**Simulation & Robotics Systems** — ROS / ROS2, Isaac Sim / Isaac Lab, MuJoCo, LeRobot, synthetic data
generation, closed-loop evaluation, Docker, Linux, Git

**Deployment & Optimization** — TensorRT / ONNX Runtime, custom CUDA plugins, quantization, real-time
inference on NVIDIA Xavier / Orin / Thor, safety-OS deployment

**Programming & Frameworks** — Python, C++, CUDA, PyTorch, TensorFlow, OpenCV, Open3D, PCL

---

# Selected Repositories

## Architecture Deep Dives

Original study work — inference ports, ablations, visualizations, and written analysis I produced
working through each architecture. Each lives on the `dev_albertl` branch of its repo.

| Study | What I built |
|---|---|
| [PETR / PETRv2](https://github.com/albertnew2012/PETR/tree/dev_albertl) | ~6,600 lines in `petr_port/` — standalone inference port for PETR and PETRv2, a compatibility layer, and written deep dives on 3D positional encoding, query/reference-point construction, temporal modeling, and cls/reg confidence — plus frustum and BEV visualizations. |
| [OpenVLA](https://github.com/albertnew2012/openvla/tree/dev_albertl) | ~3,600 lines — LIBERO evaluation harness, QLoRA fine-tuning on a single RTX 3090, SimplerEnv zero-shot WidowX eval, an action-tokenization explainer, and a reproducible Docker setup. |
| [OpenVLA-OFT](https://github.com/albertnew2012/openvla-oft/tree/dev_albertl) | ~2,800 lines — latency benchmark measuring effective control frequency, open-loop horizon ablation on the chunking speed/robustness tradeoff, parallel-decode attention-mask visualization, and a stage 0–3 training recipe. |
| [nanochat](https://github.com/albertnew2012/nanochat/tree/dev_albertl) | ~5,500 lines — a 16-part reading curriculum plus 14 runnable probes dissecting Karpathy's released d32 / base-d20 checkpoints on a single RTX 3090: KV-cache engine, sliding-window attention, RoPE, RMSNorm, minimal SFT, and tool use. Includes a backward-compatibility fix to `checkpoint_manager` so pre-`autoresearch` checkpoints load by zero-filling only the params that are exact no-ops in `forward()`. |

## Robot Foundation Models

Where my current work sits: VLA policy design, and specifically how action representation and decoding
scheme determine the control frequency you can hit on embedded compute.

| Repo | Role |
|---|---|
| [openpi-pytorch](https://github.com/albertnew2012/openpi-pytorch) | π0 reimplemented in PyTorch — flow-matching action expert, worked through end to end. |
| [Isaac-GR00T](https://github.com/albertnew2012/Isaac-GR00T) | NVIDIA's open humanoid VLA — VLM backbone with a diffusion transformer action head, cross-embodiment. |
| [openvla](https://github.com/albertnew2012/openvla) | Autoregressive VLA baseline — discrete action tokens over a VLM backbone. |
| [openpi](https://github.com/albertnew2012/openpi) | π0's flow-matching action expert, the contrast case to token decoding. |
| [nanoVLM](https://github.com/albertnew2012/nanoVLM) | Minimal VLM training loop — the backbone half of a VLA, stripped down. |

## Foundation Models — Implemented From Scratch

Building these end to end is how I work through an architecture rather than just reading the paper.

| Repo | What it is |
|---|---|
| [VLM](https://github.com/albertnew2012/VLM) | Vision-language model: image encoder, projection into the LM embedding space, multimodal training loop. |
| [LLM](https://github.com/albertnew2012/LLM) | Transformer language model built from the ground up — attention, tokenization, training. |
| [transformer](https://github.com/albertnew2012/transformer) / [ViT-pytorch](https://github.com/albertnew2012/ViT-pytorch) | Core transformer and Vision Transformer implementations. |
| [stable-diffusion](https://github.com/albertnew2012/stable-diffusion) | Latent diffusion — UNet denoiser, scheduler, conditioning. |
| [DynamiCrafter](https://github.com/albertnew2012/DynamiCrafter) | Video diffusion — the generative side of synthetic scene and rare-event data. |

## 3D Perception & Sensor Fusion

| Repo | What it is |
|---|---|
| [camera-lidar-sensor-fusion](https://github.com/albertnew2012/camera-lidar-sensor-fusion) | Camera-to-LiDAR fusion — projection, calibration, and fused 3D output. |
| [detr](https://github.com/albertnew2012/detr) | DETR-style end-to-end detection with transformer decoders. |
| [Driver-Attention-Monitor-System](https://github.com/albertnew2012/Driver-Attention-Monitor-System) | Face-orientation and drowsiness monitoring for driver state estimation. |
| [UNET-Semantic-Segementation](https://github.com/albertnew2012/UNET-Semantic-Segementation) | U-Net semantic segmentation on Cityscapes. |

**Stacks I work in.** Upstream implementations of the architectures behind the multimodal BEV fusion stack
I build on professionally, mirrored here for experimentation.

| Repo | Role in the stack |
|---|---|
| [PETR](https://github.com/albertnew2012/PETR) | Camera-only 3D detection via 3D position-aware embeddings — the camera branch. See the [deep dive](https://github.com/albertnew2012/PETR/tree/dev_albertl) above. |
| [DSVT](https://github.com/albertnew2012/DSVT) | Dynamic sparse voxel transformer — the LiDAR branch. |
| [DSVT-AI-TRT](https://github.com/albertnew2012/DSVT-AI-TRT) | DSVT through CUDA/TensorRT/C++ — the path to real-time onboard inference. |
| [Deformable-DETR](https://github.com/albertnew2012/Deformable-DETR) | Deformable attention, the block under most BEV detection heads. |

## Deployment & Real-Time Systems

The part that decides whether a policy actually runs on a robot.

| Repo | What it is |
|---|---|
| [pytorch-onnx-tensorRT](https://github.com/albertnew2012/pytorch-onnx-tensorRT) | PyTorch → ONNX → TensorRT engine, with C++ inference. |
| [CUDA_kernel_debug](https://github.com/albertnew2012/CUDA_kernel_debug) | Custom CUDA kernel development and debugging setup. |
| [point_cloud_saver](https://github.com/albertnew2012/point_cloud_saver) | ROS2 node subscribing to point-cloud messages and writing PCD files. |
| [ros_debug](https://github.com/albertnew2012/ros_debug) | ROS2 debugging workflow in VSCode. |
