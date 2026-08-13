## Hi there 👋

<!--
**albertnew2012/albertnew2012** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
# Zhengzhi Liu

**Embodied AI & Robot Learning** — vision-language-action policies, world models, and onboard deployment.

Senior ML engineer with 6+ years building the full perception → world model → action stack for autonomous
machines (Lucid Motors, SafeAI, Audi/VW), now focused on humanoid robotics. Ph.D. in 3D imaging, postdoc at Stanford.
I work across the whole path: multimodal 3D perception, latent world models, and getting models to run in
real time on embedded NVIDIA hardware.

📍 San Jose, CA · [LinkedIn](https://www.linkedin.com/in/zhengzhiliu/) · [Google Scholar](https://scholar.google.com/citations?user=JZgUXF8AAAAJ) · albertnew2012@gmail.com

---

## 3D Perception & Sensor Fusion

| Repo | What it is |
|---|---|
| [camera-lidar-sensor-fusion](https://github.com/albertnew2012/camera-lidar-sensor-fusion) | Camera-to-LiDAR fusion — projection, calibration, and fused 3D output. |
| [detr](https://github.com/albertnew2012/detr) | DETR-style end-to-end detection with transformer decoders. |
| [Driver-Attention-Monitor-System](https://github.com/albertnew2012/Driver-Attention-Monitor-System) | Face-orientation and drowsiness monitoring for driver state estimation. |
| [UNET-Semantic-Segementation](https://github.com/albertnew2012/UNET-Semantic-Segementation) | U-Net semantic segmentation on Cityscapes. |

## Foundation Models — Implemented From Scratch

Building these end to end is how I work through an architecture rather than just reading the paper.

| Repo | What it is |
|---|---|
| [VLM](https://github.com/albertnew2012/VLM) | Vision-language model: image encoder, projection into the LM embedding space, multimodal training loop. |
| [LLM](https://github.com/albertnew2012/LLM) | Transformer language model built from the ground up — attention, tokenization, training. |
| [transformer](https://github.com/albertnew2012/transformer) / [ViT-pytorch](https://github.com/albertnew2012/ViT-pytorch) | Core transformer and Vision Transformer implementations. |
| [stable-diffusion](https://github.com/albertnew2012/stable-diffusion) | Latent diffusion — UNet denoiser, scheduler, conditioning. |

## Deployment & Real-Time Systems

The part that decides whether a policy actually runs on a robot.

| Repo | What it is |
|---|---|
| [pytorch-onnx-tensorRT](https://github.com/albertnew2012/pytorch-onnx-tensorRT) | PyTorch → ONNX → TensorRT engine, with C++ inference. |
| [CUDA_kernel_debug](https://github.com/albertnew2012/CUDA_kernel_debug) | Custom CUDA kernel development and debugging setup. |
| [point_cloud_saver](https://github.com/albertnew2012/point_cloud_saver) | ROS2 node subscribing to point-cloud messages and writing PCD files. |
| [ros_debug](https://github.com/albertnew2012/ros_debug) | ROS2 debugging workflow in VSCode. |

---

## Currently

Working through modern robot foundation models — OpenVLA / OpenVLA-OFT, π0 (openpi), and NVIDIA Isaac GR00T —
with a focus on action representation (discrete tokenization vs. continuous regression), action chunking and
parallel decoding, and what those choices cost at inference time on embedded compute.

> Repos on this profile forked from upstream projects are reading material for that study, not my own work.
> Everything listed above is mine.
