# Zhengzhi Liu

**Embodied AI & Robot Learning** — vision-language-action policies, world models, and onboard deployment.

Senior ML engineer with 6+ years building the full perception → world model → action stack for autonomous
machines (Lucid Motors, SafeAI, Audi/VW), now focused on humanoid robotics. Ph.D. in 3D imaging, postdoc at Stanford.
I work across the whole path: multimodal 3D perception, latent world models, and getting models to run in
real time on embedded NVIDIA hardware. Current focus is VLA policy design — OpenVLA / OpenVLA-OFT, π0, and
Isaac GR00T — specifically how action representation and decoding scheme set the control frequency you can
actually hit onboard.

📍 San Jose, CA · [LinkedIn](https://www.linkedin.com/in/zhengzhiliu/) · [Google Scholar](https://scholar.google.com/citations?user=JZgUXF8AAAAJ)

---

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
| [PETR](https://github.com/albertnew2012/PETR) | Camera-only 3D detection via 3D position-aware embeddings — the camera branch. |
| [DSVT](https://github.com/albertnew2012/DSVT) | Dynamic sparse voxel transformer — the LiDAR branch. |
| [DSVT-AI-TRT](https://github.com/albertnew2012/DSVT-AI-TRT) | DSVT through CUDA/TensorRT/C++ — the path to real-time onboard inference. |
| [Deformable-DETR](https://github.com/albertnew2012/Deformable-DETR) | Deformable attention, the block under most BEV detection heads. |

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

## Robot Foundation Models

Where my current work sits: VLA policy design, and specifically how action representation and decoding
scheme determine the control frequency you can hit on embedded compute.

| Repo | Role |
|---|---|
| [Isaac-GR00T](https://github.com/albertnew2012/Isaac-GR00T) | NVIDIA's open humanoid VLA — VLM backbone with a diffusion transformer action head, cross-embodiment. |
| [openvla](https://github.com/albertnew2012/openvla) | Autoregressive VLA baseline — discrete action tokens over a VLM backbone. |
| [openpi](https://github.com/albertnew2012/openpi) | π0's flow-matching action expert, the contrast case to token decoding. |
| [nanoVLM](https://github.com/albertnew2012/nanoVLM) | Minimal VLM training loop — the backbone half of a VLA, stripped down. |
