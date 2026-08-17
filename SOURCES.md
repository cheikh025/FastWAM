# Sources and references

Use current primary sources during setup/research; do not rely on stale memory when repositories or evaluation procedures may have changed.

## FastWAM

- Working fork: https://github.com/cheikh025/FastWAM
- Official upstream: https://github.com/yuantianyuan01/FastWAM
- FastWAM released models/data pointers are documented by the current repository README.
- Parent checkpoint durable repository: https://huggingface.co/cheikh025/ASR

## RoboTwin

- Official RoboTwin 2.0 repository: https://github.com/RoboTwin-Platform/RoboTwin
- Official project site/documentation: https://robotwin-platform.github.io/
- FastWAM's preprocessed RoboTwin dataset pointer is documented by the FastWAM repository; current public dataset: https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam

## LIBERO

- Use the current FastWAM repository's pinned/expected LIBERO setup and evaluator behavior as the project integration source of truth.
- FastWAM preprocessed LIBERO data pointer: https://huggingface.co/datasets/yuanty/LIBERO-fastwam

## Multi-embodiment literature

Read the full paper, not only the abstract:

- Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments — https://arxiv.org/abs/2605.30280
  - Particularly relevant: embodiment-aware conditioning, unified padded action/trajectory tensor interface, per-channel validity masking, per-dataset normalization, heterogeneous data mixing, and progressive training.

Then search/read other directly relevant recent multi-embodiment/generalist robot-policy work as needed. Potential starting points (evaluate relevance rather than mechanically copying them):

- LAP: Language-Action Pre-Training Enables Zero-shot Cross-Embodiment Transfer — https://arxiv.org/abs/2602.10556
- Qwen-RobotManip: Alignment Unlocks Scale for Robotic Manipulation Foundation Models — https://arxiv.org/abs/2606.17846
- other current primary papers/code on cross-embodiment action alignment, heterogeneous action spaces, multi-task co-training, continual learning/retention, and bimanual manipulation.

## Optional RL infrastructure

- RLinf docs: https://rlinf.readthedocs.io/
- FastWAM integration PR (verify current status before use): https://github.com/RLinf/RLinf/pull/1398

## Claude Code / Hugging Face

- Use current official Claude Code documentation for skill/project behavior when needed.
- Use current Hugging Face CLI/API documentation for authenticated checkpoint download/upload. Never write token values into project files.
