# Public References Used for the Agent Design

## Fast-WAM

- Paper: https://arxiv.org/abs/2603.16666
- Project page: https://yuantianyuan01.github.io/FastWAM/
- Official repository: https://github.com/yuantianyuan01/FastWAM

The project page reports Fast-WAM on LIBERO at Spatial 98.2, Object 100.0, Goal 97.0, Long 95.2, average 97.6. It also describes Fast-WAM as retaining video/world-model co-training during training while directly generating actions at inference without explicit future-video generation.

The official repository provides the released `libero_uncond_2cam224.pt` checkpoint and FastWAM training/evaluation entry points.

## AutoResearch

- Repository: https://github.com/karpathy/autoresearch
- Program: https://github.com/karpathy/autoresearch/blob/master/program.md

Useful ideas adapted here include baseline-first experimentation, immutable evaluation, Git experiment state, compact result logs, explicit keep/discard/crash recording, simplicity as a tie-breaker, and continuous autonomous iteration.

## RLinf

- RL methods: https://rlinf.readthedocs.io/en/latest/rst_source/examples/methods_index.html
- FastWAM integration PR: https://github.com/RLinf/RLinf/pull/1398

RLinf is referenced as optional embodied training/evaluation infrastructure when relevant to a chosen candidate. PR #1398 tracks FastWAM evaluation and FSDP SFT integration work; its current status should be checked when used.


## Claude Code skills

- https://code.claude.com/docs/en/skills

Project skills live under `.claude/skills/<skill-name>/SKILL.md`.

## Hugging Face Hub

- Upload guide: https://huggingface.co/docs/huggingface_hub/guides/upload
- Environment variables / `HF_TOKEN`: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables

The package uses `HF_TOKEN` for Hub authentication and `cheikh025/ASR` as the durable model-repository target for important checkpoints.
