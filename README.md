# Sherpa: Teaching LLMs to Teach Adaptively

> <sub>*Named after the Sherpa people, renowned Himalayan mountaineering guides,
> our framework aims to enable language models to help people reach new heights
> through learning.*</sub>

<p align="center">
  <a href="https://arxiv.org/abs/2610.08778"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=flat-square&logo=arxiv&logoColor=white&labelColor=24292F" alt="Paper"></a>
  <a href="https://huggingface.co/collections/SALT-NLP/sherpa"><img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=flat-square&logo=huggingface&logoColor=FFD21E&labelColor=24292F" alt="Models"></a>
</p>

<p align="center">
  <img src="assets/pipeline.png" alt="Overview of the Sherpa framework" width="100%">
</p>

## Abstract

Large language models (LLMs) have become increasingly capable problem solvers, but being
able to solve a problem is not the same as being able to teach it. Existing approaches to
training LLMs as teachers rely on demonstrations, preference data, or predefined
pedagogical criteria that specify what good teaching looks like. However, these signals
are often not grounded in individual student learning outcomes, where effective teaching
strategies can vary substantially across learners. To address this, we introduce Sherpa,
a multi-turn reinforcement learning framework that instantiates multiple student
archetypes with LLMs conditioned on distinct learning preferences and trains a teacher
model to adapt its instruction by directly maximizing their learning outcomes. Teacher
LLMs trained with Sherpa improve instructed students' performance across all archetypes
by an average of 20.5 percentage points. Under MathTutorBench's evaluation, Sherpa raises
the overall pedagogy score from 52.5% to 79.2%, indicating better teaching responses. Our
human studies show that the trained teacher is preferred over the base model in 79.6% of
pairwise comparisons. Together, Sherpa trains LLM teachers to adapt to diverse simulated
students and become better aligned with human teachers, paving the road towards AI tutors
teaching real students.

*This repository contains the training and evaluation code for the
[paper](assets/paper.pdf), built on [AReaL](https://github.com/inclusionAI/AReaL)
v1.0.3.*

## Contents

```text
Sherpa/
├── examples/
│   ├── sherpa/              # Sherpa: training, configs, evaluation
│   ├── pedagogical_rl/      # the PedagogicalRL baseline on the same framework
│   ├── math_tutor_bench/    # MathTutorBench runner (local checkpoints and API models)
│   ├── common/              # shared model-calling and parsing utilities
│   └── areal_examples/      # AReaL's own examples
├── envs/ministral/          # the separate Ministral-3 environment
├── areal/                   # AReaL v1.0.3 with our changes
└── assets/                  # the paper and its overview figure
```

## Installation

Linux with NVIDIA GPUs. The rollout GPUs hold
both the teacher's inference engine and the student server. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra cuda
```

This is AReaL's default environment (Transformers 4.57, SGLang 0.5.9); see AReaL's
[installation guide](https://github.com/inclusionAI/AReaL/blob/v1.0.3/docs/en/tutorial/installation.md)
for details. When training Ministral, you can create its environment in a separate checkout from [`envs/ministral/`](envs/ministral) (see
[Ministral-3-8B](examples/sherpa/README.md#ministral-3-8b)).

Endpoints and keys should be in `.env`; see [`.env.example`](.env.example).

## Quick start

```bash
# 1. Data: the paper's filtered MATH split (759 train / 528 test MATH problems).
python examples/sherpa/scripts/prepare_data.py

# 2. Models (loaded from the local Hugging Face cache during training).
hf download Qwen/Qwen3-8B
hf download Qwen/Qwen3-1.7B

# 3. Train Sherpa on eight GPUs: Qwen3-8B teacher, local Qwen3-1.7B student.
bash examples/sherpa/scripts/run/qwen3.sh
```

Details and other teacher models:
[`examples/sherpa/README.md`](examples/sherpa/README.md). The PedagogicalRL baseline:
[`examples/pedagogical_rl/README.md`](examples/pedagogical_rl/README.md).

## Evaluation

- **Student-archetype benchmark**:
  [`examples/sherpa/eval`](examples/sherpa/README.md#evaluate), one runner with a config
  per teacher (Sherpa, Qwen3-8B, PedagogicalRL, Ministral, Gemini, GPT).
- **MathTutorBench**:
  [`examples/math_tutor_bench`](examples/math_tutor_bench/README.md).

## Change model/method

- **Another teacher model**: point `actor.path` of a Sherpa config at it (see the
  [Sherpa README](examples/sherpa/README.md#train)), including models that need their own environment. The
  [Ministral recipe](examples/sherpa/README.md#ministral-3-8b) is a worked example.
- **Another student model**: `student_models` in the
  [Sherpa config](examples/sherpa/configs/qwen3_8b.yaml) lists the students, each served through an OpenAI-compatible endpoint.
- **Another training method**: write an AReaL rollout workflow and a config
  (refer to [`examples/pedagogical_rl`](examples/pedagogical_rl)), and evaluate its LoRA
  checkpoints with the same runner.

## License

[Apache-2.0](LICENSE), as AReaL. The filtered MATH split contains problems from the
[MATH dataset](https://github.com/hendrycks/math) (Hendrycks et al., 2021), released
under the MIT License.

## Citation

```bibtex
@misc{xu2026sherpa,
  title         = {Sherpa: Teaching LLMs to Teach Adaptively},
  author        = {Weixian Xu and Yanzhe Zhang and Zora Zhiruo Wang and Changyu Chen and Diyi Yang},
  year          = {2026},
  eprint        = {2610.08778},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.08778}
}
```
