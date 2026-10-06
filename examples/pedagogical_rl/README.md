# PedagogicalRL baseline

The PedagogicalRL method on AReaL, trained like Sherpa: Qwen3-8B with rank-16 LoRA, the
Qwen3-1.7B student, the filtered MATH training split, ten teacher turns and 1,500 rollout
batches of 16 problems x 8 trajectories.

From the official method it keeps the native prompts, the GUIDED/ATTEMPTED classroom
assignment, the two whole-dialogue judges (answer leakage and pedagogical values, two
attempts each, used as hard gates), eight final student attempts, one reward per episode
normalized over the problem's eight trajectories, PPO clipping 0.2, `mu=2` and a
`beta=0.001` KL term. The judges run on the rollout model's base with the LoRA disabled,
and the KL reference is the actor with its adapter disabled.

The prompts in [`prompts.py`](prompts.py) are taken from
[eth-lre/PedagogicalRL](https://github.com/eth-lre/PedagogicalRL) (Dinucu-Jianu et al.,
2025), released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Train

Prepare the data (see the [Sherpa README](../sherpa/README.md#train)), then train:

```bash
bash examples/pedagogical_rl/scripts/run/qwen3.sh [CONFIG.yaml] [OVERRIDES...]
```

The config is [`configs/pedrl_qwen3_8b.yaml`](configs/pedrl_qwen3_8b.yaml). The
[launcher](scripts/run/qwen3.sh) serves the student on the rollout GPUs, and `DRY_RUN=1`
prints the commands without starting them.

## Evaluate

Serve Qwen3-8B with the PedagogicalRL adapter as described in the
[Sherpa README](../sherpa/README.md#evaluate), then:

```bash
bash examples/sherpa/eval/run.sh pedrl-qwen3-8b
LORA_PROBE=warn bash examples/math_tutor_bench/run.sh pedrl-qwen3-8b
```
