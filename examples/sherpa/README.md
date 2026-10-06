# Sherpa

Sherpa trains a teacher LLM with multi-turn reinforcement learning to adapt its teaching
to student archetypes: a frozen Qwen3-1.7B student with one of seven learning
preferences. The teacher is rewarded by how much the student improves on the problem
after the tutoring.

## Train

```bash
python examples/sherpa/scripts/prepare_data.py   # the paper's filtered MATH split
hf download Qwen/Qwen3-8B
hf download Qwen/Qwen3-1.7B
bash examples/sherpa/scripts/run/qwen3.sh        # one node with eight GPUs
```

The [launcher](scripts/run/qwen3.sh) serves the Qwen3-1.7B student on the rollout GPUs
and trains with [`configs/qwen3_8b.yaml`](configs/qwen3_8b.yaml): 4 actor GPUs (FSDP), 4
rollout GPUs (SGLang), rank-16 LoRA, 1500 updates of 16 problems x 8 rollouts. The
guidance gate, the adaptive gate and the answer judge use the teacher's base model with
the LoRA disabled. Arguments after the config are
config overrides, and `DRY_RUN=1` prints the commands without starting them. Relaunching with the same `trial_name` resumes.

```bash
bash examples/sherpa/scripts/run/qwen3.sh examples/sherpa/configs/qwen3_8b.yaml \
  trial_name=my-run cluster.fileroot=/data/runs
```

| Setting                                                                  | Paper               | Meaning                                     |
| ------------------------------------------------------------------------ | ------------------- | ------------------------------------------- |
| `max_turns`                                                              | 10                  | teacher/student rounds                      |
| `gconfig.max_new_tokens` / `gconfig.max_tokens`                          | 1024 / 24576        | teacher tokens per turn / context budget    |
| `guidance_gate`, `adaptive_gate`                                         | true                | true/false turns the corresponding gate on/off |
| `mask_rejected_turns`                                                    | true                | false = credit every turn, episode baseline |
| `retest_replays`                                                         | 8                   | student re-tests after the dialogue         |
| `teacher_pre.enabled`                                                    | true                | the teacher prepares first        |
| `reward.format_error_penalty`, `reward.teacher_exact_repeat_penalty`     | -0.5                | penalty, and the episode ends               |
| `reward.guidance_gate_fail_penalty`, `reward.adaptive_gate_fail_penalty` | 0                   | gate penalties                     |
| `student_models`                                                         | four ID preferences | the students trained against                |

**Another teacher model**: set `actor.path` and check that the model follows the tagged
reply format, `<reasoning>...</reasoning><output>...</output>` or
`<reasoning>...</reasoning><end></end>`.

### Ministral-3-8B

Ministral-3 needs Transformers 5 and SGLang 0.5.10. Create its environment in a separate
checkout, download the pinned checkpoint and train:

```bash
cp envs/ministral/pyproject.toml pyproject.toml && cp envs/ministral/uv.lock uv.lock
uv sync --extra cuda
hf download mistralai/Ministral-3-8B-Instruct-2512-BF16 --revision f6fae9795746f63c9be8344932f01275f3c63734
bash examples/sherpa/scripts/run/ministral3.sh
```

[`configs/ministral3_8b.yaml`](configs/ministral3_8b.yaml) differs from the Qwen config
in the model, the language-only LoRA targets, EOS token 2, a learning rate of 1.5e-5,
the soft overlong penalty and 300 updates.

## Evaluate

One runner evaluates every teacher. A config in [`eval/configs`](eval/configs) names the
teacher, the student and the judge and how each is called:

| Config                                                                                                               | Teacher                                     |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| [`sherpa-qwen3-8b`](eval/configs/sherpa-qwen3-8b.yaml)                                                               | Qwen3-8B with the Sherpa LoRA               |
| [`qwen3-8b`](eval/configs/qwen3-8b.yaml)                                                                             | Qwen3-8B, untrained                         |
| [`pedrl-qwen3-8b`](eval/configs/pedrl-qwen3-8b.yaml)                                                                 | Qwen3-8B with the PedagogicalRL LoRA        |
| [`sherpa-ministral3-8b`](eval/configs/sherpa-ministral3-8b.yaml), [`ministral3-8b`](eval/configs/ministral3-8b.yaml) | Ministral-3-8B with and without Sherpa LoRA |
| [`gemini-3.8-flash`](eval/configs/gemini-3.8-flash.yaml), [`gpt-5.6-luna`](eval/configs/gpt-5.6-luna.yaml)           | commercial APIs                             |

Serve the models with any OpenAI-compatible server, under the names the configs use.
With SGLang, for example:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --served-model-name qwen3-8b \
  --enable-lora --lora-paths /path/to/adapter --max-lora-rank 16 \
  --context-length 40960 --port 30000                                   # teacher
python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B \
  --served-model-name qwen3-1.7b --context-length 40960 --port 30001    # student
python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b-fp8 --port 30002                      # judge
```

Drop the LoRA flags for an untrained teacher. Serve Ministral from
[its environment](#ministral-3-8b) and add `--skip-server-warmup` (SGLang's startup
warmup sends an image, which crashes the server once the LoRA is enabled). Serve the judge, a
Qwen3.5 model, with SGLang 0.5.19 in a separate environment. Put the URLs, the keys and
`TEACHER_ADAPTER` (the adapter path as registered on the teacher server) in `.env` (see
[`.env.example`](../../.env.example)), then:

```bash
bash examples/sherpa/eval/run.sh sherpa-qwen3-8b --preflight   # the three servers answer
bash examples/sherpa/eval/run.sh sherpa-qwen3-8b --limit 1 --output-dir output/eval/smoke
bash examples/sherpa/eval/run.sh sherpa-qwen3-8b               # 528 problems x 7 students
```

Results go to `output/eval/<run_name>/`; repeat a command to resume it. For a commercial
API set its URL and key (for example `GEMINI_BASE_URL`, `GEMINI_API_KEY`). To spread a
run over N processes, run each with `--shard-count N --shard-index I` (I = 0..N-1) and
merge the results with `python -m examples.sherpa.eval.summarize DIR... --output report.json`.

MathTutorBench has its own runner: [`examples/math_tutor_bench`](../math_tutor_bench).

## Data

[`data/math_1.7b_8b/math_filtered.{train,test}.jsonl`](data/math_1.7b_8b) is the paper's
split: 759 training and 528 test MATH problems (Hendrycks et al., 2021; MIT License)
that Qwen3-1.7B fails in two attempts and Qwen3-8B solves.
[`scripts/prepare_data.py`](scripts/prepare_data.py) writes it as the dataset the
configs read; its `filter` mode builds a split for another student/teacher pair.
