# Reproducing and extending PAVE

This page describes the public code's data interfaces and external
requirements. Start with the [CPU example](../examples/controller_demo.py)
for a small demonstration of the controller API without pretrained weights.

## Cached-feature data contract

The training script operates on frozen features saved with `torch.save`.
Each feature file contains:

| Field | Shape / type | Meaning |
| --- | --- | --- |
| `current_tokens` | `[N, D_visual]` | Current post-encoder visual tokens |
| `xy` | `[N, 2]` | Normalized token centers in the original current-screen lattice |
| `task_embedding` | `[D_task]` | Frozen task representation |
| `target_mask` | `[N]` | Offline sufficient-mask supervision |
| `experience_id` | String | State identifier for provenance |

The matching prior file contains `spatial_prior` with shape `[N, 3]`.
Each index entry identifies the feature and prior files:

```json
[
  {
    "experience_id": "episode_001:3",
    "feature_path": "outputs/features/episode_001_3.pt",
    "spatial_prior_path": "outputs/priors/episode_001_3.pt"
  }
]
```

The feature tensor shapes and prior rows must refer to the same ordered
current-token lattice. Both files are needed by the dataset loader, including
for `--current-only`; that flag zeros the loaded priors before model execution.
Paths are resolved from the current working directory, not from the index
file's directory. The example paths describe files you prepare; no feature
cache is shipped with the repository.

See the [example index](../examples/feature_index.example.json) and
[feature-record schema](../examples/feature_record.schema.json).

## Offline sufficient-mask supervision

The [supervision generator](../scripts/generate_guiodyssey_supervision.py)
expects a history bank, aligned embedding arrays, an existing sparse-label
bank, screenshots, a system-prompt file, and access to the frozen GUI-Owl
backbone. Inspect its arguments with:

```bash
python scripts/generate_guiodyssey_supervision.py --help
```

The procedure is:

1. Select query states and retrieve training-bank support while excluding the
   query's own episode.
2. Construct candidate masks from target neighborhoods, task-matched OCR,
   layout context, and retrieved sparse support.
3. Evaluate those candidates with the frozen action backbone against the
   externally checkable target outcome.
4. Retain the smallest successful candidate. When none succeeds, mark the
   state as unresolved by this candidate search.

Ground-truth boxes are used for offline label construction and evaluation,
not as runtime controller inputs. Full-only labels do not become all-one
historical spatial priors. The generated supervision JSON is an intermediate
artifact; training additionally requires the corresponding cached feature
and prior files described above.

## Training and evaluation

The [training commands](../README.md#training) run either the history-aware
controller or a matched current-only ablation. Controller precision, recall,
and F1 describe mask prediction. Evaluating GUI task success additionally
requires the frozen action model and an external action/task evaluator.

For comparisons, hold the backbone, controller architecture, data split,
visual-token budget, task instances, and evaluator fixed. Record both the
number of retained post-encoder visual tokens and the task outcome.

## Online experience bank

`memento.online.EpisodeAtomicBank` stages records during a rollout and commits
them only after `finish_episode(success)` receives an external success signal.
Failed or interrupted episodes do not become retrievable history.

- Full-route successes can add templates without sparse masks.
- Compact-route successes add sparse evidence only when the realized retained
  ratio satisfies the configured admission rule.
- Read-only mode supports frozen-bank revisit experiments.

To audit a saved bank without loading an action backbone:

```bash
python scripts/audit_online_bank.py /path/to/bank \
  --out outputs/bank_audit.json
```

The audit checks record/label/embedding alignment and reports record kinds,
per-package coverage, embedding shapes, and SHA-256 file hashes. It expects
`bank.json`, `labels.json`, and `embeddings.npz` in the bank directory.

## Backbone and benchmark integration

- [GUI-Owl adapter](../memento/guiowl.py): post-encoder token intervention for
  the GUI-Owl interface used by the supplied supervision script.
- [Qwen2.5-VL adapter](../memento/qwen.py): position-aligned selection from
  the merged visual-feature lattice.
- [AndroidWorld integration contract](../integrations/androidworld/README.md):
  connects episode reset, decision staging, evaluator results, and bank writes.

The public repository provides integration utilities and contracts rather
than a complete benchmark runner. Dataset preparation, model weights,
trained controller checkpoints, emulator configuration, and benchmark task
initialization must be supplied separately.
