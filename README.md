# CTPA-SF landslide detection

This is the current low-rank SF implementation and experiment snapshot. **The baseline is YOLO11n.** CTPA is an intermediate architecture and the uncompressed continuation control. Current SF is `sf_lr_aggressive`. The previous SharedFormer manuscript snapshot is superseded and retained for provenance in Git history and the archive branch.

## Current model and results

SF factorizes CTPA's joint attention input (128→48→384), both FFN maps (128→32→256 and 256→32→128), and fusion (256→80→256), followed by recovery training. Attention retains four 32-channel heads and an unchanged 128→128 output projection. The same LayerNorm is reused at two sites. This uses established truncated-SVD approximation; general novelty of SVD is not claimed.

Unfused parameters: YOLO11n 2,590,035; CTPA 2,493,267; SF 2,403,155. SF reduces parameters by 7.22% versus YOLO11n and 3.61% versus CTPA. Bijie test mAP50–95: YOLO11n 51.05±2.34%; CTPA continuation control 56.34±4.55%; SF 56.97±4.55% (three-seed mean±sample SD).

**Training budget limitation:** YOLO11n uses 200 source epochs. The CTPA control receives an additional 200-epoch continuation stage; SF uses the 200-epoch CTPA teacher and 200-epoch recovery. Initial checkpoints remain eligible in validation selection. The 5.92-point SF-minus-YOLO11n mean is not an equal-total-budget architectural effect. Against the matched CTPA continuation control, the mean paired difference is +0.64 points. Current SF latency has not been measured.

## Data and frozen splits

Download [Bijie](https://gpcv.whu.edu.cn/data/Bijie_pages.html) and [CAS](https://doi.org/10.5281/zenodo.10294997) from their original providers. Data licensing terms remain applicable; no imagery or masks are redistributed. CAS documentation includes CC BY-NC 4.0 and sensor-specific imagery terms; public download is not unrestricted reuse permission.

Bijie uses 770 positive originals, excluding the 2,003 negative images: 537 train, 158 validation, 75 test. Filename place-code prefixes define groups; df/zdf are conservatively coassigned. One train-only horizontal flip gives 1,074 training images. Validation/test are unchanged. Filename grouping is not independently georeferenced site separation. `splits/bijie_manifest.json` fixes every assignment and transform.

CAS uses relative-pixel registration within the largest retained component, then spatial bands with excluded buffers. All five strata are retained: Jiuzhai 0.5 m, Moxi 0.2 m, Moxi 1 m, Mengdong 0.5 m and Longxi satellite 0.5 m. Moxi resolutions are one region, and Moxi 1 m has only 17 retained test images. These relative-coordinate splits do not establish absolute geographic independence. Each 8-connected foreground component of at least one pixel becomes an enclosing axis-aligned box.

## Environment and reproduction

Use Python 3.10, PyTorch 2.0.1+cu118 and torchvision 0.15.2+cu118, then install `requirements.txt`. The archived `ultralytics/` runtime reports 8.3.185; AGPL licensing and upstream attribution are retained. Generic runtime classes remain for compatibility, but are not current experiment choices. Use `sf/run.py` for the current experiments; historical experiment runners and custom YAML entry points have been removed from this default snapshot.

From the repository root:

```sh
python sf/prepare_data.py bijie --raw /data/Bijie-landslide-dataset --output outputs/bijie
python sf/run.py source --model yolo11n --pretrained /weights/yolo11n.pt --data outputs/bijie/data.yaml --output outputs/yolo11n_seed0 --seed 0
python sf/run.py source --model ctpa --pretrained /weights/yolo11n.pt --data outputs/bijie/data.yaml --output outputs/ctpa_seed0 --seed 0
python sf/run.py compress --checkpoint outputs/ctpa_seed0/weights/best.pt --output outputs/sf_initial_seed0 --seed 0
python sf/run.py recover --model sf --checkpoint outputs/sf_initial_seed0/initial.pt --data outputs/bijie/data.yaml --output outputs/sf_seed0 --seed 0
python sf/run.py recover --model ctpa --checkpoint outputs/ctpa_seed0/weights/best.pt --data outputs/bijie/data.yaml --output outputs/ctpa_control_seed0 --seed 0
```

Repeat seeds 0, 1, 2. Before recovery, screening compares all three variants in `sf/sf_lowrank.py` using Bijie validation only; 48/32/80 was selected. For each recovery run, `selection.json` identifies the checkpoint selected using validation, which may retain the initial model.

```sh
python sf/run.py evaluate --checkpoint /selected/source.pt --data outputs/bijie/data.yaml --output outputs/source_test --split test
python sf/prepare_data.py cas --domain jiuzhai_05 --raw '/data/Jiuzhai valley (UAV-0.5m)' --output outputs/jiuzhai
python sf/run.py evaluate --checkpoint /selected/source.pt --data outputs/jiuzhai/data.yaml --output outputs/before_jiuzhai --split test
python sf/run.py adapt --model sf --checkpoint /selected/source.pt --data outputs/jiuzhai/data.yaml --output outputs/adapt_jiuzhai_sf_seed0 --seed 0
python sf/run.py evaluate --checkpoint outputs/adapt_jiuzhai_sf_seed0/weights/best.pt --data outputs/jiuzhai/data.yaml --output outputs/after_jiuzhai_sf_seed0 --split test
```

The custom trainer preserves factorized layers; rebuilding from ordinary CTPA YAML can erase SF compression. Source/adapted checkpoints use identical target test lists. Adaptation uses 200 epochs with labeled target training data and target-validation checkpoint selection. It is not zero-shot transfer. Protocol files record optimizer and augmentation settings. `source` also supports YOLOv8n/YOLOv10n with their respective COCO weights. Consult `results/common_baselines_status.json`; pending seeds are not final aggregates.

Checkpoint hashes identify actual study artifacts. **Weight binaries are not included in this commit**; the commands reconstruct/train the models. Hashes are not a promise of bit-for-bit independent training reproduction. Numeric tables retain original GPU evaluations; the PR audit records differences from native fixed-checkpoint CPU exports.

## Interpretation and provenance

`results/` includes all current Bijie seed results, complete five-stratum zero-shot scores and before/after target adaptation. Jiuzhai and Mengdong were selected post hoc as main-text cases; other strata are retained. SF does not uniformly outperform both comparators across targets. Positive-only cohorts do not estimate background-only-scene false-alarm rates. Three positive source compression-minus-control differences are descriptive rather than a general significance claim.

`figures/` contains code-based diagrams/charts, not imagery panels. `checkpoints/` contains identifiers/hashes, not binaries. `${PROJECT_ROOT}` tokens in archived protocols denote workspace-relative paths; the portable runner uses your CLI paths. No DOI is assigned to this Git snapshot. See `CHANGELOG.md` for the revision transition.


## Completed source comparison

All six common-baseline runs completed 200 epochs each. Values below use original GPU test evaluations.

| Model | Parameters | mAP50 (%) | mAP50–95 mean±SD (%) |
|---|---:|---:|---:|
| yolov8n | 3,011,043 | 95.59 | 57.26±2.72 |
| yolov10n | 2,707,430 | 89.19 | 49.71±6.92 |
| YOLO11n baseline | 2,590,035 | 90.00 | 51.05±2.34 |
| CTPA continuation control | 2,493,267 | 95.41 | 56.34±4.55 |
| CTPA-SF | 2,403,155 | 95.68 | 56.97±4.55 |

SF is 0.29 percentage points below the YOLOv8n mean mAP50–95, with 20.19% fewer parameters. These are descriptive comparisons; source training budgets differ as stated above. No uniformly best-model claim is made.


## Completed component ablations (2026-11-26)

Nine leave-one-compression-out recoveries (three configurations × seeds0,1,2,200epochs each) and nine fixed test evaluations are complete. Restoring full-rank attention, FFN or fusion gives mean Bijie test mAP50–95 of57.10%,57.51%,57.30%, compared with56.97% for full SF. Parameter counts increase to2,427,731 /2,444,115 /2,427,731 from2,403,155. Full SF remains the minimum-parameter configuration, not the highest-accuracy variant in this table. Differences are exploratory descriptive results on a previously inspected test split, not proof of significance. All nine outcomes are retained in results/component_ablation.json.

Use sf/ablation.py initialize --restore attention (or ffn/fusion) --checkpoint /source/ctpa_seed0.pt --output outputs/ablation_initial, then sf/ablation.py recover --checkpoint outputs/ablation_initial/initial.pt --data outputs/bijie/data.yaml --output outputs/ablation_recovery --seed 0. Use the same seeds and teacher checkpoints as the full-SF experiment, restore only one group, and use validation-only checkpoint selection. No main-model replacement or additional target fine-tuning follows automatically.
