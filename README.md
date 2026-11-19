# CTPA-SF landslide detection reproduction package

This source snapshot supports the revision of PONE-D-26-46654,
Lightweight Local-Global CNN-Transformer Collaborative Learning for Landslide Detection in Complex Mountainous Remote Sensing Imagery.

The public Bijie comparison uses three matched training seeds. All individual
results, including exploratory variants and low cross-domain scores, are included
in results/. The manuscript's original SF model remains the primary model.

## Contents

- ultralytics/ contains the archived detection implementation used in these runs.
  It reports version 8.3.185 and includes the CTPA and SharedFormer modules.
- plos_revision/ contains preparation, training, evaluation and aggregation scripts.
- splits/ contains image identifiers, exported pixel hashes and Bijie split membership.
- results/ contains seed-level outcomes, validation comparisons, source-conversion
  checks and inference/timing protocols.
- source_checksums.json identifies the copied runtime source files.

The software is distributed under AGPL-3.0 with the upstream license and
attribution retained. Source changes include custom CTPA modules and model
configurations and the revision experiment scripts. Dataset terms remain those
of the original providers.

## Environment

Experiments used Python 3.10.20 and an NVIDIA RTX 3080, with PyTorch
2.0.1+cu118 and torchvision 0.15.2+cu118. requirements.txt records the other
observed package versions. Install the CUDA-compatible torch/torchvision pair
appropriate to your environment, then install requirements.txt.

Use this repository's ultralytics directory. The training runner prepends the
specified repository to the import path. A newly installed upstream package may
have different custom-module registration and training behavior.

## Data

Bijie source imagery, positive masks and negative images are available from
https://gpcv.whu.edu.cn/data/Bijie_pages.html .
CAS source imagery and masks are available from
https://doi.org/10.5281/zenodo.10294997 .
Acquire them through their original sources, following the source terms.

The Bijie release contains 770 positive and 2,003 negative images. Each
eight-connected foreground mask component becomes one enclosing axis-aligned
box. No minimum area filter beyond one foreground pixel is used.

For the optional CAS external preparation, create a UTF-8 CSV manifest with columns
region_id,sensor,image_path,mask_path. List every image/mask pair from
Moxi town UAV-0.2m and UAV-1m, with region_id set to Moxi town. The image and
mask paths must point to your locally extracted files. The released identifiers
are listed in splits/cas_moxi_manifest.csv. Pass this local index with --cas-manifest when preparing CAS. Bijie-only reproduction can be run directly with the commands below.

The Motuo imagery is licensed to the authors for internal research. Access on
equivalent terms has not been established. The publicly reproducible benchmark
uses Bijie. The interpretation of restricted Motuo analyses remains subject to
the journal's review of the Data Availability Statement.

## Reproduce Bijie

Run these commands from the repository root, substituting local source paths.

    python plos_revision/prepare_external.py --bijie /data/Bijie-landslide-dataset --output outputs/external
    python plos_revision/prepare_public_benchmark.py --bijie outputs/external/bijie --output outputs/bijie_benchmark --seed 42

The resulting public split has 1,941 training, 555 validation and 277 test
images, containing 539, 154 and 77 instances. Compare the generated manifest to
splits/bijie_manifest.csv before training.

Download the official COCO-pretrained yolo11n.pt to weights/yolo11n.pt.
The same source checkpoint is used for all shape-compatible layers.

    python plos_revision/run_queue.py --repo . --data outputs/bijie_benchmark/bijie_benchmark.yaml --output outputs/public --models yolo11n ctpa sf --seeds 0 1 2 --epochs 60 --patience 15 --batch 8 --split val
    python plos_revision/run_study.py --repo . --data outputs/bijie_benchmark/bijie_benchmark.yaml --output outputs/public --models yolo11n ctpa sf --seeds 0 1 2 --stage evaluate --split test

The first command trains/evaluates sequentially and skips completed matching
training jobs. A partial training folder is reported for manual diagnosis.
Best checkpoints use source validation fitness, 0.1 mAP50 + 0.9 mAP50–95.

For exploratory variants, the model names are sf_full_qkv, sf_wide_ffn,
sf_no_eca, sf_fixed_weights and sf_qkv16. In the reported work, all variants
were inspected on Bijie validation, and sf_no_eca and sf_wide_ffn were then
evaluated on its test split. These follow-up tests are exploratory.

## Interpretation

On Bijie test, SF achieves 45.87 ± 4.22% mAP50–95 and the baseline achieves
43.17 ± 0.77%. One of three seed-matched strict-IoU differences is negative.
SF has 6.4% fewer parameters and higher measured FP32 batch-one latency.
Motuo-to-Bijie and Motuo-to-Moxi transfer performs poorly. TTA and input-size
follow-ups use already-inspected external datasets and are labeled exploratory.

S1 reports both validation and test rows, identified by split and data columns.
S2 contains every fixed exploratory inference configuration. S3 summarizes the
eight Bijie validation configurations. S4 records the conversion audit, S5 the
timing protocol, and S6 the external inference protocol.

Public code and derived results: https://github.com/mask-lmk/CTPA-SF-landslide. Use a commit-specific URL when citing the exact revision. No DOI has been assigned.
