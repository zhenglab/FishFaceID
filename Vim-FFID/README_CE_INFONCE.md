# Usage Guide for CE+InfoNCE Combined Loss Function

## Overview

CE+InfoNCE combined loss is a hybrid loss function that combines cross-entropy classification loss and InfoNCE contrastive loss, leveraging the advantages of both:
- Cross Entropy (CE) loss: Provides stable classification supervision signal
- InfoNCE contrastive loss: Enhances discriminability in feature space, making same-class features more compact and different-class features more dispersed

## Parameter Settings

To use CE+InfoNCE combined loss, the following parameters need to be set:

```bash
--main-loss-type ce_infonce  # Specify the use of CE+InfoNCE combined loss
--contrast-weight 0.5        # Set the weight for InfoNCE loss (default 0.5, meaning InfoNCE loss weight is half of CE loss)
--infonce-temp 0.07          # Temperature parameter for InfoNCE loss (optional, default 0.07)
```

## Output Format Requirements

When using `ce_infonce` as the `main-loss-type`, the model needs to return output in triplet format:
```
(logits, image_features, prompt_features)
```

- `logits`: Classification logits with shape [batch_size, num_classes]
- `image_features`: Image feature vectors with shape [batch_size, embed_dim]
- `prompt_features`: Class prompt feature vectors with shape [num_classes, embed_dim]

## Auxiliary Losses

CE+InfoNCE combined loss also supports auxiliary losses, which can be set through the following parameters:

```bash
--aux-losses center arcface  # Add center loss and ArcFace loss as auxiliary losses
--aux-weights 0.1 0.2        # Set weights for corresponding auxiliary losses
```

Supported auxiliary losses include:
- `center`: Center loss, pulls same-class features closer
- `triplet`: Triplet loss, used for metric learning
- `arcface`: ArcFace loss, increases angular margin between class boundaries
- `cosface`: CosFace loss, increases cosine margin between class boundaries
- `cosine_sim`: Cosine similarity loss, used for feature alignment

## Training Example

Below is an example command for training a model using CE+InfoNCE combined loss:

```bash
python main_clip.py \
    --model vim_tiny_patch16_224_clip_prompts \
    --batch-size 128 \
    --data-path /path/to/dataset \
    --output_dir ./output/ce_infonce_test \
    --main-loss-type ce_infonce \
    --contrast-weight 0.5 \
    --infonce-temp 0.07 \
    --aux-losses center \
    --aux-weights 0.1 \
    --prompts-per-class 5
```

## Notes

1. Model implementation needs to support returning output in triplet format `(logits, image_features, prompt_features)`
2. When the main loss is set to `ce_infonce`, mixed sample augmentation (Mixup/Cutmix) might affect the performance of the InfoNCE component
3. Monitor the CE loss and InfoNCE loss values separately in the training logs to determine if `contrast-weight` needs adjustment
4. It's recommended to use the `--prompts-per-class` parameter to set the number of prompts per class to enhance contrastive learning effects