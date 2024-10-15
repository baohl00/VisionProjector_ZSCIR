# Vision Projector for ZSCIR

This repo contains implementation of MagicLens. The code here uses Jax and Flax.
Note that the current implementation does not yet support training.
For further information about MagicLens, please visit the [website](https://open-vision-language.github.io/MagicLens/).

## Abstract

We introduce MagicLens, a series of self-supervised image retrieval models that support
open-ended instructions. The core thesis of MagicLens is that text
instructions can enable retrieving images with
richer relations beyond visual similarity. MagicLens is built on a
key novel insight: image pairs that naturally occur
on the same web pages contain a wide range of implicit relations (e.g., inside view of), and we
can bring those implicit relations explicit by synthesizing instructions via large multimodal models (LMMs) and large language models (LLMs).
Trained on 36.7M (query image, instruction, target image) triplets with rich semantic relations
mined from the web, MagicLens achieves comparable or better results on eight benchmarks of
various image retrieval tasks than prior state-of-the-art (SOTA) methods. Remarkably, it outperforms previous SOTA but with a 50× smaller
model size on multiple benchmarks. Additional
human analyses on a 1.4M-image unseen corpus
further demonstrate the diversity of search intents
supported by MagicLens.
![Intro image](https://open-vision-language.github.io/MagicLens/static/images/magiclens_overview.png)

## Setup
```
conda create --name magic_lens python=3.9
conda activate magic_lens
git clone https://github.com/google-research/scenic.git
cd scenic
pip install .
pip install -r scenic/projects/baselines/clip/requirements.txt
# you may need to install corresponding GPU version of jax following https://jax.readthedocs.io/en/latest/installation.html
# e.g.,
# # CUDA 12 installation
# Note: wheels only available on linux.
# pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# # CUDA 11 installation
# Note: wheels only available on linux.
# pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Model Download & Data Preparation
Please follow the instruction in [here](https://github.com/google-deepmind/magiclens/blob/main/data/README.md).

## Inference
We prepare two different files for inference stage. You can choose base or large version of MagicLens, if you run this:
```
bash scripts/inference.sh
```
or run both two versions:   
```
bash scripts/fast.sh
```

Due to the weight conversion, the performance may be slightly different:

In `CIRCO`
| Model | map@5 | map@10 | map@25 | map@50 |
|----------|----------|----------|----------|----------|
| Prior SOTA | 26.8 | 27.6 | 30.0 | 31.0 |
| Base (original) | 23.1 | 23.8 | 25.8 | 26.7 |
| Base (reproduced) | 25.5 | 26.5 | 28.5 | 29.4 |
| Base + _VisionProjector_ | 25.7 | 26.7 | 28.6 | 29.6 |
| Large (original) | 29.6 | 30.8 | 33.4 | 34.4 |
| Large (reproduced) | 30.1 | 31.4 | 33.8 | 34.9 |
| Large + _VisionProjector_ | **35.8** | **36.8** | **39.3** | **40.4** |

## Citing this work

Add citation details here, usually a pastable BibTeX snippet:

```latex
@inproceedings{zhang2024magiclens,
  title={MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions},
  author={Zhang, Kai and Luan, Yi and Hu, Hexiang and Lee, Kenton and Qiao, Siyuan and Chen, Wenhu and Su, Yu and Chang, Ming-Wei},
  booktitle={The Forty-first International Conference on Machine Learning (ICML)},
  year={2024},
  pages={to appear}
}
```

## Acknowledgement 

We extend our gratitude to the open-source efforts of [MagicLens](https://github.com/google-deepmind/magiclens). 
