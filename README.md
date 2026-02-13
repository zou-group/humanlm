# HumanLM: Simulating Users with State Alignment Beats Response Imitation

<div align="left">

[![](https://img.shields.io/badge/Website-HumanLM-purple?style=plastic&logo=Google%20Chrome)](https://humanlm.stanford.edu/)
[![](https://img.shields.io/badge/Datasets_&_Models-HuggingFace-yellow?style=plastic&logo=Hugging%20Face)](https://huggingface.co/snap-stanford/collections)
[![](https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arxiv)](https://humanlm.stanford.edu/HumanLM_paper.pdf)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

Can language models truly act like specific humans — not just produce humanlike text, but reflect individual values, opinions, and communication styles? **HumanLM** tackles this challenge by aligning LMs to internal user *states* (stances, beliefs) rather than merely imitating surface-level responses.


## Quick Start

### 1. Data Collection & Processing

We provide end-to-end tooling for collecting raw data from six sources and processing them into train/val/test splits with LLM-generated user personas. See [`humanual_datasets/README.md`](humanual_datasets/README.md) for full instructions.

### 2. Human Evaluation

The user study interface lets annotators compare their own responses against model-generated ones on Reddit posts.

```bash
# Start the required vLLM model servers
vllm serve Qwen/Qwen3-8B --dtype auto --host 0.0.0.0 --port 8000 --tensor-parallel-size 3 --max-model-len 7168
vllm serve snap-stanford/humanlm-opinions --dtype auto --host 0.0.0.0 --port 63456 --tensor-parallel-size 2 --max-model-len 7168

# Launch the Gradio annotation interface
cd user_study
python gradio_app.py          # add --debug to skip validation constraints
```

### 3. Training

Coming soon.

### Bibtex

```bibtex
@article{wu2026humanlm,
  title={HUMANLM: Simulating Users with State Alignment Beats Response Imitation},
  url={https://humanlm.stanford.edu/},
  author={Wu, Shirley and Choi, Evelyn and Khatua, Arpandeep and
          Wang, Zhanghan and He-Yueya, Joy and Weerasooriya, Tharindu Cyril and
          Wei, Wei and Yang, Diyi and Leskovec, Jure and Zou, James},
  year={2026}
}
```
