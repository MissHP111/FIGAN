# FIGAN: Diversity-Oriented Traffic Generation for Industrial Protocol Format Inference

## Abstract

Protocol Format Inference is a pivotal step in the reverse engineering of proprietary protocols, yet its effectiveness is constrained by the scarcity of high-quality training data. In industrial control systems, the rigid and cyclical nature of traffic results in a "long-tail" distribution, where diverse functional scenarios are severely underrepresented. Existing generative approaches, primarily designed for fuzzing or intrusion detection, fail to resolve the intrinsic conflict between syntactic validity and semantic diversity required for protocol format inference. To bridge this gap, we propose FIGAN, a stage-wise decoupled generative framework tailored to synthesize high-fidelity traffic for protocol format inference. By isolating flexible distribution learning from rigid syntax enforcement, FIGAN liberates the generative process to extrapolate novel payload variations from a continuous latent space, effectively surmounting the limitations of sparse seed data. Specifically, the framework integrates three synergistic modules: first, heuristic pre-processing that constructs semantic templates as a prior knowledge base; second, a generative adversarial architecture optimized via discrete relaxation to explore high-dimensional payload patterns independently of syntax rules; and finally, a closed-loop verification mechanism that performs syntactic calibration and functional validation against simulated device responses. Evaluations on four real-world protocols (Modbus TCP, S7Comm, Omron FINS, and DNP3) demonstrate that FIGAN significantly outperforms state-of-the-art baselines. By establishing a data foundation with superior validity and diversity, FIGAN facilitates robust downstream protocol inference.

---

## Index Terms

* Communication system traffic
* Data Augmentation
* Generative Adversarial Networks
* Industrial control
* Inference algorithms.

---

## Repository Contents

### `config.py`

Global configuration constants (model hyperparameters, protocol-specific parameters). Adjust `NUM_FUNC`, `HEADER_LEN`, and the validation logic in `evaluate.py` according to your dataset and target protocol.

### `model/model.py`

Generator and Critic network definitions.

### `encode.py`

Nibble-level tokenization and one-hot encoding of raw pcap traffic.

### `heuristic.py`

Protocol identifier and length field mining from raw pcap traffic.

### `train.py`

WGAN-GP adversarial training.

### `reconstruct.py`

Sequence reconstruction, syntactic calibration, and interaction-based verification.

### `evaluate.py`

Evaluation metrics: TIAR, DGD, DGB.

### `utils.py`

Training utilities: Gumbel-Softmax sampling, file I/O helpers.

### `workflow.py`

Full FIGAN pipeline entry point (Phase 1 → Phase 2 → Phase 3).

### `toy.pcap`

Toy Modbus TCP pcap dataset for quick verification.

---

## Requirements

* Python 3.8+
* torch
* numpy
* scapy
* scipy

Install dependencies:

```bash
pip install torch numpy scapy scipy
```

---

## Quick Start

Run the full pipeline on the toy dataset:

```bash
python workflow.py
```

Or use the `run()` function programmatically:

```python
from workflow import run

augmented = run(
    pcap_path="toy.pcap",
    onehot_path="toy_onehot.npy",
    n_gen=100,
)
```

To adjust training hyperparameters (epochs, batch size, etc.), modify `config.py` before running.

Note: To ensure seamless reproducibility without physical ICS hardware, the provided toy dataset utilizes a software-level mock for the interaction-based verification step.

---

## Adapting to Your Protocol

When applying FIGAN to a different ICS protocol, you need to:

1. Replace `is_valid()` in `evaluate.py` with your own protocol-specific validation
2. Update `NUM_FUNC` and `HEADER_LEN` in `config.py`
3. Implement `interaction_verify()` in `reconstruct.py` with real device replay logic

---

## License

This repository is provided for academic and research purposes.

---

## Citation

This paper is currently under review. The BibTeX citation will be provided once the paper is accepted and published. 
For now, if you use this framework, please refer to this repository link.
