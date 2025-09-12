# IJCAI 2025 - PKA

This repository contains the official implementation of the paper:

**Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification**  
*Zhijie Lu, Wuxuan Shi, He Li, Mang Ye*  
IJCAI 2025  
[[Paper]](./ijcai2025-camera-ready.pdf)

---

## 🔍 Introduction
Lifelong Person Re-identification (LReID) aims to continually adapt to new environments while preserving previously acquired knowledge.  
However, most existing methods rely on storing exemplar data, which raises **privacy concerns**.  
To address this, we propose **PKA**, a **non-exemplar-based framework** with two key components:

- **Prototype-guided Knowledge Propagation (PKP):**  
  Utilizes prototypes with triplet constraints to separate old and new identity distributions, mitigating catastrophic forgetting.  

- **Adaptive Parameter Evolution (APE):**  
  Dynamically fuses model parameters across tasks by assessing parameter importance, ensuring effective integration of old and new knowledge.

PKA achieves **state-of-the-art performance** on five benchmark datasets (Market1501, CUHK-SYSU, DukeMTMC, MSMT17, CUHK03), outperforming existing prototype-based methods in both **seen** and **unseen domains**.

---

## 📂 Datasets
We follow the standard LReID benchmark setup and evaluate on:
- Market-1501
- CUHK-SYSU
- DukeMTMC-ReID
- MSMT17-V2
- CUHK03  

For generalization, additional unseen datasets (CUHK01, CUHK02, VIPeR, PRID, i-LIDS, GRID, SenseReID) are also tested.

---
## ✨ Citation
If you find this work useful, please cite:

```bibtex
@article{luprototype,
  title={Prototype-guided Knowledge Propagation with Adaptive Learning for Lifelong Person Re-identification},
  author={Lu, Zhijie and Shi, Wuxuan and Li, He and Ye, Mang}
}
