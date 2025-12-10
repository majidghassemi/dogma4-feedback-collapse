# Dogma 4: Feedback Collapse and Internal-Feedback Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Research_Preview-blue)]()

**Official code repository for the paper: "Dogma 4: Feedback Collapse and the Case for Internal-Feedback Agents."**

## 🚨 The Problem: Feedback Collapse
Modern AI alignment (e.g., RLHF) relies on a hidden assumption we call **Dogma 4**: 
> *"Feedback is exogenous, truthful, and fixed."*

In reality, social feedback is noisy, biased, and often **sycophantic**. We prove that standard RL agents trained on such feedback suffer from **Feedback Collapse**—converging to policies that maximize observed reward but fail the latent true objective (e.g., safety).

## 🛡️ The Solution: Internal-Feedback Agents
We introduce a new agent architecture that weighs external feedback against internal **epistemic and ethical source critics**. By modeling *who* to trust, these agents recover aligned policies even when the majority of feedback is biased.

## 🚀 Quick Start

### Installation
```bash
git clone [https://github.com/yourusername/dogma4-feedback-collapse.git](https://github.com/yourusername/dogma4-feedback-collapse.git)
cd dogma4-feedback-collapse
pip install -r requirements.txt
