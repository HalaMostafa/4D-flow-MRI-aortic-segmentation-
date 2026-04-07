# Aorta Segmentation from 4D Flow MRI using Deep Learning

This repository contains an implementation of a deep learning-based segmentation pipeline for extracting the thoracic aorta from 4D Flow MRI magnitude images, inspired by:

"Segmentation of the Aorta in Systolic Phase from 4D Flow MRI: Multi-Atlas vs Deep Learning" (Marin-Castrillon et al., 2023).

This project focuses on:
- Preprocessing 4D Flow MRI data
- Extracting the systolic phase
- Training a 3D U-Net model for aorta segmentation
- Evaluating the model using Dice Score, Hausdorff Distance, etc.
- Preparing the segmented aorta geometry for potential CFD workflows

---

## Project Overview

Thoracic aortic aneurysm assessment increasingly relies on hemodynamic biomarkers, which require accurate segmentation of the aorta.
4D Flow MRI provides both anatomy (magnitude images) and velocity fields, making it ideal for biomarker extraction, but segmentation is challenging due to noise and low contrast.

This repository aims to provide a reproducible deep-learning segmentation pipeline for:
- 4D Flow MRI magnitude preprocessing
- Aortic lumen segmentation
- Comparison with ground truth masks
- Easy extension to 4D (time-resolved) segmentation

---

## Repository Structure

```
.
├── preprocessing/
├── segmantation_of_aorta_from_satndard_mri/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── requirements-prod.txt
```

---

## Docker Quick Start

This project provides a minimal Docker setup with:
- `dev` stage: full data-science environment (Jupyter + plotting + nnUNet v2)
- `production` stage: minimal runtime image (no Jupyter)

### Prerequisites

- Docker and Docker Compose
- NVIDIA driver + NVIDIA Container Toolkit (for GPU usage)

### Start development environment

```bash
make build
make up
make logs
```

Open Jupyter Lab in your browser:
- URL: `http://localhost:8889`
- Token: `aorta-seg-dev-token`

The repository root is bind-mounted to `/workspace` in the container, so notebook edits and outputs are saved directly to your local project files.

### Stop development environment

```bash
make down
```

### Build production image (minimal)

```bash
make build-prod
```
