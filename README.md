# Aorta Segmentation from 4D Flow MRI using Deep Learning

This repository contains an implementation of a **deep learning–based segmentation pipeline** for extracting the **thoracic aorta** from **4D Flow MRI magnitude images**, inspired by the methodology described in:

**“Segmentation of the Aorta in Systolic Phase from 4D Flow MRI: Multi‑Atlas vs Deep Learning” (Marin‑Castrillón et al., 2023)**

This project focuses on:
- Preprocessing 4D Flow MRI data  
- Extracting the systolic phase  
- Training a 3D U‑Net model for aorta segmentation  
- Evaluating the model using Dice Score, Hausdorff Distance, etc.  
- Preparing the segmented aorta geometry for potential CFD workflows  

---

## 🧠 **Project Overview**

Thoracic aortic aneurysm assessment increasingly relies on **hemodynamic biomarkers**, which require accurate segmentation of the aorta.  
4D Flow MRI provides both anatomy (magnitude images) and velocity fields, making it ideal for biomarker extraction — but segmentation is challenging due to noise and low contrast.

This repository aims to provide a reproducible deep-learning segmentation pipeline for:
- 4D Flow MRI magnitude preprocessing  
- Aortic lumen segmentation  
- Comparison with ground truth masks  
- Easy extension to 4D (time‑resolved) segmentation  

---

## 📂 **Repository Structure**

``