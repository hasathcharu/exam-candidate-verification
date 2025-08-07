# ✍️ Exam Candidate Verification Through Handwritten Artifacts

This repository contains our final year research project demonstration titled **Exam Candidate Verification Through Handwritten Artifacts**, completed at the **Faculty of Information Technology, University of Moratuwa, Sri Lanka**.

Biometrics identify individuals using physical or behavioral traits. While physiological biometrics like fingerprints are common, handwriting remains a reliable behavioral biometric used in legal and educational settings. Questioned Document Examination (QDE) helps verify document authorship using handwriting. While physiological biometrics like fingerprints are widely used, handwriting remains crucial in education and assessments due to its resistance to forgery over long texts. However, manual handwriting verification is slow and expert-dependent, making it impractical in large-scale scenarios like exam fraud detection. Recent machine learning advancements offer a scalable solution, but lack of transparency limits trust in automated decisions. This project addresses that by integrating explainable handwriting verification, enabling users to understand and trust the system’s predictions.

## 🧩 Project Modules

Our solution is divided into three major modules:

#### 1. **Signature Forgery Detection using Vision and Text Embeddings**

This module detects offline signature forgeries using CLIP, without requiring prior samples from the individual. It addresses challenges like intra-personal variations common in exam settings. Instead of comparing signatures directly, the model uses visual–textual associations to assess authenticity.

#### 2. **Quick Handwriting Verification with Automatic Feature Extraction**

This module verifies handwriting using automatically extracted features from compact texture representations. It supports two verification modes:

- Standard Mode: Uses one sample per writer.
- Two-Speed Mode: Uses both normal and fast handwriting samples to handle intra-writer variability.

This approach is fast and accurate, though less interpretable than manual methods.

#### 3. **Personalized Handwriting Verification with Manual Feature Extraction**

This writer-dependent module uses manually extracted features to enhance explainability. It combines:

- Global traits (e.g., pressure, curvature)
- Local features from frequently used letters (like e)

It also integrates SHAP-based explanations to provide feature-level interpretability, helping reviewers understand which handwriting traits influenced each decision.

## 📐 High Level Architecture of the Overall System

![System Architecture](assets\highlevel-architectur.jpg)

The system accepts both signature and handwriting samples. Initially, a rapid assessment is performed using the signature forgery detection module (Module 1) and the quick handwriting verification module (Module 2), both of which aim to provide a preliminary decision with high efficiency.

If the predictions from these independent modules are in agreement and demonstrate high confidence, a final verification decision is rendered. In scenarios where confidence is insu!cient or disagreement occurs, the system moves to a more thorough verification process using personalized writer verification module (Module 3). This involves acquiring additional handwriting samples and leveraging a personalized verification module
that uses interpretable, manually engineered features.

To ensure reliability and transparency, outputs from all modules are integrated through a voting mechanism, which consolidates the predictions into a final decision
and to provide explanations for the final decision as well.
