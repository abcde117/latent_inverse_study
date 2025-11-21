Neural Latent Mapping Experiment
Bidirectional latent modeling with Encoder / Decoder / Inverse Encoder
This project investigates how the latent representation z can be reinterpreted as a structural variable connecting input x and output y.
Unlike standard end-to-end learning (x → z → y), this work introduces a bidirectional mapping:
Forward path: x → Encoder → z → Decoder → y
Inverse path: y → Inverse Encoder → z → Decoder → y
The goal is to evaluate whether latent variables can be made structurally consistent when mapped from both x and y, and how different training orders affect the representation quality and learning stability.
Purpose and Motivation
Neural networks typically treat the latent space as an opaque intermediate representation.
This project redefines the latent as a structural connector between x and y by:
Training Encoder and Inverse Encoder to satisfy f(x) ≈ g(y)
Sharing a common Decoder for both directions
Understanding x→y as a measure transformation mediated by z
Separating x-side and y-side processing to increase interpretability
This framework is related to autoencoders, but instead of focusing on self-reconstruction, the emphasis is on structural understanding between input and target.
Project Structure (as described)
exp.ipynb
A complete demonstration of the entire experiment flow, including model definitions, training paths, and evaluation.
modular_nn_experiment1/
Contains all modularized .py model files (MiniViT, MiniUNet, MLP, PCA+MLP, etc.), experimental notebooks using these Python models, and collected experiment results (loss curves, cosine similarity, comparison tables).
idea_explanation.pdf
A detailed explanation of the theoretical motivation, model architecture, and the design of the two bidirectional paths.
idea_summary.pdf
A concise summary of the core ideas and experimental findings (this README is based mainly on this document).
Key Findings
1. Joint training of Encoder + Decoder + Inverse Encoder performs poorly
Loss reduction becomes unstable or insufficient, especially for Transformer-based models (ViT).
2. Training order (flow) has a major impact on performance
Two training strategies were tested:
Path 1
Train Encoder & Inverse Encoder together
Train Decoder
Fine-tune all modules
Path 2
Train Inverse Encoder + Decoder first
Freeze Inverse Encoder
Train Encoder
Fine-tune all modules
→ Path 2 consistently outperforms Path 1 and often surpasses standard end-to-end training.
3. When the Decoder becomes too strong early, Encoder training becomes difficult
The latent signal becomes weak, but overall loss can still be lower than end-to-end.
4. ViT and MLP show statistical results comparable to end-to-end
t-test and KS-test sometimes detect no significant difference, meaning the bidirectional structure can match or exceed conventional training.
5. Decoder quality is the most important factor
A well-designed Decoder contributes more to the final performance than minimizing latent-matching loss.
Theoretical Viewpoint
Latent z acts as a measure-transforming intermediate space between distributions of x and y.
Encoder f(x) and Inverse Encoder g(y) aim to produce structurally identical latents.
Shared Decoder enforces consistency in both forward and inverse directions.
The bidirectional setup improves interpretability and modularity, enabling more explicit reasoning about the mapping structure between x and y.
Models and Implementation
The project includes modular implementations of:
MiniViT (Attention-based) – global information weighting
MiniUNet (Conv-based) – strong local spatial structure
MLP – dense compression
PCA + MLP – explicit low-dimensional basis + linear extraction
All models follow a unified design using:
jaxtyping for type safety
einops for tensor manipulation
Custom Dense layers for consistent initialization
Fourier time embeddings and label embeddings as additional conditioning signals
Experimental Results 
Example comparisons:
Model	End-to-End Loss	Path2 Loss	End-to-End Cos	Path2 Cos
MiniViT	1136.6	1128.6	0.640	0.643
MiniUNet	1177.4	1134.8	0.623	0.640
MLP	1172.8	1161.6	0.625	0.629
PCA+MLP	1250.3	1256.2	0.594	0.590
Most models achieve equal or better performance under Path 2 compared to end-to-end.
Overall Summary
This project demonstrates a framework where the latent space is treated as a bidirectional structural variable instead of a fixed intermediate point.
Key contributions include:
Constructing forward and inverse mapping paths
Comparing flows to reveal strong dependence on training order
Showing that non-end-to-end training can reach or surpass standard training
Providing a modular, interpretable design for analyzing latent mappings
