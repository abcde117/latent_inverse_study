Neural Latent Mapping Experiment
Bidirectional latent modeling with Encoder / Decoder / Inverse Encoder
This project explores how the latent representation z can be treated as a structural variable connecting input x and output y. Instead of relying only on the standard end-to-end mapping (x → z → y), the project introduces a bidirectional structure:
Forward path: x → Encoder → z → Decoder → y
Inverse path: y → Inverse Encoder → z → Decoder → y
The goal is to understand whether latent variables derived from x and y can be aligned, how training order affects performance, and how this structure improves interpretability.
Purpose and Background
Traditional neural networks treat latent space as a black-box intermediate.
This project redefines latent space as a structural connector by:
Training Encoder and Inverse Encoder to satisfy f(x) ≈ g(y)
Using a shared Decoder for both directions
Viewing the x → y mapping as a measure transformation mediated by latent space
Structuring x-side and y-side processing so they can be treated separately and more interpretably
The aim is not self-reconstruction, but understanding the structural relationship between x and y through latent representations.
Project Structure (summary)
exp.ipynb
Demonstrates the full experimental flow, including forward/inverse paths, training procedures, and evaluation.
modular_nn_experiment1/
Contains modularized .py model implementations (MiniViT, MiniUNet, MLP, PCA+MLP, etc.), experiment notebooks using these models, and experimental results such as loss and cosine similarity outputs.
idea_explanation.pdf
Detailed theoretical explanation of the model design, bidirectional structure, and training strategies.
idea_summary.pdf
A concise summary of all major ideas and findings.
(This README is based primarily on that document.)
Key Findings
1. Joint training of all three modules performs poorly
Training Encoder, Decoder, and Inverse Encoder simultaneously leads to insufficient loss reduction, especially for Transformer-based models.
2. Training order strongly affects performance
Two training flows were tested:
Path 1
Train Encoder + Inverse Encoder
Train Decoder
Fine-tune all modules
Path 2
Train Inverse Encoder + Decoder
Freeze Inverse Encoder
Train Encoder
Fine-tune all modules
Path 2 consistently performs better and often surpasses standard end-to-end training.
3. Over-strong Decoder early in training makes Encoder learning difficult
Even so, the overall final loss can still be lower than end-to-end.
4. Some models show no statistical difference from end-to-end training
According to t-tests and KS-tests, ViT and MLP often show no significant difference compared to end-to-end training.
This means the bidirectional framework can match or exceed end-to-end performance.
5. Decoder quality matters more than latent matching loss
A well-designed Decoder is the most important factor in achieving good representation quality and overall performance.
Theoretical Perspective
Latent space is interpreted as a measure-transforming intermediate space between x and y.
Encoder f(x) and Inverse Encoder g(y) aim to produce consistent latent structures.
A shared Decoder enforces alignment between forward and inverse directions.
This separation of paths increases interpretability and provides clearer insight into the x ↔ y relationship.
Models and Implementation
Models included:
MiniViT — attention-based global aggregation
MiniUNet — convolution-based local structure
MLP — dense compression
PCA + MLP — explicit low-dimensional basis followed by MLP
Implementation uses:
jaxtyping for type-safe array annotations
einops for clean tensor manipulation
A custom Dense layer to unify initialization across models
Fourier time embeddings and label embeddings as conditioning inputs
Experimental Results (Highlights)
Model	End-to-End Loss	Path 2 Loss	End-to-End Cosine	Path 2 Cosine
MiniViT	1136.6	1128.6	0.640	0.643
MiniUNet	1177.4	1134.8	0.623	0.640
MLP	1172.8	1161.6	0.625	0.629
PCA+MLP	1250.3	1256.2	0.594	0.590
Path 2 achieves equal or better performance compared to end-to-end training for most models.
Summary
This project proposes a bidirectional latent learning framework where the latent variable is treated as a structural connector rather than a simple intermediate feature. The main contributions are:
Introducing forward and inverse mapping paths
Comparing training flows to reveal strong dependence on training order
Showing that non-end-to-end training can match or surpass end-to-end baselines
Providing a modular, interpretable architecture for analyzing latent mappings
