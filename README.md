Neural Latent Mapping Experiment
Bidirectional latent modeling with Encoder / Decoder / Inverse Encoder
This project investigates how the latent representation z can be reinterpreted as a structural variable connecting input x and output y.
Instead of relying only on the standard end-to-end direction (x → z → y), this work introduces a bidirectional mapping:
Forward path: x → Encoder → z → Decoder → y
Inverse path: y → Inverse Encoder → z → Decoder → y
The purpose is to analyze how latent variables behave when mapped from both x and y, and how different training orders affect the quality, stability, and interpretability of the representations.
Purpose and Background
Neural networks usually treat the latent space as a black-box intermediate representation.
This project redefines the latent as a structural connector between x and y by:
Training Encoder and Inverse Encoder to satisfy f(x) ≈ g(y)
Using a shared Decoder in both directions
Viewing the x → y transformation as a measure transformation through z
Structuring the model so x-side and y-side processing can be treated separately
The focus is not self-reconstruction, but structural understanding of the mapping between x and y.
Project Structure (based on your description)
exp.ipynb
Demonstrates the entire experimental flow, including model behavior, bidirectional paths, training orders (flows), and evaluation.
modular_nn_experiment1/
Contains the .py implementations of all models (MiniViT, MiniUNet, MLP, PCA+MLP, etc.), their modular components, experiment notebooks, and saved experimental results such as loss and cosine similarity metrics.
idea_explanation.pdf
A detailed description of the theoretical motivation, model architecture, and the reasoning behind the bidirectional design.
idea_summary.pdf
A concise summary of the ideas and findings.
(This README is primarily based on that document.)
Key Findings
Joint learning of Encoder, Decoder, and Inverse Encoder performs poorly.
The loss often does not decrease sufficiently, especially for Transformer-based architectures such as ViT.
Training order has a strong influence on performance.
Two training flows were tested:
Path 1: Train Encoder + Inverse Encoder → Train Decoder → Fine-tune all
Path 2: Train Inverse Encoder + Decoder → Freeze Inverse Encoder → Train Encoder → Fine-tune all
Path 2 consistently performs better and in many cases surpasses standard end-to-end learning.
Decoder strength greatly affects training difficulty.
If the Decoder quickly becomes very strong, Encoder training becomes harder, although the overall average loss can still fall below end-to-end performance.
Some models (ViT, MLP) show no statistical difference from end-to-end training
according to t-tests and KS tests.
This indicates that the bidirectional design can match or exceed end-to-end quality.
Decoder quality is the most critical factor.
A strong Decoder contributes more to final representation quality than minimizing latent matching losses.
Theoretical View
The latent z is viewed as a measure-transforming intermediate space between the distributions of x and y.
Encoder f(x) and Inverse Encoder g(y) aim to produce structurally aligned latent representations.
A shared Decoder enforces consistency across both forward and inverse mappings.
This bidirectional design increases interpretability by allowing explicit examination of how x and y relate through z.
Models and Implementation
Models included in the project:
MiniViT (attention-based, global information aggregation)
MiniUNet (convolution-based, strong local structure)
MLP (dense compression)
PCA + MLP (explicit low-dimensional basis followed by MLP)
Implementation uses:
jaxtyping for type-safe array specifications
einops for clean tensor manipulation
a custom Dense layer for consistent initialization
Fourier time embeddings and label embeddings for conditioning
Experimental Results (Highlights)
Examples of performance comparisons:
Model	End-to-End Loss	Path 2 Loss	End-to-End Cosine	Path 2 Cosine
MiniViT	1136.6	1128.6	0.640	0.643
MiniUNet	1177.4	1134.8	0.623	0.640
MLP	1172.8	1161.6	0.625	0.629
PCA+MLP	1250.3	1256.2	0.594	0.590
Path 2 achieves equal or better performance than end-to-end in most cases.
Summary
This project proposes a bidirectional latent learning framework in which the latent variable is treated as a structural connector rather than a single-pass intermediate feature.
The main contributions include:
Forward and inverse mapping paths
Flow-matching style training strategies
Demonstrating that non-end-to-end training can match or surpass standard training
Providing a modular and interpretable foundation for studying latent mappings
