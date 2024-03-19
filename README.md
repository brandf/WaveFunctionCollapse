
# nanoGPT
This exploration is based on the awesome [nanoGPT](https://github.com/karpathy/nanoGPT) repository. For the original README, see upstream.

# Wave Function Collapse
Wave Function Collapse (WFC) is an idea I have for how to build langauge models using Transformers.  It takes inspiration from a loose analogy to Quantum Mechanics, however it's not required that you understand QM to understand how WFC works. This repository both describes the idea and will attempt to implement/validate small scale models.

The goal of WFC is to be used for generative AI, similar to popular transformer based models such as ChatGPT. The concept isn't specific to text LLMs, however text is a familar and a relatively low cost way to validate it.  

The WFC approach is a bit different from your typical decoder-only LLM in terms of architecture, training, inference, and loss, but it's compatible with the same training sets and makes use of standard Transformer blocks.  Before we get into the details of this idea, let's do a quick review of a typical decoder-only LLM model:


