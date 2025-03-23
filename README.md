# Build nanoGPT from Scratch

This repository contains my step-by-step implementation of the **nanoGPT** project, following along with Andrej Karpathy's YouTube tutorial: [Build nanoGPT](https://www.youtube.com/watch?v=l8pRSuU81PU&t=2150s). The aim of this project is to recreate the **nanoGPT** model from scratch, starting with an empty file and gradually building it up to a working version of the GPT-2 (124M) model. 

Each commit in this repository corresponds to a specific step in the process, allowing you to easily follow along and understand how the model is built over time. Additionally, I am documenting the entire process through this YouTube video, where I explain each step as I implement it.

### Project Overview

We begin with nothing but an empty file and work our way to reproducing the GPT-2 (124M) model, which can be trained in around **1 hour** for approximately **$10** with a cloud GPU. If you have more patience or resources, you can also reproduce the larger GPT-3 models. This project is focused on building a simple language model, trained on internet documents, similar to GPT-2 and GPT-3.

Note: This repository does **not** cover fine-tuning or interactive chatbot development (e.g., like ChatGPT). Fine-tuning will be addressed in future updates.

### Steps Followed in the Project:

1. **Start from Scratch**: Begin with an empty file and build each component of the GPT-2 (124M) model.
2. **Reproduce GPT-2 Model**: Recreate the architecture and training loop to reproduce the GPT-2 (124M) model.
3. **Optional**: Extend the process to larger models like GPT-3 if you have the required hardware and budget.

### Model Output

After training for **10B tokens**, the model will generate outputs similar to this when prompted with "Hello, I'm a language model":

- "Hello, I'm a language model, and my goal is to make English as easy and fun as possible for everyone, and to find out the different grammar rules."
- "Hello, I'm a language model, so the next time I go, I'll just say, I like this stuff."

After **40B tokens** of training:

- "Hello, I'm a language model, a model of computer science, and it's a way (in mathematics) to program computer programs to do things like write."
- "Hello, I'm a language model, not a human. This means that I believe in my language model, as I have no experience with it yet."

### Requirements

- **GPU**: A cloud GPU is recommended for training. If you don't have access to one, I suggest using **Lambda**.
- **Time**: Around **1 hour** for GPT-2 (124M) model, longer for larger models.

### Future Updates

Fine-tuning and advanced features will be covered in future tutorials. This repository currently focuses only on training a basic language model.

### For Discussions and Questions

Feel free to use the **Discussions** tab for any questions. For faster communication, join the **Zero To Hero Discord**, and check out the **#nanoGPT** channel.
