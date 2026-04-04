# :rocket: Fine-tuning LLMs from scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)

This repo holds a collection of Jupyter notebooks to fine-tune Large Language Models from scratch.

![banner](./media/banner.jpg)

> Research shows that the pattern-recognition abilities of foundation language models are so powerful that they sometimes require relatively little additional training to learn specific tasks. That additional training helps the model make better predictions on a specific task. This additional training, called fine-tuning, unlocks an LLM's practical side.

Read more about Fine-tuning process here: [View](https://developers.google.com/machine-learning/crash-course/llm/tuning).

## Contents:

- :star2: [Notebook](./fine-tune-gpt2-spam-classifier.ipynb): Fine-tune [GPT2 (Small)](https://huggingface.co/openai-community/gpt2) 125 Million parameter model for classifying spam messages.
- :sunflower: [Notebook](./fine-tune-tiny-stories.ipynb): Fine-tune [TinyStories 19M](https://huggingface.co/SauravP97/tiny-stories-19M) model to summarize stories.
- :gear: [Notebook](./fine_tune_llama_1b_summarization.ipynb): Fine-tune [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-1B) 1 Billion parameter model for summarization task using LoRA.