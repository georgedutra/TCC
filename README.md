# TCC

Repository for the Undergraduate Thesis of student George Dutra, registration number 221708043, in the Data Science and Artificial Intelligence program at FGV-EMAp (Fundação Getulio Vargas - School of Applied Mathematics).

This thesis was implemented using Python 3.10.12, and all necessary dependencies will be compiled in the [requirements](./requirements.txt) file.

## Summary

        Since the publication of the groundbreaking paper “Attention is All You Need” (Vaswani et al., 2017), transformer-based architectures have revolutionized machine learning, enabling rapid advancements in Large Language Models (LLM's).
        However, state-of-the-art LLM's face critical challenges: (1) limited accessibility due to proprietary licensing, expensive APIs, and high computational costs for training/inference; (2) resource inefficiency, as even open-source models require specialized hardware (e.g., GPUs with high VRAM) for deployment. These barriers hinder adoption in resource-constrained environments, such as academic research and small-scale applications.

## Embedding

For the retrieval-augmented generation (RAG) experiments, this study utilized the Dicionário Histórico-Biográfico Brasileiro (DHBB). It's official repository can be found in [GitHub](https://github.com/cpdoc/dhbb) and must be cloned in order to realize it's vectorial embedding. Both the original files location and the index target folder must be defined in [dhbb_embedding.ipynb](./src/embedding/dhbb_embedding.ipynb). This notebook can be used to generate the FAISS index, and it only needs to be executed once.

## Baseline Model

For most experiments, Mistral 7B will be used as the baseline model, as it is a reliable, well-known, and widely adopted LLM. In parallel studies, I conducted tests comparing the response time and reliability of five different models. Although Mistral 7B exhibits relatively high latency, its accuracy compensates for this limitation. For quantization tests, the Mistral 7B text quantized 4-bits and 8-bits variants will be used, both found in Ollama website.

