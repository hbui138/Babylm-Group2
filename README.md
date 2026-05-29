# Babylm-Group2
Repository for Babylm group 2 Praktikum - TUM SoSe26

Hieu's part:

# Artificial Metalinguistic Awareness in Low-Resource Bilingual LMs
## BabyLM 2026 Research Project (Strict-Small Track - 10M Words)

This document compiles the core ideas, scientific rationale, data structure, and implementation roadmap for designing the data system of a small-scale (10M words) bilingual language model project.

---

## 1. Overview & Research Motivation

* **The Problem:** Current Large Language Models (LLMs) acquire multilingual capabilities through "brute-force computation" and massive data scales (billions to trillions of tokens) to blindly discover structural similarities. In contrast, children in bilingual environments only need exposure to approximately 10 million words to master both languages without language confusion.
* **Cognitive Motivation:**
    * **Telegraphic Speech:** Children learning to speak often omit complex functional words, focusing solely on the core syntactic framework (Subject - Verb - Object / S-V-O). English (morphological variation) and Vietnamese (isolating language), despite their vast differences, share this common S-V-O word order "skeleton."
    * **Metalinguistic Awareness:** Bilingual children develop the ability to decouple grammatical rule structures from the surface lexical layer at a very early stage.
* **Research Hypothesis:** Actively forcing an ultra-small language model (10M parameters) to learn the abstract syntactic "skeleton" first (via Universal POS - UPOS tags), combined with a Curriculum Learning strategy, will generate **Artificial Metalinguistic Awareness**. This approach will maximize sample efficiency within the strict 10M word limit of the BabyLM competition.

---

## 2. Data Design & Pipeline

### 2.1. Dataset
* **Repository:** `ura-hcmut/PhoMT` (A clean Parquet distribution of VinAI's PhoMT on Hugging Face).
* **Scale:** Precisely extract bilingual English (`en` column) and Vietnamese (`vi` column) structures within the Word Budget of the Strict-Small track.

### 2.2. Dual-Stream In-Context Formatting
Data is organized into two parallel streams in the `.jsonl` output file:
* `text_block`: `[English Sentence] \n\n [Vietnamese Sentence] [SPLIT]`
* `pos_block`: `[English UPOS] \n\n [Vietnamese UPOS] [SPLIT]`

*Example of the Vietnamese UPOS mapping mechanism from PhoNLP:* PhoNLP's native tags (e.g., `Nc`, `Ny`, `Np`, `L`) are automatically mapped to the international UPOS standard using a strict mapping table derived from the Universal Dependencies project documentation (e.g., `Nc -> NOUN`, `L -> DET`).

### 2.3. Curriculum Learning Score
Utilize the `spaCy` library to parse the syntax of the original English sentences, counting verbs (`VERB`) and subordinating conjunctions (`SCONJ`) to calculate a structural difficulty score:
$$\text{Complexity Score} = \text{Count}(\text{VERB}) + 2 \times \text{Count}(\text{SCONJ})$$

---

## 3. Custom Bilingual Tokenizer Architecture

To conserve the parameter budget for the 10M model, the system utilizes a **Custom BPE Tokenizer** trained from scratch on the aggregated dataset:
* **Vocab Size:** Strictly constrained to `16,000` tokens (to reduce the Embedding matrix size to ~4M parameters, leaving room for deeper Attention layers downstream).
* **Special Tokens Protection:** Hardcode the 17 UPOS tags (`NOUN`, `VERB`, `ADJ`, etc.) and the `[SPLIT]` navigation tag into the `special_tokens` list. The BPE algorithm will never fragment these tags, ensuring 100% integrity of the syntactic stream.

---

## 4. Phase 1 Training Strategy (Vanilla GPT-2 10M)

Following the supervisor's guidance, the project will initially rely on the most basic (Vanilla) architectures to isolate and clarify the data's effectiveness before intervening in the Transformer's Head splitting mechanisms. The process consists of 2 stages:

1.  **Stage 1: Syntactic Bootstrapping (10% Budget - 1M tokens):**
    * The model is only fed the `pos_block` data stream (abstract UPOS character sequences).
    * Objective: Force the Attention matrix to learn cross-lingual S-V-O word order alignment without lexical interference.
2.  **Stage 2: Lexical Grounding & Curriculum (90% Budget - 9M tokens):**
    * The model learns from the `text_block` stream (actual sentences).
    * Data is loaded via a **Curriculum Sampler** (developed by Ivan): Starting from short sentences with a low `Complexity Score` (simulating Telegraphic Speech) and gradually increasing in complexity across Epochs.

---

## 5. Evaluation Strategy

The system will run trials on **3 parallel 10M Vanilla models** to conduct A/B testing:
* **Model A (Pure English Baseline):** Trained on 10M randomly shuffled English words.
* **Model B (Random Bilingual):** Trained on 10M randomly shuffled English-Vietnamese words.
* **Model C (Bilingual Curriculum + Bootstrapping):** Applies the team's entire Data Pipeline.

### Measurement Metrics:
* **Official Stream (BabyLM Eval Harness):** Run the organizers' standard English test set to obtain **BLiMP** (unconscious syntactic awareness) and **AoA** (Age of Acquisition vocabulary curve) scores. Objective: Prove that Model C achieves higher English structural scores thanks to Vietnamese supplementation.
* **Custom Stream (In-house built):** Process through the `evaluate` library to measure `sacrebleu/chrF` translation scores and write Regex functions to export LaTeX X-bar syntax tree diagrams to visualize metalinguistic capabilities in the report.

---

## 6. Team Tasks

1.  **Hieu (Data Engineer):**
    * Manage the data loading pipeline from Hugging Face PhoMT.
    * Extract difficulty scores and package the dual-stream `.jsonl` file.
    * Train and export the `16k` Custom Tokenizer config file containing POS Special Tokens.
2.  **Ivan (DataLoader & Curriculum Loop):**
    * Teacher model
3.  **Sauhard (Architecture & Training Loop):**
    * Try to add POS and Tense to attention head