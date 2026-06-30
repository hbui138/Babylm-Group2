# Babylm-Group2
Repository for Babylm group 2 Praktikum - TUM SoSe26

Hieu's part:

# Injecting Explicit Syntax into BabyLM: Data-Centric Scaffolding (GPT-2)
## BabyLM 2026 Research Project (Strict-Small Track - 10M Words)

This document compiles the core ideas, scientific rationale, experimental designs, and data implementation roadmap for injecting explicit syntactic structures into a small-scale (10M words) language model using Data-Centric Curriculum Learning.

*(Note: The initial multilingual/bilingual hypothesis exploration is preserved in the Appendix at the end of this document).*

---

## 1. Overview & Research Motivation

* **The Problem:** In low-resource settings (10M tokens), Language Models often struggle to generalize syntax and capture long-distance structural dependencies. 
* **Research Hypothesis:** Can we help a small Causal Language Model (GPT-2) learn language better by explicitly teaching it the grammatical "skeleton" first? By injecting Universal Part-of-Speech (UPOS) tags and altering the training curriculum, we aim to induce structural scaffolding before standard lexical immersion.
* **Paradigm:** **Data-Centric Scaffolding.** Unlike architecture-centric approaches (which modify the loss function or add classification heads), this approach manipulates the *training data distribution and sequence* over time to guide the model's learning trajectory.

---

## 2. Data Design & The "No Space" Tokenizer Trick

To explicitly teach the model POS tags without expanding the parameter budget, we manipulated the data formatting and the tokenizer:

* **The "No Space" Trick:** When injecting POS tags (e.g., `<UPOS_NOUN>`), standard BPE tokenizers often treat the preceding space (`Ġ`) as part of the token, which can lead to fragmentation (e.g., splitting into `Ġ` and `NOUN`). 
* **Implementation:** Before training, we stripped all spaces from the POS tag sequences. This ensures the BPE algorithm treats the tag as a single, indivisible token. 
* **Result:** The model learns a singular, highly concentrated embedding vector for each grammatical concept, rather than wasting capacity on fragmented subwords.

---

## 3. Experimental Configurations & Curriculum Strategies

We experimented with several curriculum strategies to determine the optimal way to transition the model from abstract syntax (POS) to concrete vocabulary (Text).

### 3.1. Standard Baseline
* **English Baseline (Pure Text Immersion):** The model trains exclusively on standard raw English text for the entire duration (10 epochs) with regular data shuffling. This serves as the control group.

### 3.2. Sequential Scaffolding
* **10 POS + 10 Text:** A sharp, two-phase curriculum learning approach. It forces the model to build a grammatical foundation by training entirely on POS tags for 10 epochs. Afterward, it switches abruptly to 100% standard text for the remaining 10 epochs.
* **English with POS (Two-Stage Scaffolding / 1 POS + 9 Text):** A brief syntactic warm-up. The model looks at POS tags for just the 1st epoch to initialize attention heads toward grammatical structures, followed by 9 epochs of pure text.

### 3.3. Continuous Transitions (Mitigating Catastrophic Forgetting)
Switching abruptly from POS to Text causes massive distribution shifts. We tested transitionary approaches to "wean" the model off tags:
* **Fading POS (Gradual Curriculum):** The proportion of POS-tagged data gradually decreases across 10 virtual epochs (starting at 100% POS and dropping by ~11% each step).
* **Linear Decay:** The percentage of POS-tagged data is reduced by a fixed amount (10%) per epoch until it hits a 10% floor.
* **Exponential Decay:** Drops POS exposure abruptly in early epochs (100% -> 60% -> 35% -> 20%), then plateaus at a 10% floor.

### 3.4. Structural Modifications
* **Prefixing:** The sequence of POS tags is directly attached right before the corresponding text sentence (e.g., `<POS_Start> Noun Verb... <Text_Start> The cat jumps...`). This primes the model with the grammatical framework as context before generating the vocabulary.
* **Auxiliary Loss:** Running a parallel predictive head to guess POS tags without altering the main text input, evaluated with both fixed and dynamic (decaying) $\alpha$ weights.

---

## 4. Evaluation Results 

### 4.1. Results (block_size = 32, babylm dataset, no POS space)
*Tokenizer from the base model, special tokens added without spaces.*

| Task | English Baseline | 10POS + 10Text (UPOS) | 10POS + 10Text (EngPOS) | Fading POS (UPOS) | English with POS (UPOS) | English with POS (EngPOS) |
|---|---|---|---|---|---|---|
| blimp_filtered | 70.61 | 70.29 | 70.16 | 67.48 | 70.70 | 69.73 |
| - filler_gap_dependency | 71.42 | 71.29 | 70.25 | 71.21 | 72.08 | 70.11 |
| - subject_verb_agreement | 73.95 | 72.52 | 73.35 | 63.39 | 73.35 | 72.02 |
| - npi_licensing | 64.11 | 62.90 | 61.59 | 54.90 | 63.41 | 60.92 |
| - control_raising | 65.34 | 66.08 | 66.42 | 63.44 | 67.45 | 66.83 |
| - s-selection | 77.28 | 76.07 | 73.93 | 73.43 | 75.91 | 76.73 |
| - ellipsis | 69.14 | 67.91 | 65.21 | 59.20 | 66.13 | 62.70 |
| - determiner_noun_agreement | 87.69 | 86.75 | 87.28 | 85.46 | 87.25 | 86.68 |
| - anaphor_agreement | 90.64 | 88.43 | 90.80 | 86.44 | 87.43 | 91.85 |
| - binding | 71.78 | 73.84 | 72.53 | 69.55 | 71.92 | 70.71 |
| - argument_structure | 72.23 | 71.16 | 70.73 | 67.82 | 70.85 | 71.69 |
| - island_effects | 48.11 | 50.32 | 53.60 | 49.66 | 52.00 | 49.34 |
| - quantifiers | 67.95 | 64.82 | 60.48 | 70.52 | 68.31 | 69.23 |
| - irregular_forms | 85.92 | 83.39 | 85.97 | 89.70 | 81.19 | 81.19 |
| supplement_filtered | 55.77 | 58.10 | 56.92 | 55.36 | 55.07 | 56.44 |
| comps | 51.50 | 51.62 | 51.00 | 50.07 | 51.23 | 51.47 |
| entity_tracking | 13.83 | 18.95 | 21.17 | 17.03 | 27.45 | 32.20 |
| ewok_filtered | 50.40 | 50.42 | 50.85 | 49.66 | 50.73 | 50.50 |
| reading (Eye Tracking) | 9.56 | 9.46 | 8.82 | 9.44 | 10.36 | 9.83 |
| reading (Self-Paced) | 3.10 | 3.26 | 2.60 | 2.94 | 3.12 | 3.11 |
| **Zeroshot Avg** | **36.40** | **37.44** | **37.36** | **36.00** | **38.38** | **39.04** |

### 4.2. Results (block_size = 32, babylm dataset, with POS space)
*Testing various decay and prefixing strategies.*

| Task | English Baseline | English with POS | Fading POS | Aux Loss - 1.0 | Aux Loss - dyn alpha | Prefixing | Linear POS decay | Exp POS decay | Chunking |
|---|---|---|---|---|---|---|---|---|---|
| blimp_filtered | **70.61** | 68.32 | 66.57 | 69.22 | 69.39 | - | 66.15 | 68.63 | 68.25 |
| - filler_gap_dependency | **71.42** | 71.34 | 69.56 | 67.24 | 69.87 | - | 69.77 | 71.13 | 69.78 |
| - subject_verb_agreement | **73.95** | 66.47 | 63.74 | 71.39 | 69.23 | - | 62.89 | 69.85 | 69.69 |
| - npi_licensing | **64.11** | 57.05 | 53.91 | 62.75 | 60.07 | - | 51.06 | 52.91 | 56.86 |
| - control_raising | 65.34 | 62.94 | 62.34 | 65.46 | 64.68 | - | 63.33 | 63.65 | **65.94** |
| - s-selection | **77.28** | 72.99 | 71.73 | 74.81 | 74.59 | - | 72.33 | 75.14 | 76.18 |
| - ellipsis | **69.14** | 60.06 | 56.69 | 62.21 | 65.03 | - | 54.17 | 62.82 | 63.74 |
| - determiner_noun_agreement | 87.69 | 86.62 | 85.23 | 86.50 | 86.47 | - | 84.01 | 86.35 | **87.83** |
| - anaphor_agreement | 90.64 | 84.65 | 86.75 | **90.80** | 89.12 | - | 83.49 | 84.65 | 87.85 |
| - binding | **71.78** | 69.55 | 68.13 | 70.45 | 70.92 | - | 68.39 | 69.89 | 70.05 |
| - argument_structure | **72.23** | 68.00 | 67.09 | 70.41 | 70.87 | - | 66.37 | 68.87 | 69.45 |
| - island_effects | 48.11 | 49.44 | 47.72 | 46.39 | 49.91 | - | 50.55 | **52.69** | 47.66 |
| - quantifiers | 67.95 | **73.94** | 69.05 | 72.75 | 72.62 | - | 68.05 | 71.86 | 60.77 |
| - irregular_forms | 85.92 | 90.33 | **92.91** | 87.65 | 83.08 | - | 90.49 | 87.60 | 88.02 |

### 4.3. Results (block_size = 256, babylm dataset, with POS space)

| Task | English Baseline | English with POS | Fading POS | Aux Loss | Prefixing | Linear decay | Exp decay |
|---|---|---|---|---|---|---|---|
| blimp_filtered | 68.57 | 66.83 | 63.52 | 63.85 | 62.35 | 57.23 | 59.13 |
| - filler_gap_dependency | 69.01 | 66.98 | 69.07 | 66.05 | 63.36 | 65.22 | 64.27 |
| - subject_verb_agreement | 67.30 | 63.82 | 57.13 | 58.27 | 58.21 | 52.29 | 54.53 |
| - npi_licensing | 62.77 | 56.94 | 45.01 | 55.61 | 36.50 | 40.87 | 40.63 |
| - control_raising | 63.49 | 64.11 | 62.87 | 62.20 | 62.23 | 59.87 | 60.37 |
| - s-selection | 74.42 | 73.10 | 71.40 | 72.06 | 75.91 | 66.23 | 68.10 |
| - ellipsis | 64.17 | 61.96 | 54.23 | 51.29 | 48.96 | 34.29 | 35.95 |
| - determiner_noun_agreement | 85.68 | 86.10 | 83.26 | 78.18 | 80.57 | 61.50 | 71.85 |
| - anaphor_agreement | 90.27 | 90.96 | 85.02 | 85.07 | 84.17 | 72.87 | 75.71 |
| - binding | 67.96 | 67.28 | 64.98 | 65.04 | 64.92 | 64.21 | 65.02 |
| - argument_structure | 67.97 | 67.64 | 65.23 | 65.45 | 63.49 | 60.08 | 61.17 |
| - island_effects | 47.10 | 46.28 | 45.65 | 44.40 | 47.64 | 46.63 | 41.69 |
| - quantifiers | 76.27 | 71.23 | 63.16 | 72.23 | 80.45 | 54.49 | 63.87 |
| - irregular_forms | 86.28 | 81.45 | 91.28 | 81.92 | 76.14 | 84.24 | 93.22 |

---

## 5. Discussion & Deep Dive Analysis

The data reveals a compelling **Trade-off Phenomenon** when injecting explicit syntax. While the overall BLiMP average of the POS-scaffolded models rarely beats the pure English baseline, the sub-task breakdown highlights a systematic skill shift:

### 5.1. The "Island Effects" & "Entity Tracking" Boost
The POS scaffolding successfully taught the model to recognize strict phrase boundaries. For example, in the `10POS + 10Text` experiment, **Island Effects** surged to **53.60%** (up from the 48.11% baseline), and **Entity Tracking** jumped to **21.17%** (up from 13.83%). 
* *Rationale:* Island Effects test the model's grasp of long-distance structural dependencies. By forcing the model to learn the syntactic "skeleton" first, it became highly attuned to structural constraints and phrase boundaries, decoupling grammar from specific vocabulary.

### 5.2. Lexical Starvation (The "Irregular Forms" Drop)
Conversely, forcing the model to learn POS tags incurs a massive cost to morphology. In the `10POS + 10Text` model, **Irregular Forms** dropped to **83.39%** (down from 85.92%).
* *Rationale:* Irregular forms (e.g., *go -> went*) rely heavily on surface-level lexical memorization. By replacing words with tags (`VBD`) for half the training duration, the model suffered from *Lexical Starvation*. It spent compute cycles looking at abstract concepts rather than the actual morphology of words.

### 5.3. Catastrophic Forgetting
The `10POS + 10Text` strategy creates a severe **Distribution Shift** at epoch 10. The loss curve exhibits a massive spike when transitioning from the small, easily predictable POS vocabulary back to the full English vocabulary. Despite this shock, the model recovers quickly to map vocabulary to the learned structures, confirming that Curriculum Learning can impart structural priors, even if self-supervised learning (the baseline) is highly efficient at doing this implicitly.

### 5.4. Conclusion
While injecting explicit syntactic tags improves awareness of complex hierarchical structures, it causes interference with surface-level morphological learning. Ultimately, the self-supervised baseline remains incredibly robust, suggesting that language models are highly efficient at encoding implicit syntax purely from natural text immersion.

---
---

## APPENDIX: Artificial Metalinguistic Awareness in Low-Resource Bilingual LMs (Extra - Not planning on poster)
*(Initial explorations into cross-lingual structural transfer using Vietnamese and English).*

### 1. Bilingual Research Motivation
* **Cognitive Motivation:** Children learning to speak often omit complex functional words, focusing solely on the core syntactic framework (Subject - Verb - Object / S-V-O). English (morphological variation) and Vietnamese (isolating language), despite their vast differences, share this common S-V-O word order "skeleton."
* **Research Hypothesis:** Actively forcing an ultra-small language model (10M parameters) to learn the abstract syntactic "skeleton" first (via Universal POS - UPOS tags) will generate Artificial Metalinguistic Awareness.

### 2. Bilingual Data Pipeline
* **Dataset:** `ura-hcmut/PhoMT` (Bilingual English and Vietnamese).
* **Dual-Stream Formatting:**
    * `text_block`: `[English Sentence] \n\n [Vietnamese Sentence] [SPLIT]`
    * `pos_block`: `[English UPOS] \n\n [Vietnamese UPOS] [SPLIT]`
* **Curriculum Learning Score:** We utilized `spaCy` (for English) and `PhoNLP` (for Vietnamese) to parse syntax and assign structural difficulty scores based on length and complex syntax markers (`SCONJ`, `CCONJ`, `PUNCT`, `PART`). Sampled at 60% Easy, 30% Medium, 10% Hard.

### 3. Bilingual Evaluation Results (Zero-Shot)

block_size = 256, phomt dataset

| Task | English Baseline | Bilingual VI-EN | Bilingual Random VI-EN | English with pos |
|---|---|---|---|---|
| blimp_filtered | 69.57 | 69.67 | 70.30 | 67.31 |
| - filler_gap_dependency | 71.16 | 70.63 | 70.63 | 68.52 |
| - subject_verb_agreement | 70.08 | 69.73 | 68.36 | 68.79 |
| - npi_licensing | 60.87 | 56.83 | 62.08 | 59.67 |
| - control_raising | 66.38 | 66.26 | 66.47 | 62.39 |
| - s-selection | 76.35 | 73.27 | 76.24 | 73.71 |
| - ellipsis | 59.33 | 65.21 | 69.39 | 55.34 |
| - determiner_noun_agreement | 90.02 | 87.63 | 87.56 | 89.02 |
| - anaphor_agreement | 84.02 | 81.55 | 82.23 | 73.92 |
| - binding | 70.72 | 71.63 | 70.17 | 68.57 |
| - argument_structure | 70.42 | 69.55 | 70.37 | 68.21 |
| - island_effects | 43.49 | 48.46 | 50.71 | 41.69 |
| - quantifiers | 75.20 | 81.06 | 82.00 | 71.73 |
| - irregular_forms | 91.49 | 91.22 | 80.66 | 92.85 |
| supplement_filtered | 56.50 | 54.98 | 54.71 | - |
| comps | 50.32 | 51.34 | 51.48 | - |
| entity_tracking | 38.33 | 12.41 | 17.47 | - |
| ewok_filtered | 51.07 | 53.53 | 52.26 | - |

### 4. Bilingual Finetune Results (Accuracy %)

phomt dataset

| Task | English Baseline | Bilingual | Bilingual Random |
|---|---|---|---|
| mnli | 49.53 | 51.77 | 51.57|
| wsc | 63.46 | 61.54 | 61.54 | 
| mrpc | 68.14 | 72.55 | 74.02 |
| multirc | 64.44 | 61.06 | 61.47 | 
| boolq | 67.03 | 67.65 | 68.56 |
| rte | 54.68 | 61.15 | 55.40 |
| qqp | 72.45 | 72.59 | 73.66 |

### 5. Synthesis: The Unstable Nature of Random Mixing
The most glaring vulnerability of unstructured bilingual training is exposed in the `irregular_forms` task. The Bilingual Random model crashed completely to 80.66% (down from 91.49%). When forced to process English and Vietnamese simultaneously without guidance, the model suffered severe interference between English's complex morphology and Vietnamese's isolating nature.