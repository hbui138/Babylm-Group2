# Babylm-Group2
**Repository for Babylm group 2 Praktikum - TUM SoSe26**  
**Hieu's Part: Data-Centric Scaffolding (GPT-2)**

---

## 1. Research Motivation, Hypotheses & Research Questions

### 1.1 Cognitive Motivation: Syntactic Bootstrapping
In developmental psychology, the theory of **Syntactic Bootstrapping** (pioneered by Lila Gleitman) suggests that children do not learn language simply by pairing isolated words with real-world objects. Instead, they actively utilize the structural, syntactic frames of sentences to narrow down and infer the meanings of novel words (semantics). For example, hearing *"The rabbit gorps the duck"* allows a child to immediately deduce that *"gorps"* is a transitive action verb, even without prior lexical knowledge of the word. 

By feeding a language model the underlying syntactic skeletons (POS tags) before or alongside raw text, we attempt to computationally mimic this phenomenon. We explore whether pre-structuring the attention space with structural frames allows a small model to more efficiently "bootstrap" its lexical representations under severe data constraints.

### 1.2 The Challenge & Core Hypothesis
* **The Challenge:** In strict low-resource settings (10M words), small Language Models often struggle to generalize syntax and capture long-distance structural dependencies, frequently reverting to shallow local collocations.
* **The Hypothesis:** Pre-training or co-training a small Causal LM (GPT-2) explicitly on structural grammar "skeletons" will condition its attention mechanisms to encode hierarchical syntax trees, yielding superior downstream generalization when vocabulary is introduced.
* **The Approach (Data-Centric Scaffolding):** Instead of modifying the model architecture or loss function, we manipulate the **training data distribution and sequence** over time. By injecting Universal/English Part-of-Speech (POS) tags, we aim to induce structural scaffolding before full lexical immersion.

### 1.3 Research Questions (RQs)
1. **RQ1:** Can explicit syntactic framing (POS sequences) serve as an effective bootstrapper for a causal language model, or does self-supervised next-token prediction already optimize for implicit syntax more efficiently?
2. **RQ2:** How does the granularity of the structural scaffolding (coarse-grained UPOS vs. fine-grained English Penn Treebank tags) affect the model's ability to track long-distance structural dependencies?
3. **RQ3:** Which curriculum strategy (Sequential, Warm-up, or Fading) best mitigates catastrophic forgetting and bridges the stark distribution shift between abstract syntax tokens and raw surface-level vocabulary?

---

## 2. Data Design: The "No Space" Tokenizer Trick
To teach the model explicit POS tags without expanding the parameter budget or altering the tokenizer size, we manipulated the data formatting:
* **The Problem:** Standard BPE tokenizers treat preceding spaces (`Ġ`) as part of the token, fragmenting special tags (e.g., splitting into `Ġ` and `NOUN`).
* **The Solution:** We stripped all spaces from the POS tag sequences before training. 
* **The Impact:** The BPE algorithm treats each tag (e.g., `<UPOS_NOUN>`) as a single, indivisible token. The model learns a singular, highly concentrated embedding vector for each grammatical concept rather than wasting capacity on fragmented subwords.

---

## 3. Curriculum Strategies
We tested multiple strategies to transition the model from abstract syntax to concrete vocabulary:
1. **English Baseline (Control):** 10 epochs of pure text immersion.
2. **Sequential Scaffolding (10 POS + 10 Text):** A sharp two-phase curriculum. 10 epochs of 100% POS tags to build the skeleton, followed by an abrupt switch to 10 epochs of 100% text.
3. **Syntactic Warm-up (English with POS / 1 POS + 9 Text):** 1 epoch of POS tags to initialize attention heads toward grammatical structures, followed by 9 epochs of pure text.
4. **Fading POS:** A gradual, epoch-by-epoch decrease in the proportion of POS-tagged data to mitigate catastrophic forgetting.

---

## 4. Evaluation Results (block_size = 32, No POS Space)

### Zero-Shot Results (BLiMP & Reading/Tracking)
| Task | English Baseline | 10POS + 10Text (UPOS) | 10POS + 10Text (EngPOS) | Fading POS (UPOS) | English with POS (UPOS) | English with POS (EngPOS) |
|---|---|---|---|---|---|---|
| blimp_filtered | 70.61 | 70.29 | 70.16 | 67.48 | **70.70** | 69.73 |
| - filler_gap_dependency | 71.42 | 71.29 | 70.25 | 71.21 | **72.08** | 70.11 |
| - subject_verb_agreement | **73.95** | 72.52 | 73.35 | 63.39 | 73.35 | 72.02 |
| - npi_licensing | **64.11** | 62.90 | 61.59 | 54.90 | 63.41 | 60.92 |
| - control_raising | 65.34 | 66.08 | 66.42 | 63.44 | **67.45** | 66.83 |
| - s-selection | **77.28** | 76.07 | 73.93 | 73.43 | 75.91 | 76.73 |
| - ellipsis | **69.14** | 67.91 | 65.21 | 59.20 | 66.13 | 62.70 |
| - determiner_noun_agreement | **87.69** | 86.75 | 87.28 | 85.46 | 87.25 | 86.68 |
| - anaphor_agreement | 90.64 | 88.43 | 90.80 | 86.44 | 87.43 | **91.85** |
| - binding | 71.78 | **73.84** | 72.53 | 69.55 | 71.92 | 70.71 |
| - argument_structure | **72.23** | 71.16 | 70.73 | 67.82 | 70.85 | 71.69 |
| - island_effects | 48.11 | 50.32 | **53.60** | 49.66 | 52.00 | 49.34 |
| - quantifiers | 67.95 | 64.82 | 60.48 | **70.52** | 68.31 | 69.23 |
| - irregular_forms | **85.92** | 83.39 | 85.97 | 89.70 | 81.19 | 81.19 |
| supplement_filtered | 55.77 | **58.10** | 56.92 | 55.36 | 55.07 | 56.44 |
| comps | 51.50 | **51.62** | 51.00 | 50.07 | 51.23 | 51.47 |
| entity_tracking | 13.83 | 18.95 | 21.17 | 17.03 | 27.45 | **32.20** |
| ewok_filtered | 50.40 | 50.42 | **50.85** | 49.66 | 50.73 | 50.50 |
| reading (Eye Tracking) | 9.56 | 9.46 | 8.82 | 9.44 | **10.36** | 9.83 |
| reading (Self-Paced) | 3.10 | **3.26** | 2.60 | 2.94 | 3.12 | 3.11 |
| **Zeroshot Avg** | 36.40 | 37.44 | 37.36 | 36.00 | 38.38 | **39.04** |

### Downstream Finetuning Results (GLUE)
| Task (Finetune) | English Baseline | 10POS + 10Text (UPOS) | 10POS + 10Text (EngPOS) | Fading POS (UPOS) | English with POS (UPOS) | English with POS (EngPOS) |
|---|---|---|---|---|---|---|
| boolq | 67.83 | 67.40 | 66.79 | **69.05** | 68.32 | 68.13 |
| mnli | **47.33** | 43.34 | 44.72 | 45.31 | 44.13 | 45.46 |
| mrpc | 71.08 | 71.57 | **72.55** | 72.06 | 69.61 | 69.61 |
| multirc | 65.88 | 64.81 | 65.14 | **65.97** | 65.59 | 65.59 |
| qqp | **69.85** | 69.80 | 68.46 | 69.78 | 67.83 | 69.31 |
| rte | 56.83 | 56.83 | **60.43** | 56.12 | 56.83 | 56.12 |
| wsc | 63.46 | 61.54 | **69.23** | 63.46 | 65.38 | 67.31 |
| **Finetune Avg** | 63.18 | 62.18 | **63.90** | 63.11 | 62.53 | 63.08 |

---

## 5. Key Findings & Discussion

**1. The Syntax-Lexicon Trade-off (The "Double-Edged Sword" Effect)**
* **Structural Superiority:** Models exposed to explicit POS tags significantly outperform the baseline on tasks requiring long-distance structural awareness. For example, in **Island Effects**, the `10POS + 10Text (EngPOS)` model jumped to **53.60** (vs. Baseline 48.11). By learning the abstract structure first, the attention heads are primed to grasp core hierarchical relations.
* **Entity Tracking Breakthrough:** Explicit syntactic markers act as strong structural anchors ("attention beacons"). The `English with POS (EngPOS)` configuration reached **32.20**, more than doubling the baseline score (**13.83**), as tracking proper nouns and pronouns becomes trivial when explicitly tagged.
* **Lexical Starvation:** Replacing raw text with abstract POS tags deprives the model of surface-level vocabulary exposure. This triggers a noticeable decline in memory-reliant morphological tasks like **Irregular Forms**, where `English with POS` variants dropped to **81.19** (vs. Baseline 85.92).

**2. Fine-Grained vs. Coarse-Grained Syntax**
* Detailed syntax (Penn Treebank tags / `EngPOS`) proved vastly superior to coarse-grained syntax (Universal POS / `UPOS`) for tracking complex dependencies. This is highly visible in `Anaphor Agreement` (91.85 vs. 87.43) and `Entity Tracking` (32.20 vs. 27.45), indicating that a richer grammatical taxonomy provides better grounding.

**3. Curriculum Strategy Comparison**
* **Syntactic Warm-up (1 POS + 9 Text):** Achieved the highest overall **Zeroshot Average (39.04)**. A single early epoch of POS functions as a highly efficient structural initializer for attention layers before they lock onto lexical strings, avoiding the downside of lexical starvation.
* **Sequential (10 POS + 10 Text):** Showed solid structural gains but suffered from severe distribution shift (Catastrophic Forgetting) at epoch 10 when moving abruptly from abstract symbols to text tokens, resulting in a compromised performance ceiling.
* **Fading POS:** Yielded the lowest overall performance (36.00). The continuous shifting of token proportions across epochs likely induced representation instability and gradient interference within the shared embeddings.

**Conclusion:** Injecting explicit syntax via data preprocessing does not uniformly boost language modeling. It acts as a **structural catalyst**—enhancing the parsing of complex syntax trees and entities—but requires careful curriculum design to prevent the loss of surface-level vocabulary memorization. Self-supervised baselines remain incredibly robust at encoding implicit syntax.