<h1 align="center">🛠️🤖 From Exploration to Mastery:
<br>
Enabling LLMs to Master Tools via Self-Driven Interactions

</h1>
<p align="center">
  <a href="#-quick-start"><b>[Quick Start]</b></a> •
  <a href="https://arxiv.org/pdf/2410.08197"><b>[Paper]</b></a> •
  <a href="#%EF%B8%8F-citation"><b>[Citation]</b></a>
</p>

Repo for the paper "[From Exploration to Mastery:Enabling LLMs to Master Tools via Self-Driven Interactions](https://arxiv.org/abs/2410.08197)" [ICLR'25 Oral]

## 🔥 News

- [2025/2/24] We release all the code for DRAFT.
- [2025/2/11] DRAFT is selected to be presented as an **Oral (1.8%)**.
- [2025/1/23]  DRAFT is accepted by [**ICLR 2025**](https://iclr.cc/).
- [2024/10/10] Our [**paper**](https://arxiv.org/abs/2410.08197) and code is released.

## 💡 Introduction

Due to the inherent understanding gap between LLMs and humans, inefficiencies and inaccuracies within existing tool documentation hamper the effective utilization of tools by LLMs. Humans, acquire tool proficiency through repeated interactions and hands-on experiences, capable of maintaining an updated comprehension of these tools despite their evolving functionalities. In light of this, we propose DRAFT, conceptualized to automate the adjustment and optimization of tool documentation based on the feedback derived from the LLM's interaction with the tool.

<p align="center">
    <img src="./images/introduction.png" width="1000">
</p>

DRAFT is designed to dynamically adjust and optimize tool documentation based on the interaction feedback between LLMs and external tools, which significantly bridges the gap between them by enabling the LLMs to better comprehend and utilize the tools at their disposal, thereby enhancing the overall tool-using capabilities of LLMs.

<p align="center">
    <img src="./images/framework.png" width="1000">
</p>

## 🛠️ Setup

### Environment Setup


Our experimental environment is shown below:

```
openai version: 0.28.0
numpy version: 1.26.4
pandas version: 2.2.2
torch version: 2.3.1
```

### API Key Setup

Get OpenAI key from [OpenAI](https://platform.openai.com/playground/chat), RapidAPI key from [RapidAPI](https://rapidapi.com/hub) or ToolBench key from [ToolBench](https://github.com/OpenBMB/ToolBench) repo, TMDB key from [TMDB](https://developer.themoviedb.org/reference/intro/getting-started), and Spotify key from [Spotify](https://developer.spotify.com/documentation/web-api).

### Data Setup

You can download ToolBench dataset from the [Google Drive](https://drive.google.com/file/d/1M06p-OO1YM80MNhIbLYw2FtRB5Qmh39z/view) or [Tsinghua cloud](https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/) and RestBench dataset from [RestBench](https://github.com/Yifan-Song793/RestGPT) repo, then extract all tool documentation. Alternatively, you can directly use our preprocessed tool documentation.

## 🚀 Quick Start

### DRAFT

Run `DRAFT` to get revised tool documentation:

> python DRAFT.py

### Inference

Run `Inference_DFSDT` to perform inference using the tool documentation modified by DRAFT to examine the effectiveness of DRAFT.

> python Inference\_DFSDT -model\_name gpt-4o-2024-08-06 -data\_type G3 -method DRAFT

You can specify the model, dataset, and method by cmd line arguments.
### Evaluation

Run `Cal_path_rate` to calculate the path rate for evaluating the results.

> python Cal_path_rate.py

We use the official code provided by ToolBench to calculate the win rate. You can find the calculation method in the [ToolEval](https://github.com/OpenBMB/ToolBench/blob/master/toolbench/tooleval/README.md) repo.

## 📘 How the paper's core ideas are implemented in code

This repository implements the main methodology of the paper *From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions* through an iterative, feedback-driven loop in `DRAFT.py`.

### 1) Three-phase trial-and-error loop (paper → code)

For each API, DRAFT repeats the following phases:

1. **Experience Gathering (Explorer)**
   - Prompt file: `prompts/Explorer.txt`
   - Code path: `process_api_info(...)`
   - The model generates a user query + parameters, then calls the real API via `get_rapidapi_response(...)`.

2. **Learning from Experience (Analyzer)**
   - Prompt file: `prompts/Analyzer.txt`
   - Code path: `process_api_info(...)`
   - The model analyzes `(tool description, generated query/params, API response)` and outputs concrete suggestions for documentation improvements.

3. **Documentation Rewriting (Rewriter)**
   - Prompt file: `prompts/Rewriter.txt`
   - Code path: `process_api_info(...)`
   - The model rewrites the API description and also outputs "suggestions for exploring" used to guide the next exploration step.

This forms the self-driven interaction cycle described in the paper: exploration produces evidence, analysis extracts lessons, rewriting updates documentation, and the updated docs feed into the next round.

### 2) Diversity-promoting exploration strategy

To avoid repetitive trials, DRAFT embeds each generated query and compares cosine similarity with previously explored queries. If a new query is too similar, it asks Explorer to regenerate.  
This enforces broader coverage of tool behaviors rather than overfitting to a narrow query pattern.

### 3) Tool-adaptive termination mechanism

DRAFT computes a convergence score between consecutive rewritten descriptions using:
- embedding cosine similarity (semantic closeness), and
- BLEU score (surface-level lexical overlap).

If the averaged score is high enough, the loop stops early for that API.  
This corresponds to the paper's efficiency-oriented stopping rule that prevents unnecessary rewriting when edits have stabilized.

### 4) Hierarchical rewriting: API-level then tool-level

- **API-level refinement**: each API description in `tool_guidelines` is improved iteratively.
- **Tool-level consolidation**: after API updates, `prompts/rewrite_tool_doc.txt` is used to regenerate the overall `tool_description`.

This two-level rewrite keeps local API details accurate while maintaining a coherent high-level tool summary.

### 5) End-to-end execution order in `DRAFT.py`

1. Load Explorer/Analyzer/Rewriter prompts.
2. Load initial tool documentation JSON.
3. For each tool (asynchronously):
   - for each API: run iterative exploration → analysis → rewrite loop,
   - then rewrite whole-tool description.
4. Save successful/failed outputs and retry queue:
   - `DRAFT_success.json`
   - `DRAFT_failed.json`
   - `DRAFT_retry_queue.json`


## ☕️ Citation
If you find our code or work useful for your research, please cite our work.
```
@inproceedings{
  quexploration,
  title={From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions},
  author={Qu, Changle and Dai, Sunhao and Wei, Xiaochi and Cai, Hengyi and Wang, Shuaiqiang and Yin, Dawei and Xu, Jun and Wen, Ji-Rong},
  booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=QKBu1BOAwd}
}
```
