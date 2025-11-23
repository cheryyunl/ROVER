<div align="center">
<h2>ROVER-TG: Benchmarking Interleaved Text-Generation Reasoning</h2>

[Yongyuan Liang](https://cheryyunl.github.io/)&nbsp;
[Wei Chow](https://scholar.google.com/citations?user=br7-IGkAAAAJ)&nbsp;
[Feng Li](https://fengli-ust.github.io/)&nbsp;
[Ziqiao Ma](https://mars-tin.github.io/)&nbsp;
[Xiyao Wang](https://si0wang.github.io/)&nbsp;
[Jiageng Mao](https://pointscoder.github.io/)&nbsp;
[Jiuhai Chen](https://jiuhaichen.github.io/)&nbsp;

[Jiatao Gu](https://jiataogu.me/)&nbsp;
[Yue Wang](https://yuewang.xyz/)&nbsp;
[Furong Huang](https://furong-huang.com/)


<h4>
<a href="https://arxiv.org/abs/2511.01163">📄 arXiv Paper</a> &nbsp; 
<a href="https://huggingface.co/datasets/cheryyunl/ROVER">🤗 Hugging Face Dataset</a>
</h4>

</div>

## :bookmark_tabs: Overview
ROVER-TG (Text & Generation) focuses on evaluating models that can **interleave reasoning** by generating both text and visual aids. Unlike traditional benchmarks that only look at final text answers, ROVER-TG assesses:
1.  **Visual Reasoning Quality**: Is the generated visual aid (e.g., geometric construction, physics simulation, robot trajectory) correct?
2.  **Reasoning Alignment**: Is the final text answer grounded in the generated visual aid?

## Quick Start

### 1. Setup

Install neccessary repo:
```bash
pip3 install -r requirements.txt
```

Configure your OpenAI credentials (choose one):

**Option A: Environment Variables (Recommended)**
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4o"
```

**Option B: Edit config.py**
```python
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL = "gpt-4o"
```

### 2. Configure Data Path

Set the generation directory path (choose one):

**Option A: Environment Variable (Recommended)**
```bash
export VORTEX_GEN_DIR="/path/to/your/generated/results"
export MAX_RETRIES="3"  # Optional: number of retries for failed evaluations
```

**Option B: Edit config.py**
```python
VORTEX_GEN_DIR = "/path/to/your/generated/results"
MAX_RETRIES = 3  # Number of retries for failed evaluations
```

### 3. Generated Files Format

Your generation directory should contain files in this format:
```
your_gen_dir/
├── gen_{task_id}.png          # Generated reasoning image (required)
├── gen_{task_id}.txt          # Model's reasoning text & answer (required for alignment)
└── ...
```

**File Naming Convention:**
- Images: `gen_{task_id}.png`
- Text: `gen_{task_id}.txt`
- Task IDs match the ID field in the Hugging Face dataset.

### 4. Run Evaluation

```bash
# Evaluate all available results
python evaluate_rover.py --output_dir results

# Filter by problem type
python evaluate_rover.py --output_dir results --problem_type physical
python evaluate_rover.py --output_dir results --problem_type logical
python evaluate_rover.py --output_dir results --problem_type embodied
python evaluate_rover.py --output_dir results --problem_type jigsaw
python evaluate_rover.py --output_dir results --problem_type multi-view

# Custom worker count
python evaluate_rover.py --output_dir results --workers 5
```

### 5. View Results

The results are saved in `results/rover_metrics.jsonl`. Each line contains the detailed evaluation for one task.

## Evaluation Metrics

The system evaluates 2 core metrics:

| Metric | Code | Description | Inputs Used |
|--------|------|-------------|-------------|
| **Interleaved Reasoning** | IR | Quality of the generated visual aid compared to Ground Truth. | Generated Image, GT Reasoning Image |
| **Reasoning Alignment** | RA | Consistency between text answer and generated image, plus answer correctness. | Generated Image, Text Answer, GT Answer |

### Problem Types & Evaluation Logic

*   **Physical**: Simulates physics outcomes.
    *   *IR Check*: Does the generated image show a physically plausible result matching GT?
*   **Embodied**: Plans robot trajectories.
    *   *IR Check*: Does the generated trajectory (waypoints) match the GT path and avoid obstacles?
*   **Logical**: Solves geometry/math problems.
    *   *IR Check*: Does the generated image contain correct auxiliary lines/constructions matching GT?
*   **Jigsaw (Perception)**: Completes images.
    *   *IR Check*: Is the missing area filled coherently matching GT?
*   **Multi-view**: Synthesizes views.
    *   *IR Check*: Does the generated wider view correctly combine information from two input views?

## Data Source

The evaluation uses the `cheryyunl/ROVER` dataset (subset `ROVER-TG`) from Hugging Face, which is automatically downloaded.

## Citation

If you use this benchmark in your research, please consider citing:

```bibtex
@article{liang2025rover,
  title={ROVER: Benchmarking Reciprocal Cross-Modal Reasoning for Omnimodal Generation},
  author={Liang, Yongyuan and Chow, Wei and Li, Feng and Ma, Ziqiao and Wang, Xiyao and Mao, Jiageng and Chen, Jiuhai and Gu, Jiatao and Wang, Yue and Huang, Furong},
  journal={arXiv preprint arXiv:2511.01163},
  year={2025}
}
```
