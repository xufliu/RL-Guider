# <p align=center> [ACL 2025] An LLM Agent for Molecular Optimization and Drug Pharmaceutical Improvement</p>

<div align="center">

[![Code](https://img.shields.io/badge/RL_Guider-Code-g.svg)](https://github.com/xufliu/RL-Guider)

</div>

---

>**RL-Guider: Leveraging Historical Decisions and Feedback for Drug Editing with Large Language Models** <br>

>[Xufeng Liu](https://xufliu.github.io/)<sup>* </sup>, [Yixuan Ding]()<sup>*† </sup>, [Jingxiang Qu](https://tom-q.netlify.app/), [Yichi Zhang](), [Wenhan Gao](https://wenhangao21.github.io/)<sup>‡ </sup>, [Yi Liu](https://jacoblau0513.github.io/)<sup>‡ </sup> <br>
>\* Equal contribution. <br>
>† Work done during an internship at Stony Brook University. <br>
>‡ Equal senior contribution. <br> 
>ACL 2025 <br>

> **Introduction:** Recent advances in large language models (LLMs) across diverse domains highlight their potential to transform scientific discovery, including drug editing. Traditional drug editing relies on iterative conversations with domain experts, refining a molecule until the desired property is achieved. This interactive process mirrors the strengths of LLMs. *However, existing approaches edit each molecule independently without leveraging knowledge from past edits.*  
>
> Human experts develop intuition about effective modifications over time by learning from historical experience. Accumulating past knowledge is pivotal for both humans and LLMs. *In this work, we propose RL-Guider — a reinforcement-learning agent that suggests edits to LLMs and improves over time by learning from evaluation feedback on past results.*  
>
> **RL-Guider** is the first framework to combine the comprehensive **"world-level"** knowledge of LLMs with knowledge accumulated from historical feedback. As a result, RL-Guider mitigates shortcomings of existing approaches and achieves superior performance.

<hr />

## Framework
<p align="center">
  <img src="figure/main_fig.jpg" /> 
</p>

## Environment
All dependencies are listed in [requirements.txt](requirements.txt). Install them with:

```bash
pip install -r requirements.txt
```

## Quickstart

Run an example to perform drug editing (small molecule) with LLaMA:

```bash
python run_ChatDrug.py --task_id=101 --C=0 --constraint='loose' --conversational_LLM='llama' --conversation_type='single'
```

Arguments:
- `--task_id`: The task identifier.  
- `--C`: Constraint strength.  
- `--constraint`: Editing constraint type (e.g., `loose`, `strict`).  
- `--conversational_LLM`: Choice of LLM (e.g., `llama`).  
- `--conversation_type`: Mode of conversation (e.g., `single`, `multi`).

### Using RL-Guider
Follow these steps to use RL-Guider:

1. Run the scripts in `gather_buffer/` to interact with LLMs and collect data.
2. Run the scripts in `process_buffer/` to preprocess the collected data.
3. Train RL models using the scripts in the `rl_train/` folder.
4. Run `run_planner_tree.py` to perform drug editing with RL-Guider.

## License

Released under the **MIT License**. See [LICENSE](LICENSE).

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{liu2025rl,
  title={RL-Guider: Leveraging Historical Decisions and Feedback for Drug Editing with Large Language Models},
  author={Liu, Xufeng and Ding, Yixuan and Qu, Jingxiang and Zhang, Yichi and Gao, Wenhan and Liu, Yi},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  pages={13121--13138},
  year={2025}
}