# Editing knowledge in distilled language models

This project implements causal tracing and several knowledge editing baselines against [DistilGPT2](https://huggingface.co/distilgpt2)

The code in this project is mainly taken from the [MEND](https://github.com/eric-mitchell/mend) and [ROME](https://github.com/kmeng01/rome) projects. The code has been updated to support distilled language models and the
causal tracing component has been re-written. 

NOTE: Requires a CUDA GPU to run

For setup, run 
```
bash setup_conda.sh
```

To run causal tracing (output heatmaps are saved to pdf files)

```commandline
python3 run_causal_trace.py
```

To run an editing algorithm, such as ROME

```commandline
python3 edit.py \
    --alg_name=ROME \
    --model_name=distilgpt2 \
    --hparams_fname=distilgpt2.json
```

To summarize results
```commandline
python3 summarize.py  --dir_name=ROME --runs=run_000
```
For more details on causal tracing and knowledge editing, see: 

[Mitchell, Eric, et al. "Fast model editing at scale." arXiv preprint arXiv:2110.11309 (2021).](https://arxiv.org/pdf/2110.11309.pdf)

[Meng, Kevin, et al. "Locating and editing factual knowledge in gpt." arXiv preprint arXiv:2202.05262 (2022).](https://arxiv.org/pdf/2202.05262.pdf)
