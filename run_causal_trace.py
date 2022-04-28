import os

import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
import numpy

MODEL = 'distilgpt2'
NOISE = 0.1
prng = numpy.random.RandomState(1)
KNOWN_FACTS_PATH = './counterfact/compiled/distilgpt2_cf.json'
device = 'cuda'

from matplotlib import pyplot as plt


def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None):
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel("single restored layer")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers", fontsize=5)
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def get_subject_index(prompt_ids, subject_ids):
    for idx in range(len(prompt_ids) - len(subject_ids) + 1):
        if prompt_ids[idx: idx + len(subject_ids)] == subject_ids:
            return idx, idx + len(subject_ids)


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, prompt, subject):
    prompt_ids = tok(prompt, return_tensors='pt').to(device)
    prompt_ids = prompt_ids['input_ids'][0]
    toks = decode_tokens(tokenizer, prompt_ids)
    whole_string = "".join(toks)
    char_loc = whole_string.index(subject)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(subject):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def get_prediction(model, tok, prompt):
    prompt_ids = tok(prompt, return_tensors='pt').to(device)
    outs = model(**prompt_ids)['logits'][:, -1]
    smax = torch.softmax(outs, dim=1)
    prob, idx = torch.max(smax, dim=1)
    return prob, idx


def get_modules_to_trace(model, layers):
    module_dict = {}
    for n, m in model.named_modules():
        if n in layers:
            module_dict[n] = m
    return module_dict


def get_layer_name(layer_idx, kind=None):
    return f"transformer.h.{layer_idx}" if not kind else f"transformer.h.{layer_idx}.{kind}"


def trace_hook(layer, tok_idx, subject_range, self, inputs, outputs):
    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    if layer == 'transformer.wte':
        if subject_range is not None:
            b, e = subject_range
            outputs[1:, b:e] += NOISE * torch.from_numpy(prng.randn(outputs.shape[0] - 1, e - b, outputs.shape[2])).to(
                outputs.device)
            return outputs
    elif tok_idx != None:
        h = untuple(outputs)
        h[1:, tok_idx] = h[0, tok_idx]
        return outputs


def run_single_trace(model, num_layers, prompt_ids, subject_range, tok_idx, layer_idx, answer_tok_idx, window=None,
                     kind=None):
    layers = set(get_layer_name(l, kind) for l in
                 range(max(0, layer_idx - window // 2), min(num_layers, layer_idx - (-window // 2))))
    layers.add(get_layer_name(layer_idx, kind))
    layers.add('transformer.wte')
    module_dict = get_modules_to_trace(model, layers)
    ## Register hooks
    hooks = []
    for layer, module in module_dict.items():
        hooks.append(module.register_forward_hook(partial(trace_hook, layer, tok_idx, subject_range)))
    ## Calculate outputs
    outs = model(**prompt_ids)
    prob = torch.softmax(outs.logits[1:, -1, :], dim=1).mean(dim=0)[answer_tok_idx]
    ## Unregister hook
    for hook in hooks:
        hook.remove()
    return prob


def causal_trace(model, tok, prompt, subject_range, kind=None, window=3):
    window = 0 if kind == None else window

    answer_prob, answer_tok_idx = get_prediction(model, tok, prompt)

    prompt = [prompt] * 11
    prompt_ids = tok(prompt, return_tensors='pt').to(device)

    num_tokens = prompt_ids['input_ids'].size(1)
    num_layers = model.config.n_layer
    results = []
    for tok_idx in range(num_tokens):
        tok_results = []
        for layer_idx in range(num_layers):
            single_trace_diff = run_single_trace(model, num_layers, prompt_ids, subject_range, tok_idx, layer_idx,
                                                 answer_tok_idx, kind=kind, window=window)
            tok_results.append(single_trace_diff)
        results.append(torch.stack(tok_results))
    diffs = torch.stack(results).detach().cpu()
    diffs = diffs.reshape(diffs.shape[:2])

    low_score = run_single_trace(model, num_layers, prompt_ids, subject_range, None, layer_idx, answer_tok_idx,
                                 kind=kind, window=window)
    return dict(
        scores=diffs,
        low_score=low_score,
        high_score=answer_prob,
        input_ids=prompt_ids["input_ids"][0],
        input_tokens=decode_tokens(tok, prompt_ids["input_ids"][0]),
        subject_range=subject_range,
        answer=tok.decode(answer_tok_idx),
        window=window,
        kind=kind or "",
    )


model = AutoModelForCausalLM.from_pretrained(MODEL)
model.to(device)
tok = AutoTokenizer.from_pretrained(MODEL)


def run_all_traces(model, kind=None, window=3):
    window = 0 if kind == None else window
    avg_scores = torch.zeros(6, 6)
    avg_low_scores = 0
    with open(KNOWN_FACTS_PATH) as f:
        knowns = json.load(f)
    i = 0
    for known in tqdm(knowns[:300]):
        # if i % 1 == 0:
        #   print_num_tensors()
        #   print(torch.cuda.memory_allocated() / 1000000000)
        # i+=1
        prompt = known['requested_rewrite']['prompt'].replace('{}', known['requested_rewrite']['subject'])
        subject = known['requested_rewrite']['subject']
        if '.net' not in prompt.lower():
            subject_range = find_token_range(tok, prompt, subject)
            trace = causal_trace(model, tok, prompt, subject_range, kind=kind)
            avg_scores += summarize_scores(trace, 6)
            avg_low_scores += trace['low_score'].detach().cpu()
    results = {'scores': avg_scores / len(knowns),
               'low_score': avg_low_scores / len(knowns),
               'kind': kind,
               'window': window}
    return results


def plot_avg_heatmap(result, savepdf=None, title=None, xlabel=None):
    differences = result["scores"]
    low_score = result["low_score"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = ["First subj token", "Mid subj tokens", "Last subj token",
              "First subsequent token", "Intermediate tokens", "Final token"]

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel("single restored layer")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def summarize_scores(result, num_layers):
    agg_scores = torch.zeros(6, num_layers)
    first_subject_idx, last_subject_idx = result['subject_range']
    mid_subject_range = (first_subject_idx + 1, last_subject_idx - 1)
    first_rem_idx = last_subject_idx + 1
    last_rem_idx = len(result['input_tokens']) - 1
    mid_rem_range = (first_rem_idx + 1, last_rem_idx - 1)
    # print(first_subject_idx, mid_subject_range, last_subject_idx, first_rem_idx, mid_rem_range, last_rem_idx)

    scores = result['scores'].detach().cpu()
    agg_scores[0] = scores[first_subject_idx]
    if mid_subject_range[0] < mid_subject_range[1]:
        agg_scores[1] = scores[mid_subject_range[0]:mid_subject_range[1] + 1].mean(dim=0)
    agg_scores[2] = scores[last_subject_idx]
    agg_scores[3] = scores[first_rem_idx]
    if mid_rem_range[0] <= mid_rem_range[1]:
        agg_scores[4] = scores[mid_rem_range[0]:mid_rem_range[1] + 1].mean(dim=0)
    agg_scores[5] = scores[last_rem_idx]
    return agg_scores


def plot_all_traces(model):
    pdf_base_name = './{}_causal_trace_results.pdf'
    for kind in [None, "mlp", "attn"]:
        if kind is None:
            pdf_name = pdf_base_name.format("base")
        else:
            pdf_name = pdf_base_name.format(kind)
        results = run_all_traces(model, kind=kind)
        plot_avg_heatmap(results, savepdf=pdf_name)

plot_all_traces(model)

# with open(KNOWN_FACTS_PATH, 'r') as f:
#     knowns = json.load(f)
#     for kind in [None, "attn", "mlp"]:
#       i = 0
#       for known in tqdm(knowns[:300]):
#           if i % 10 == 0:
#             print(torch.cuda.memory_allocated() / 1000000000)
#           i += 1
#           prompt = known['requested_rewrite']['prompt'].replace('{}', known['requested_rewrite']['subject'])
#           subject = known['requested_rewrite']['subject']
#           if '.net' not in prompt.lower():
#             subject_range = find_token_range(tok, prompt, subject)
#             trace = causal_trace(model, tok, prompt, subject_range, kind=kind)
