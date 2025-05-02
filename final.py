# import torch
# from transformers import AutoTokenizer, AutoModel
# import numpy as np
# import pandas as pd
# import random
# import json
# import matplotlib.pyplot as plt
# from collections import defaultdict
# import os
#
# # Setup
# model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
# model.eval()
# os.makedirs("outputs", exist_ok=True)
#
#
# # Encode with top-k pruning
# def encode_topk(text, k):
#     with torch.no_grad():
#         inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#         outputs = model(**inputs).last_hidden_state.squeeze(0)
#         norms = torch.norm(outputs, dim=1)
#         topk_indices = torch.topk(norms, min(k, norms.size(0))).indices
#         selected = outputs[topk_indices]
#         return selected.mean(dim=0).numpy(), topk_indices.tolist()
#
#
# # Expanded dataset generator
# def generate_large_dataset(num_queries=30, num_docs=15):
#     topics = [
#         "what is a black hole", "how do vaccines work", "capital of France", "who wrote Hamlet",
#         "how to boil an egg", "photosynthesis process", "first law of thermodynamics",
#         "definition of democracy", "best way to study", "causes of global warming",
#         "largest mammal on Earth", "benefits of meditation", "how does a car engine work",
#         "symptoms of COVID-19", "basic algebra rules", "importance of sleep",
#         "effect of caffeine", "how to change a tire", "fastest land animal", "history of the internet"
#     ]
#
#     dataset = []
#     for i in range(num_queries):
#         q_topic = random.choice(topics)
#         relevant = f"This passage provides a clear and specific answer about {q_topic}."
#         candidates = [{"text": relevant, "relevant": True}]
#         for _ in range(num_docs - 1):
#             unrelated = f"This is irrelevant text related to {random.choice(topics)}."
#             candidates.append({"text": unrelated, "relevant": False})
#         random.shuffle(candidates)
#         dataset.append({"query": q_topic, "candidates": candidates})
#
#     print(dataset)
#     return dataset
#
#
# # IR metrics
# def compute_ir_metrics(ranks):
#     mrr = np.mean([1 / r for r in ranks])
#     p1 = np.mean([1 if r == 1 else 0 for r in ranks])
#     recall3 = np.mean([1 if r <= 3 else 0 for r in ranks])
#     return {"MRR": mrr, "P@1": p1, "Recall@3": recall3, "Avg Rank": np.mean(ranks)}
#
#
# # Run full experiment
# def run_experiment(data, k_values):
#     metrics = []
#     similarities = defaultdict(list)
#     token_overlap = defaultdict(list)
#     base_tokens = {}
#
#     for k in k_values:
#         ranks = []
#         for ex in data:
#             query = ex["query"]
#             q_vec, q_topk = encode_topk(query, k)
#             scores = []
#             doc_tok_map = {}
#
#             for idx, doc in enumerate(ex["candidates"]):
#                 d_vec, d_topk = encode_topk(doc["text"], k)
#                 sim = np.dot(q_vec, d_vec)
#                 scores.append((sim, doc["relevant"], idx))
#                 doc_tok_map[idx] = d_topk
#                 similarities[query].append({
#                     "k": k, "doc_id": idx, "relevant": doc["relevant"], "score": float(sim)
#                 })
#
#             sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
#             for i, (_, rel, _) in enumerate(sorted_scores, start=1):
#                 if rel:
#                     ranks.append(i)
#                     break
#
#             # Token overlap tracking
#             if k == 512:
#                 base_tokens[query] = {"q": set(q_topk), "d": {i: set(doc_tok_map[i]) for i in doc_tok_map}}
#
#             else:
#                 q_overlap = len(set(q_topk) & base_tokens[query]["q"]) / max(len(set(q_topk)), 1)
#                 d_overlap = np.mean([
#                     len(set(doc_tok_map[i]) & base_tokens[query]["d"][i]) / max(len(set(doc_tok_map[i])), 1)
#                     for i in doc_tok_map if i in base_tokens[query]["d"]
#                 ])
#                 token_overlap["query"].append((k, q_overlap))
#                 token_overlap["doc"].append((k, d_overlap))
#
#         m = compute_ir_metrics(ranks)
#         m["k"] = k
#         metrics.append(m)
#
#     return metrics, similarities, token_overlap
#
#
# # Plotting function
# def generate_plots(metrics, token_overlap):
#     df = pd.DataFrame(metrics)
#     df.to_csv("outputs/enhanced_metrics.csv", index=False)
#
#     plt.figure()
#     plt.plot(df["k"], df["MRR"], label="MRR")
#     plt.plot(df["k"], df["P@1"], label="P@1")
#     plt.plot(df["k"], df["Recall@3"], label="Recall@3")
#     plt.xlabel("Top-k tokens")
#     plt.ylabel("Score")
#     plt.title("IR Metrics vs. Top-k")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig("outputs/metrics_vs_k.png")
#
#     # Token overlap
#     for key in token_overlap:
#         overlap = token_overlap[key]
#         overlap = sorted(overlap, key=lambda x: x[0])
#         ks = [x[0] for x in overlap]
#         vals = [x[1] for x in overlap]
#         plt.figure()
#         plt.plot(ks, vals, marker='o')
#         plt.xlabel("Top-k")
#         plt.ylabel("Token Overlap with k=512")
#         plt.title(f"{key.capitalize()} Token Overlap vs. k")
#         plt.grid(True)
#         plt.savefig(f"outputs/token_overlap_{key}.png")
#
#
# # Execute
# dataset = generate_large_dataset()
# k_vals = [2, 4, 6, 8, 10, 20, 30, 40, 50, 100, 512]
# metric_data, sim_log, tok_overlap = run_experiment(dataset, k_vals)
#
# # Save and plot
# with open("outputs/similarity_log.json", "w") as f:
#     json.dump(sim_log, f, indent=2)
# generate_plots(metric_data, tok_overlap)
#
# # Display summary metrics
# df_final = pd.DataFrame(metric_data)
# print("\nFinal Enhanced IR Metrics")
# print(df_final)
# # import ace_tools as tools;
#
# # tools.display_dataframe_to_user(name="Final Enhanced IR Metrics", dataframe=df_final)


import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import random
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# Setup and model loading
model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
os.makedirs("outputs", exist_ok=True)

# Semantic topic pool
real_topics = [
    "what is a black hole", "how do vaccines work", "capital of France", "who wrote Hamlet",
    "how to boil an egg", "photosynthesis process", "first law of thermodynamics",
    "definition of democracy", "best way to study", "causes of global warming",
    "largest mammal on Earth", "benefits of meditation", "how does a car engine work",
    "symptoms of COVID-19", "basic algebra rules", "importance of sleep",
    "effect of caffeine", "how to change a tire", "fastest land animal", "history of the internet"
]

# Top-k pooling function
def encode_topk(text, k):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs).last_hidden_state.squeeze(0)
        norms = torch.norm(outputs, dim=1)
        topk_indices = torch.topk(norms, min(k, norms.size(0))).indices
        selected = outputs[topk_indices]
        return selected.mean(dim=0).numpy(), topk_indices.tolist()

# Realistic synthetic dataset generation
def generate_realistic_dataset(num_queries=30, num_docs=15):
    dataset = []
    for i in range(num_queries):
        query = real_topics[i % len(real_topics)]
        relevant = f"This passage provides a clear and accurate answer to the question: {query}."
        candidates = [{"text": relevant, "relevant": True}]
        for _ in range(num_docs - 1):
            distractor = f"This document discusses unrelated topics and does not answer: {query}."
            candidates.append({"text": distractor, "relevant": False})
        random.shuffle(candidates)
        dataset.append({"query": query, "candidates": candidates})

    print(dataset)
    return dataset

# IR metrics
def compute_ir_metrics(ranks):
    mrr = np.mean([1 / r for r in ranks])
    p1 = np.mean([1 if r == 1 else 0 for r in ranks])
    recall3 = np.mean([1 if r <= 3 else 0 for r in ranks])
    return {"MRR": mrr, "P@1": p1, "Recall@3": recall3, "Avg Rank": np.mean(ranks)}

# Full experiment runner
def run_experiment(data, k_values):
    metrics = []
    similarities = defaultdict(list)
    token_overlap = defaultdict(list)
    base_tokens = {}

    for k in k_values:
        ranks = []
        for ex in data:
            query = ex["query"]
            q_vec, q_topk = encode_topk(query, k)
            scores = []
            doc_tok_map = {}

            for idx, doc in enumerate(ex["candidates"]):
                d_vec, d_topk = encode_topk(doc["text"], k)
                sim = np.dot(q_vec, d_vec)
                scores.append((sim, doc["relevant"], idx))
                doc_tok_map[idx] = d_topk
                similarities[query].append({
                    "k": k, "doc_id": idx, "relevant": doc["relevant"], "score": float(sim)
                })

            sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
            for i, (_, rel, _) in enumerate(sorted_scores, start=1):
                if rel:
                    ranks.append(i)
                    break

            if k == 512:
                base_tokens[query] = {"q": set(q_topk), "d": {i: set(doc_tok_map[i]) for i in doc_tok_map}}
            elif query in base_tokens:
            # else:
                q_overlap = len(set(q_topk) & base_tokens[query]["q"]) / max(len(set(q_topk)), 1)
                d_overlap = np.mean([
                    len(set(doc_tok_map[i]) & base_tokens[query]["d"].get(i, set())) / max(len(set(doc_tok_map[i])), 1)
                    for i in doc_tok_map
                ])
                token_overlap["query"].append((k, q_overlap))
                token_overlap["doc"].append((k, d_overlap))

        m = compute_ir_metrics(ranks)
        m["k"] = k
        metrics.append(m)

    return metrics, similarities, token_overlap

# Plotting
def generate_plots(metrics, token_overlap):
    # print(f"Token Overlap: {token_overlap}")
    df = pd.DataFrame(metrics)
    df.to_csv("outputs/enhanced_metrics.csv", index=False)

    plt.figure()
    plt.plot(df["k"], df["MRR"], label="MRR")
    plt.plot(df["k"], df["P@1"], label="P@1")
    plt.plot(df["k"], df["Recall@3"], label="Recall@3")
    plt.xlabel("Top-k tokens")
    plt.ylabel("Score")
    plt.title("IR Metrics vs. Top-k")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/metrics_vs_k.png")

    for key in token_overlap:
        overlap = sorted(token_overlap[key], key=lambda x: x[0])
        ks = [x[0] for x in overlap]
        vals = [x[1] for x in overlap]
        plt.figure()
        plt.plot(ks, vals, marker='o')
        plt.xlabel("Top-k")
        plt.ylabel("Token Overlap with k=512")
        plt.title(f"{key.capitalize()} Token Overlap vs. k")
        plt.grid(True)
        plt.savefig(f"outputs/token_overlap_{key}.png")

# Run everything

# Load dataset
with open("improved_real_passages.json", "r") as f:
    dataset = json.load(f)
# dataset = generate_realistic_dataset()
k_vals = [512, 2, 4, 6, 8, 10, 20, 30, 40, 50, 100]
metric_data, sim_log, tok_overlap = run_experiment(dataset, k_vals)

# Save logs and metrics
with open("outputs/similarity_log.json", "w") as f:
    json.dump(sim_log, f, indent=2)
generate_plots(metric_data, tok_overlap)

# Final display
df_final = pd.DataFrame(metric_data)
# df_final.to_csv("enhanced_metrics.csv", index=False)
# Display summary metrics
print("\nFinal Enhanced IR Metrics")
print(df_final)
