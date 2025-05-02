from transformers import pipeline, set_seed
import random
import json

generator = pipeline("text-generation", model="gpt2-xl")
set_seed(42)

queries = [
    "What causes climate change?",
    "How does a car engine work?",
    "Who discovered penicillin?",
    "What is quantum entanglement?",
    "How do vaccines provide immunity?",
    "What is the process of photosynthesis?",
    "Why is the sky blue?",
    "What is blockchain technology?",
    "How does the internet work?",
    "What are the symptoms of diabetes?"
]

def generate_relevant_passage(query):
    prompt = f"Q: {query}\nA:"
    return generator(prompt, max_length=200, num_return_sequences=1, temperature=0.9)[0]["generated_text"].strip()

distractor_topics = [
    "banana farming", "jazz music history", "volcanic eruptions", "modern abstract art",
    "electric skateboard features", "frog mating rituals", "Mediterranean cuisine",
    "esports competitions", "interior design in 2024", "19th-century exploration"
]

def generate_distractor():
    prompt = f"This passage discusses {random.choice(distractor_topics)}. "
    return generator(prompt, max_length=200, num_return_sequences=1, temperature=0.9)[0]["generated_text"].strip()

def generate_model_backed_dataset(num_queries=10, num_docs=10):
    dataset = []
    for i in range(num_queries):
        query = queries[i % len(queries)]
        relevant = generate_relevant_passage(query)
        candidates = [{"text": relevant, "relevant": True}]
        for _ in range(num_docs - 1):
            candidates.append({"text": generate_distractor(), "relevant": False})
        random.shuffle(candidates)
        dataset.append({"query": query, "candidates": candidates})
    return dataset

dataset = generate_model_backed_dataset()
with open("improved_real_passages.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("✅ Saved to improved_real_passages.json")
