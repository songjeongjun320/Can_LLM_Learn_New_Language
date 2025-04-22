import json
import torch
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
from typing import Literal
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- ModelConfig dataclass ---
@dataclass
class ModelConfig:
    name: str
    model_path: str
    is_local: bool
    model_type: Literal["causal", "encoder", "encoder-decoder"] = "causal"

# --- Model configuration list ---
MODEL_CONFIGS = [
    ModelConfig(
        name="Llama-3.2-3b-it",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.2-3B-Instruct",
        is_local=True,
        model_type="causal"
    ),
    ModelConfig(
        name="Llama-3.1-8b-it",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/downloaded_models/Llama-3.1-8B-Instruct",
        is_local=True,
        model_type="causal"
    ),
    # --- Encoder Model ---
    ModelConfig(
         name="BERT-base-uncased",
         model_path="bert-base-uncased",
         is_local=False,
         model_type="encoder"
     ),
    ModelConfig(
        name="bert-uncased-finetuned-kr-eng-v2",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/Tuned_Results/bert-uncased-finetuned-kr-eng-v2",
        is_local=True, # Assuming this is local based on path pattern
        model_type="encoder"
    ),
    ModelConfig(
        name="bert-uncased-finetuned-subtitle_dt",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/Tuned_Results/bert-uncased-finetuned-subtitle_dt",
        is_local=True, # Assuming this is local based on path pattern
        model_type="encoder"
    ),
    ModelConfig(
        name="bert-uncased-finetuned-subtitle_dt-used-reverse-v1",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/Tuned_Results/bert-uncased-finetuned-subtitle_dt_v1",
        is_local=True, # Assuming this is local based on path pattern
        model_type="encoder"
    ),
    ModelConfig(
        name="bert-uncased-finetuned-subtitle_dt-used-reverse-v2",
        model_path="/scratch/jsong132/Can_LLM_Learn_New_Language/FineTuning/BERT/Tuned_Results/bert-uncased-finetuned-subtitle_dt_v2",
        is_local=True, # Assuming this is local based on path pattern
        model_type="encoder"
    ),
]

# --- Function to get embeddings from a model ---
def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# --- Function to load Korean words from JSON ---
def load_korean_words(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['kor']

# --- Function to process Korean words and find closest English word ---
def find_closest_english_words(korean_words, model, tokenizer, english_vocab, english_embeddings):
    closest_words = {}
    for korean_word in korean_words:
        korean_embedding = get_embedding(korean_word, model, tokenizer)
        cosine_similarities = cosine_similarity([korean_embedding], english_embeddings)
        most_similar_idx = np.argmax(cosine_similarities)
        closest_words[korean_word] = english_vocab[most_similar_idx]
    return closest_words

# --- Main function ---
def main():
    # Load Korean words
    korean_words = load_korean_words('/scratch/jsong132/Can_LLM_Learn_New_Language/Evaluation/WORD_Embedding/words.json')

    for model_config in MODEL_CONFIGS:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_path)
        model = AutoModel.from_pretrained(model_config.model_path)

        # Example English vocabulary and embeddings (can be loaded from a pre-computed file or database)
        english_vocab = ["apple", "banana", "orange", "fruit", "car", "tree", "computer", "city", "dog", "cat"]  # Example list of English words
        english_embeddings = np.array([get_embedding(word, model, tokenizer) for word in english_vocab])  # Compute embeddings for English words

        # Find closest English words for each Korean word
        closest_words = find_closest_english_words(korean_words, model, tokenizer, english_vocab, english_embeddings)

        # Save the result to a JSON file named after the model
        output_filename = f"closest_english_words_{model_config.name}.json"
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            json.dump(closest_words, outfile, ensure_ascii=False, indent=4)

        print(f"Completed processing for {model_config.name} and saved the closest words to {output_filename}.")

if __name__ == "__main__":
    main()
