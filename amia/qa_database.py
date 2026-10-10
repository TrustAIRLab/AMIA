"""
Flat key-value memory store for the QA agent (DPR retrieval).

The agent's memory is the StrategyQA paragraph corpus embedded with
`facebook/dpr-ctx_encoder-single-nq-base`. `retrieve(query)` returns the
top-knn passage by cosine similarity; `update(response)` appends a new memory
entry (this is where an injected response is persisted).

Corpus and the embedding cache live under `data_dir` (default: the repo's
`data/` folder). The cache is computed once from the corpus and reused; delete
it to force a recompute.
"""
import json
import os
import pickle
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, DPRContextEncoder
from tqdm import tqdm

# Repo root = parent of this package; data lives in <root>/data by default.
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# CUDA when available, CPU otherwise (e.g. macOS). Override with AMIA_DEVICE.
_DEVICE = os.environ.get("AMIA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


class QADataBase:
    def __init__(self, embedder_name=None, knn=1, memory_type="answer_based",
                 data_dir=_DEFAULT_DATA_DIR, corpus_file="strategyqa_train_paragraphs.json",
                 multi_database_folder=None):
        self.knn = knn
        self.memory_type = memory_type
        self.data_dir = data_dir
        if "dpr" in embedder_name:
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                "facebook/dpr-ctx_encoder-single-nq-base")
            self.embedding_model = DPRContextEncoder.from_pretrained(
                "facebook/dpr-ctx_encoder-single-nq-base").to(_DEVICE)
        self.load_db(embedder_name, corpus_file=corpus_file,
                     multi_database_folder=multi_database_folder)

    def _embed(self, text):
        tokenized = self.embedding_tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        input_ids = tokenized["input_ids"].to(_DEVICE)
        attention_mask = tokenized["attention_mask"].to(_DEVICE)
        with torch.no_grad():
            return self.embedding_model(input_ids, attention_mask).pooler_output

    def load_db(self, embedder_name, corpus_file="strategyqa_train_paragraphs.json",
                multi_database_folder=None):
        embedder_name = embedder_name.split("/")[-1]
        with open(os.path.join(self.data_dir, corpus_file), "r") as f:
            self.database = json.load(f)
        print("DataBase initialized: ", len(self.database))

        cache_dir = os.path.join(self.data_dir, "qa_embeddings")
        os.makedirs(cache_dir, exist_ok=True)
        embedding_path = os.path.join(cache_dir, f"embeddings_strategyqa_{embedder_name}.pkl")

        if Path(embedding_path).exists():
            with open(embedding_path, "rb") as f:
                self.db_embeddings = pickle.load(f)
        else:
            self.db_embeddings = []
            for paragraph_id in tqdm(self.database):
                emb = self._embed(self.database[paragraph_id]["content"])
                self.db_embeddings.append(emb.detach().cpu().numpy().tolist())
            with open(embedding_path, "wb") as f:
                pickle.dump(self.db_embeddings, f)

        self.db_embeddings = torch.tensor(self.db_embeddings, dtype=torch.float32).to(_DEVICE)
        self.db_embeddings = self.db_embeddings.squeeze(1)
        print("Embeddings loaded: ", self.db_embeddings.shape)

        self.embedding_id = list(self.database.keys())

        # Cross-agent transfer: append another agent's exported memory delta
        # (updated_*.pkl written by the source agent) into this victim's store.
        if multi_database_folder is not None:
            with open(os.path.join(multi_database_folder, "updated_embeddings.pkl"), "rb") as f:
                updated_embeddings = pickle.load(f)
            with open(os.path.join(multi_database_folder, "updated_database.pkl"), "rb") as f:
                updated_database = pickle.load(f)
            with open(os.path.join(multi_database_folder, "updated_embedding_id.pkl"), "rb") as f:
                updated_embedding_id = pickle.load(f)
            self.db_embeddings = torch.cat((self.db_embeddings, updated_embeddings.to(_DEVICE)), dim=0)
            self.embedding_id.extend(updated_embedding_id)
            self.database.update(updated_database)

        print("Total Demonstrations loaded: ", len(self.embedding_id))

    def retrieve(self, query):
        query_embedding = self._embed(query).detach().cpu().numpy().tolist()
        query_embedding = torch.tensor(query_embedding, dtype=torch.float32).to(_DEVICE)

        cos_sim = torch.nn.functional.cosine_similarity(
            query_embedding, self.db_embeddings, dim=1).cpu().numpy()
        sorted_indices = cos_sim.argsort()[::-1]
        # Pick one of the top-knn (knn=1 is a plain argmax).
        target_index = random.choice(sorted_indices[:self.knn])

        top_id = self.embedding_id[target_index]
        return self.database[top_id]["content"] + "\n"

    def update(self, response, question=None):
        # answer_based keys the memory on the response; question_based keys it on
        # the (attacker-supplied) question. Stored content is the response either way.
        key_text = response if self.memory_type == "answer_based" else question
        query_embedding = self._embed(key_text)

        self.db_embeddings = torch.cat((self.db_embeddings, query_embedding), 0)
        new_id = f"new_memory_{len(self.database)}"
        self.embedding_id.append(new_id)
        self.database[new_id] = {"content": response}
