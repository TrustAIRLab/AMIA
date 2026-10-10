"""
Local Wikipedia-style environment for the ReAct agent (StrategyQA).

`WikiEnv` is a gym environment exposing the ReAct action interface:
  Search[entity]  -- DPR retrieval over the StrategyQA paragraph corpus (the
                     agent's memory); returns the top-knn passage.
  Lookup[keyword] -- next sentence containing keyword in the current passage.
  Finish[answer]  -- ends the episode and, unless disabled, writes `answer` to
                     memory (this is where an injected answer is persisted).

Corpus and the DPR embedding cache live under `data_dir`. The cache is the same
StrategyQA DPR cache used by the QA agent (identical corpus + model), so the two
agents share it.
"""
import json
import os
import pickle
import random
from pathlib import Path

import gym
import torch
from transformers import AutoTokenizer, DPRContextEncoder
from tqdm import tqdm

_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# CUDA when available, CPU otherwise (e.g. macOS). Override with AMIA_DEVICE.
_DEVICE = os.environ.get("AMIA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


class textSpace(gym.spaces.Space):
    def contains(self, x) -> bool:
        return isinstance(x, str)


class WikiEnv(gym.Env):
    def __init__(self, embedder_name="dpr-ctx_encoder-single-nq-base", knn=1,
                 memory_type="answer_based", disable_update=False,
                 data_dir=_DEFAULT_DATA_DIR,
                 corpus_file="strategyqa_train_paragraphs.json",
                 multi_database_folder=None):
        super().__init__()
        self.memory_type = memory_type
        self.disable_update = disable_update
        self.knn = knn
        self.data_dir = data_dir
        self.page = None            # current passage (set by Lookup context)
        self.obs = None
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        self.question = None
        self.observation_space = self.action_space = textSpace()

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
        print("Local WikiEnv initialized: ", len(self.database))

        # Shared with the QA agent's QADataBase (same StrategyQA corpus + DPR model).
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
        self.embedding_id = list(self.database.keys())

        # Cross-agent transfer: append another agent's exported memory delta.
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

    def construct_lookup_list(self, keyword):
        if self.page is None:
            return []
        paragraphs = [p.strip() for p in self.page.split("\n") if p.strip()]
        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        return [s for s in sentences if keyword.lower() in s.lower()]

    def local_retrieve_step(self, entity):
        query_embedding = self._embed(entity).detach().cpu().numpy().tolist()
        query_embedding = torch.tensor(query_embedding, dtype=torch.float32).to(_DEVICE)
        cos_sim = torch.nn.functional.cosine_similarity(
            query_embedding, self.db_embeddings, dim=1).cpu().numpy()
        sorted_indices = cos_sim.argsort()[::-1]
        target_index = random.choice(sorted_indices[:self.knn])
        top_id = self.embedding_id[target_index]
        self.obs = self.database[top_id]["content"] + "\n"
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

    def step(self, action):
        reward, done = 0, False
        action = action.strip()
        if self.answer is not None:    # already finished
            return self.obs, reward, True, {"steps": self.steps, "answer": self.answer}

        if action.startswith("search[") and action.endswith("]"):
            self.local_retrieve_step(action[len("search["):-1])
        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup["):-1]
            if self.lookup_keyword != keyword:
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                self.obs = "No more results.\n"
            else:
                self.obs = (f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) "
                            + self.lookup_list[self.lookup_cnt])
                self.lookup_cnt += 1
        elif action.startswith("finish[") and action.endswith("]"):
            self.answer = action[len("finish["):-1]
            if not self.disable_update:
                self.update(self.answer, self.question if self.memory_type == "question_based" else None)
            done = True
            self.obs = f"Episode finished, reward = {reward}\n"
        elif action.startswith("think[") and action.endswith("]"):
            self.obs = "Nice thought."
        else:
            self.obs = f"Invalid action: {action}"

        self.steps += 1
        info = {"steps": self.steps, "answer": self.answer}
        if self.answer is not None:    # reset for the next batched question
            self.answer = None
        return self.obs, reward, done, info

    def update(self, answer, question=None):
        key_text = answer if self.memory_type == "answer_based" else question
        query_embedding = self._embed(key_text)
        self.db_embeddings = torch.cat((self.db_embeddings, query_embedding), 0)
        new_id = f"new_memory_{len(self.database)}"
        self.embedding_id.append(new_id)
        self.database[new_id] = {"content": answer}
