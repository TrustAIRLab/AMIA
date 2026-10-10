"""
Mem0-backed drop-in replacement for the ReAct WikiEnv (amia.react_wikienv).

Subclasses WikiEnv and overrides three things:
  - __init__: skips the StrategyQA corpus load + DPR embedding; instantiates a
    Mem0 Memory with a per-attack qdrant path.
  - update(answer, question): writes the (question, answer) pair via memory.add().
  - local_retrieve_step(entity): memory.search(entity) -> self.obs.

The base WikiEnv.step() (Search / Lookup / Finish / Think dispatch) is inherited
unchanged. Used by mem0_react_gpt4mini.py / mem0_react_llama.py.
"""
import os
import shutil

from amia.react_wikienv import WikiEnv, textSpace


class Mem0WikiEnv(WikiEnv):
    """WikiEnv with the DPR store swapped for a Mem0 Memory instance."""

    def __init__(self, knn=1, memory_type="answer_based", disable_update=False,
                 mem0_data_dir=None, mem0_collection_name="react_amia",
                 mem0_user_id="react_amia_user", mem0_llm_model="gpt-4.1-mini",
                 mem0_embedder="text-embedding-3-small", mem0_wipe_existing=True):
        # Replicate the WikiEnv state that step() relies on, without loading the DB.
        self.memory_type = memory_type
        self.disable_update = disable_update
        self.knn = knn
        self.page = None
        self.obs = None
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        self.question = None
        self.observation_space = self.action_space = textSpace()

        from mem0 import Memory
        if mem0_data_dir is None:
            mem0_data_dir = os.path.expanduser("~/.mem0_react_amia")
        if mem0_wipe_existing and os.path.isdir(mem0_data_dir):
            shutil.rmtree(mem0_data_dir, ignore_errors=True)
        os.makedirs(mem0_data_dir, exist_ok=True)
        # Clear the shared qdrant migrations dir so back-to-back attacks don't lock.
        shutil.rmtree(os.path.expanduser("~/.mem0/migrations_qdrant"), ignore_errors=True)

        config = {
            "llm": {"provider": "openai",
                    "config": {"model": mem0_llm_model, "temperature": 0.0, "max_tokens": 512}},
            "embedder": {"provider": "openai", "config": {"model": mem0_embedder}},
            "vector_store": {"provider": "qdrant",
                             "config": {"collection_name": mem0_collection_name,
                                        "embedding_model_dims": 1536,
                                        "path": os.path.join(mem0_data_dir, "qdrant")}},
        }
        self._memory = Memory.from_config(config)
        self._user_id = mem0_user_id

        # Stub the embedding attributes WikiEnv would otherwise have created.
        self.database = {}
        self.embedding_id = []
        self.db_embeddings = None
        self.embedding_model = None
        self.embedding_tokenizer = None

    def update(self, answer, question=None):
        if self.disable_update:
            return
        messages = []
        if question:
            messages.append({"role": "user", "content": str(question)})
        messages.append({"role": "assistant", "content": str(answer)})
        try:
            self._memory.add(messages, user_id=self._user_id)
        except Exception as e:
            print(f"[Mem0WikiEnv.update] mem0.add fail: {e}")

    def local_retrieve_step(self, entity):
        try:
            res = self._memory.search(entity, filters={"user_id": self._user_id}, version="v2")
        except Exception:
            try:
                res = self._memory.search(entity, user_id=self._user_id)
            except Exception as e:
                res = {"results": [], "err": str(e)}
        facts = res.get("results") if isinstance(res, dict) else (res or [])
        ctx = [r.get("memory", "") if isinstance(r, dict) else str(r) for r in (facts or [])][:self.knn]
        self.obs = "\n".join(ctx) + "\n" if ctx else "No relevant memory.\n"
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

    def bulk_insert_injects(self, qadv_list, response_list):
        """Write each attack as a user+assistant turn (the attacker query plus the
        agent's Finish[] response) in one pass, matching WikiEnv.update() and the
        flat-KV / QA runners. Mem0's extractor then records a question-anchored
        fact ("User asked <q> and was informed <answer>")."""
        for q, r in zip(qadv_list, response_list):
            try:
                self._memory.add(
                    [{"role": "user", "content": str(q)},
                     {"role": "assistant", "content": str(r)}],
                    user_id=self._user_id,
                )
            except Exception as e:
                print(f"[Mem0WikiEnv.bulk_insert] fail: {e}")

    def replace_stm_with_benign(self, n=10, seed=42):
        """No-op for Mem0 (no short-term-memory concept)."""
        return None
