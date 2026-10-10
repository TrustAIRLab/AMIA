"""
A-MEM-backed drop-in replacement for the ReAct WikiEnv (amia.react_wikienv).

Subclasses WikiEnv and overrides three things:
  - __init__: skips the StrategyQA corpus load + DPR embedding; instantiates an
    AgenticMemorySystem (local all-MiniLM embeddings, ChromaDB store).
  - update(answer, question): writes the agent response via add_note().
  - local_retrieve_step(entity): search_agentic(entity) -> self.obs.

The base WikiEnv.step() (Search / Lookup / Finish / Think dispatch) is inherited
unchanged. The vendored `agentic_memory` package sits next to this file (added to
sys.path by the runner). Used by amem_react_gpt4mini.py / amem_react_llama.py.
"""
from amia.react_wikienv import WikiEnv, textSpace
from agentic_memory.memory_system import AgenticMemorySystem


class AMemWikiEnv(WikiEnv):
    """WikiEnv with the DPR store swapped for AgenticMemorySystem."""

    def __init__(self, knn=1, memory_type="answer_based", disable_update=False,
                 amem_llm_model="gpt-4.1-mini", amem_embed_model="all-MiniLM-L6-v2",
                 amem_evo_threshold=100, amem_api_key=None, amem_base_url=None):
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

        self._memory = AgenticMemorySystem(
            model_name=amem_embed_model, llm_backend="openai", llm_model=amem_llm_model,
            evo_threshold=amem_evo_threshold, api_key=amem_api_key,
        )

        self.database = {}
        self.embedding_id = []
        self.db_embeddings = None
        self.embedding_model = None
        self.embedding_tokenizer = None

    def update(self, answer, question=None):
        if self.disable_update:
            return
        content = f"User: {question}\nAssistant: {answer}" if question else str(answer)
        try:
            self._memory.add_note(content=content)
        except Exception as e:
            print(f"[AMemWikiEnv.update] add_note fail: {e}")

    def local_retrieve_step(self, entity):
        try:
            results = self._memory.search_agentic(entity, k=self.knn)
        except Exception:
            results = []
        ctx = [r.get("content", "") or "" for r in (results or [])[:self.knn] if isinstance(r, dict)]
        self.obs = "\n".join(ctx) + "\n" if ctx else "No relevant memory.\n"
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

    def bulk_insert_injects(self, qadv_list, response_list):
        """Write each parsed Finish[] content as an A-MEM note. A-MEM stores the
        agent response verbatim (paper: response-only); it is not role-tagged, and
        the PI query is never stored (that would leak the attack scaffolding into
        the retrieved note)."""
        for _q, r in zip(qadv_list, response_list):
            content = str(r)
            try:
                self._memory.add_note(content=content)
            except Exception as e:
                print(f"[AMemWikiEnv.bulk_insert] fail: {e}")

    def replace_stm_with_benign(self, n=10, seed=42):
        """No-op for A-MEM (no short-term-memory concept)."""
        return None
