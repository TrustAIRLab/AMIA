"""
MemoryOS-backed drop-in replacement for the ReAct WikiEnv (amia.react_wikienv).

Subclasses WikiEnv and overrides three things:
  - __init__: skips the StrategyQA corpus load + DPR embedding; instantiates a
    MemoryOSAdapter (which preloads the same StrategyQA paragraph corpus WikiEnv
    used, mirroring the flat-KV benign base memory).
  - update(answer, question): delegates to adapter.update().
  - local_retrieve_step(entity): adapter.retrieve(entity) -> self.obs.

The base WikiEnv.step() (Search / Lookup / Finish / Think dispatch) is inherited
unchanged. Used by memoryos_react_gpt4mini.py / memoryos_react_llama.py.
"""
from amia.react_wikienv import WikiEnv, textSpace
from memoryos_adapter import MemoryOSAdapter


class MemoryOSWikiEnv(WikiEnv):
    """WikiEnv with the DPR store swapped for a MemoryOSAdapter."""

    def __init__(self, knn=1, memory_type="answer_based", disable_update=False,
                 base_db_path=None, data_storage_path="/tmp/memoryos_react",
                 template_data_path=None, stm_mode="history",
                 embedding_model_name="all-MiniLM-L6-v2", embedding_device="cpu",
                 memoryos_knn=None):
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

        self._adapter = MemoryOSAdapter(
            base_db_path=base_db_path,
            data_storage_path=data_storage_path,
            embedding_model_name=embedding_model_name,
            embedding_device=embedding_device,
            template_data_path=template_data_path,
            knn=memoryos_knn if memoryos_knn is not None else knn,
            preload=True, wipe_existing=True, stm_mode=stm_mode,
        )

        self.database = {}
        self.embedding_id = []
        self.db_embeddings = None
        self.embedding_model = None
        self.embedding_tokenizer = None

    def update(self, answer, question=None):
        if self.disable_update:
            return
        self._adapter.update(answer, question=question)

    def local_retrieve_step(self, entity):
        self.obs = self._adapter.retrieve(entity)
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

    def bulk_insert_injects(self, qadv_list, response_list, sim_threshold=0.6):
        return self._adapter.bulk_insert_injects(qadv_list, response_list, sim_threshold=sim_threshold)

    def replace_stm_with_benign(self, n=10, seed=42):
        return self._adapter.replace_stm_with_benign(n=n, seed=seed)
