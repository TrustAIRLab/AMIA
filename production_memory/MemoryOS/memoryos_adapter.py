"""
MemoryOS adapter for QA-AMIA.

Exposes the same retrieve() / update() / reset() surface as QADataBase so the
existing single-agent driver scripts (qa_llama.py, qa_gpt4mini.py, ...) can
dispatch to MemoryOS via the MEMORY_BACKEND=memoryos env-var branch without
changing the default flat-KV code path.

Design choices
--------------
1. MemoryOS's official pipeline invokes an LLM during STM->MTM eviction
   (multi-summary + page-continuity + page-meta) and during MTM->LTM
   promotion (profile + knowledge extraction). We do NOT want those calls
   during a QA-AMIA attack run because:
     (a) they consume the same vLLM instance the QA agent is using and
         would balloon the experiment wall-clock + API cost,
     (b) the security-relevant property we want to evaluate is the
         hierarchical retrieval mechanism (STM FIFO + MTM topic segments),
         not MemoryOS's LLM-driven memory consolidation.
   We therefore monkey-patch the eight LLM utility functions at the
   memoryos.utils module level to deterministic, no-LLM fallbacks before
   instantiating Memoryos. The structural STM/MTM logic still runs and
   the resulting memory is a faithful test of MemoryOS's *retrieval*
   behaviour, just without its on-line summarisation.

2. The eviction-time multi-summary fallback returns a single synthetic
   summary whose `content` is the raw text of the evicted batch and whose
   `keywords` are extracted via a deterministic regex-based fallback.
   MTM segment routing depends on these keywords for its Jaccard-style
   match, so we must produce a non-empty list - otherwise every page
   lands in a single catch-all segment and the hierarchy collapses.

3. mid_term_heat_threshold is set to math.inf so the profile/knowledge
   promotion path is never triggered even by accident.

4. The base corpus (~9,251 StrategyQA passages) is pre-loaded as
   synthetic (user_input=passage_first_200_chars, agent_response=passage)
   QA pairs. This mirrors QADataBase's preloaded benign memory.

5. retrieve() returns the agent_response of the top mid-term page,
   matching QADataBase.retrieve which returns the top-1 document text.
   knn>1 returns multiple concatenated.

6. update(response, question=None) mirrors QADataBase.update: stores the
   response, optionally keyed by a separate question (used by the
   input-key ablation). MemoryOS does not distinguish memory_type;
   the response is always the agent_response, question (if any) is the
   user_input.

Usage
-----
    from non_prompt_injection.memoryos_experiment.memoryos_adapter import (
        MemoryOSAdapter,
    )
    qa_database = MemoryOSAdapter(
        base_db_path="vanilla_rag_agent/database/strategyqa_train_paragraphs.json",
        data_storage_path="/tmp/memoryos_qa_pi",
        embedding_model_name="all-MiniLM-L6-v2",
        knn=1,
    )
    qa_database.update(response, question=...)
    context = qa_database.retrieve(query)
"""

import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Locate the vendored MemoryOS package and make it importable.
# We ship MemoryOS as a sibling git clone (memoryos_experiment/MemoryOS_repo/)
# and import its `memoryos-pypi/` source tree directly without pip-installing.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_MEMORYOS_PYPI_DIR = _THIS_DIR / "memoryos_pkg"
if not _MEMORYOS_PYPI_DIR.exists():
    raise ImportError(
        f"MemoryOS source not found at {_MEMORYOS_PYPI_DIR}. "
        "Clone https://github.com/BAI-LAB/MemoryOS into "
        f"{_THIS_DIR}/MemoryOS_repo first."
    )
if str(_MEMORYOS_PYPI_DIR) not in sys.path:
    sys.path.insert(0, str(_MEMORYOS_PYPI_DIR))


# ---------------------------------------------------------------------------
# LLM-feature monkey-patching.
# Must run BEFORE the first `from memoryos import Memoryos` import, because
# the LLM utility functions are imported by name in memoryos/updater.py and
# memoryos/memoryos.py. Patching at the utils module level after import
# would leave the original references bound inside those modules.
# ---------------------------------------------------------------------------
def _simple_keywords(text: str, max_n: int = 8) -> list:
    """Deterministic LLM-free keyword extraction.

    Pull up to max_n unique non-stopword alphabetic tokens of length>=4.
    Used as a fallback for MTM topic segmentation when the multi-summary
    LLM call is disabled.
    """
    stop = {
        "the", "and", "for", "with", "this", "that", "have", "from",
        "they", "their", "your", "you", "are", "was", "were", "been",
        "but", "not", "all", "any", "can", "has", "had", "will", "would",
        "user", "assistant", "answer", "question",
    }
    seen = set()
    out = []
    for tok in re.findall(r"[A-Za-z]{4,}", text.lower()):
        if tok in stop or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_n:
            break
    return out or ["general"]


_LLM_CALLABLE: Optional[Callable[[List[dict], int], str]] = None
_LLM_BATCH_CALLABLE: Optional[Callable[[List[List[dict]], int], List[str]]] = None
"""Global LLM callables for MemoryOS utility functions.

_LLM_CALLABLE: single-call interface for per-eviction summary.
_LLM_BATCH_CALLABLE: batched interface for preload-time summary (much faster
on vLLM since it can batch many prompts in one forward pass).

When set (via install_llm_summary_backend), gpt_generate_multi_summary uses
the single-call callable; bulk_preload uses the batched callable if set.
Other utilities (continuity, page_meta, profile, knowledge) remain mocked
because they do not affect retrieve_context behaviour at attack scale.
"""


def install_llm_summary_backend(
    llm_callable: Callable[[List[dict], int], str],
    llm_batch_callable: Optional[Callable[[List[List[dict]], int], List[str]]] = None,
):
    """Switch MemoryOS's gpt_generate_multi_summary to use a real LLM.

    Call this AFTER importing the adapter module but BEFORE constructing
    the MemoryOSAdapter. The driver provides a callable that wraps its
    vLLM instance; the adapter installs a real implementation that goes
    through that callable.

    Idempotent. Calling again with a different callable replaces it.
    """
    global _LLM_CALLABLE, _LLM_BATCH_CALLABLE
    _LLM_CALLABLE = llm_callable
    _LLM_BATCH_CALLABLE = llm_batch_callable
    # Force re-install of mocks so the new callable is picked up.
    import utils as _mos_utils
    _mos_utils._qa_amia_llm_mocked = False
    _install_llm_mocks()


def _install_llm_mocks():
    """Replace MemoryOS's LLM-calling utilities with deterministic no-ops.

    If a real LLM callable has been installed via install_llm_summary_backend,
    gpt_generate_multi_summary uses it; the other utilities still mock.

    Idempotent: re-running is a no-op once patched."""
    import utils as _mos_utils  # noqa: E402

    if getattr(_mos_utils, "_qa_amia_llm_mocked", False):
        return

    if _LLM_CALLABLE is not None:
        # Real implementation via the driver-provided LLM.
        def real_gpt_generate_multi_summary(text, client=None, model=None):
            try:
                from prompts import (
                    MULTI_SUMMARY_SYSTEM_PROMPT,
                    MULTI_SUMMARY_USER_PROMPT,
                )
            except ImportError:
                import prompts as _p
                MULTI_SUMMARY_SYSTEM_PROMPT = _p.MULTI_SUMMARY_SYSTEM_PROMPT
                MULTI_SUMMARY_USER_PROMPT = _p.MULTI_SUMMARY_USER_PROMPT
            messages = [
                {"role": "system", "content": MULTI_SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": MULTI_SUMMARY_USER_PROMPT.format(text=text),
                },
            ]
            response_text = _LLM_CALLABLE(messages, 512)
            # MemoryOS expects: {"input": text, "summaries": [...]}
            try:
                # The LLM is asked to return a JSON array.
                # Find the first [...] block (model may pad with prose).
                m = re.search(r"\[\s*\{.*?\}\s*\]", response_text, re.DOTALL)
                if m:
                    summaries = json.loads(m.group(0))
                else:
                    summaries = json.loads(response_text)
            except (json.JSONDecodeError, AttributeError):
                # Fall back to deterministic if LLM returns malformed JSON.
                summaries = [
                    {
                        "theme": "general",
                        "content": text[:512],
                        "keywords": _simple_keywords(text),
                    }
                ]
            return {"input": text, "summaries": summaries}

        fake_gpt_generate_multi_summary = real_gpt_generate_multi_summary
    else:
        def fake_gpt_generate_multi_summary(text, client=None, model=None):
            return {
                "summaries": [
                    {
                        "theme": "general",
                        "content": text[:512],
                        "keywords": _simple_keywords(text),
                    }
                ]
            }

    def fake_check_conversation_continuity(prev_page, curr_page, client=None, model=None):
        return False

    def fake_generate_page_meta_info(last_meta, current_page, client=None, model=None):
        text = (current_page or {}).get("user_input", "")
        return text[:128]

    def fake_gpt_user_profile_analysis(*args, **kwargs):
        return "None"

    def fake_gpt_knowledge_extraction(*args, **kwargs):
        return {"private": "None", "assistant_knowledge": "None"}

    def fake_gpt_personality_analysis(*args, **kwargs):
        return "None"

    def fake_gpt_update_profile(*args, **kwargs):
        return ""

    def fake_gpt_extract_theme(*args, **kwargs):
        return "general"

    def fake_gpt_summarize_dialogs(*args, **kwargs):
        return ""

    _mos_utils.gpt_generate_multi_summary = fake_gpt_generate_multi_summary
    _mos_utils.check_conversation_continuity = fake_check_conversation_continuity
    _mos_utils.generate_page_meta_info = fake_generate_page_meta_info
    _mos_utils.gpt_user_profile_analysis = fake_gpt_user_profile_analysis
    _mos_utils.gpt_knowledge_extraction = fake_gpt_knowledge_extraction
    _mos_utils.gpt_personality_analysis = fake_gpt_personality_analysis
    _mos_utils.gpt_update_profile = fake_gpt_update_profile
    _mos_utils.gpt_extract_theme = fake_gpt_extract_theme
    _mos_utils.gpt_summarize_dialogs = fake_gpt_summarize_dialogs

    # Also patch the names already re-bound at the module-level of the two
    # callers that import LLM funcs by name. Without this, the original
    # references stay bound inside updater.py / memoryos.py and the mocks
    # in utils.* are never reached.
    for module_name in ("updater", "memoryos"):
        if module_name in sys.modules:
            mod = sys.modules[module_name]
            for sym in (
                "gpt_generate_multi_summary",
                "check_conversation_continuity",
                "generate_page_meta_info",
                "gpt_user_profile_analysis",
                "gpt_knowledge_extraction",
            ):
                if hasattr(mod, sym):
                    setattr(mod, sym, getattr(_mos_utils, sym))

    _mos_utils._qa_amia_llm_mocked = True


_install_llm_mocks()
from memoryos import Memoryos  # noqa: E402  (after mocking)
# Re-run patch in case Memoryos import dragged updater.py into sys.modules
# after our first pass.
_install_llm_mocks()


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------
class MemoryOSAdapter:
    """Drop-in replacement for QADataBase backed by MemoryOS.

    Public surface mirrors QADataBase:
        retrieve(query: str) -> str
        update(response: str, question: Optional[str] = None) -> None
        reset() -> None
    """

    def __init__(
        self,
        base_db_path: str = "vanilla_rag_agent/database/strategyqa_train_paragraphs.json",
        data_storage_path: str = "/tmp/memoryos_qa_amia",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        user_id: str = "qa_amia_user",
        assistant_id: str = "qa_amia_assistant",
        knn: int = 1,
        short_term_capacity: int = 10,
        mid_term_capacity: int = 20000,
        mid_term_similarity_threshold: float = 0.6,
        preload: bool = True,
        wipe_existing: bool = True,
        embedding_device: Optional[str] = "cuda:0",
        template_data_path: Optional[str] = None,
        stm_mode: str = "score",  # "score" (legacy, top-k filter) | "history" (native prompt-history)
    ):
        """
        Parameters that matter for performance
        --------------------------------------
        embedding_device : str | None
            Where SentenceTransformer runs. "cuda:0" pins to GPU 0 and
            gives ~10x speedup over CPU on the preload pass. Set to None
            to let SentenceTransformer auto-select (defaults to GPU if
            available, but is unreliable when vLLM has already taken GPUs).
            For 4xA100 boxes with vLLM tp=4, "cuda:0" works because vLLM
            uses 0.85 memory_utilization and leaves ~12GB free per GPU
            which is far more than the 80MB SentenceTransformer footprint.

        template_data_path : str | None
            If provided AND the directory exists AND contains a valid
            preloaded MemoryOS snapshot, the adapter copies it to
            data_storage_path and skips the (slow) preload pass. After a
            fresh preload, if template_data_path is provided and does not
            yet exist, the adapter copies the freshly-preloaded state to
            template_data_path so subsequent constructions can reuse it.
            This makes the second and later attack iterations cheap.
        """
        self.base_db_path = base_db_path
        self.data_storage_path = os.path.abspath(data_storage_path)
        self.knn = knn
        self.stm_mode = stm_mode
        assert self.stm_mode in {"score", "history"}, \
            f"stm_mode must be 'score' or 'history', got {stm_mode!r}"
        self._template_data_path = (
            os.path.abspath(template_data_path) if template_data_path else None
        )

        if wipe_existing and os.path.isdir(self.data_storage_path):
            shutil.rmtree(self.data_storage_path)

        # If a preloaded template exists, copy it in place of running preload.
        copied_from_template = False
        if (
            preload
            and self._template_data_path
            and os.path.isdir(self._template_data_path)
        ):
            print(
                f"MemoryOSAdapter: copying preloaded template from "
                f"{self._template_data_path} -> {self.data_storage_path}"
            )
            shutil.copytree(self._template_data_path, self.data_storage_path)
            copied_from_template = True

        # Heat threshold = inf disables LTM profile/knowledge promotion;
        # combined with the monkey-patched utils above, this guarantees no
        # LLM calls happen anywhere in the MemoryOS pipeline.
        embedding_kwargs = {}
        if embedding_device:
            embedding_kwargs["device"] = embedding_device
        self.memoryos = Memoryos(
            user_id=user_id,
            openai_api_key="not-used-llm-features-disabled",
            openai_base_url=None,
            data_storage_path=self.data_storage_path,
            assistant_id=assistant_id,
            short_term_capacity=short_term_capacity,
            mid_term_capacity=mid_term_capacity,
            long_term_knowledge_capacity=10,
            retrieval_queue_capacity=max(7, knn * 3),
            mid_term_heat_threshold=math.inf,
            mid_term_similarity_threshold=mid_term_similarity_threshold,
            llm_model="disabled",
            embedding_model_name=embedding_model_name,
            embedding_model_kwargs=embedding_kwargs or None,
        )

        if preload and not copied_from_template:
            self._preload_base_corpus()
            # Cache the freshly-preloaded state for future iterations.
            if self._template_data_path and not os.path.isdir(
                self._template_data_path
            ):
                print(
                    f"MemoryOSAdapter: caching preloaded state to "
                    f"{self._template_data_path}"
                )
                # Make sure on-disk state is current before copying.
                self.memoryos.short_term_memory.save()
                self.memoryos.mid_term_memory.save()
                shutil.copytree(self.data_storage_path, self._template_data_path)

        # Disable on-disk persistence for the remainder of this adapter's
        # lifetime. The post-preload writes from search_sessions (~129 MB JSON
        # save per retrieve call) dominated the experiment wall-clock at
        # ~17 sec / retrieve. We do not need persistence: each attack starts
        # fresh from the cached template, and within-attack state is purely
        # in-memory. After this point, save() is a no-op on STM/MTM/LTM.
        self._disable_disk_saves()

    def _disable_disk_saves(self):
        """Replace save() on STM/MTM/LTM modules with a no-op.

        Called once at end of __init__ after preload completes. The on-disk
        snapshot is already persisted (and copied to template if requested);
        from here on we only need in-memory state.
        """
        noop = lambda *a, **k: None
        self.memoryos.short_term_memory.save = noop
        self.memoryos.mid_term_memory.save = noop
        self.memoryos.user_long_term_memory.save = noop
        if self.memoryos.assistant_long_term_memory:
            self.memoryos.assistant_long_term_memory.save = noop

    # -----------------------------------------------------------------------
    # Pre-loading the existing flat-KV base corpus into MemoryOS's STM/MTM
    # -----------------------------------------------------------------------
    def _preload_base_corpus(self):
        """Bulk-load the 9,251 StrategyQA passages into mid-term memory.

        Uses a fast batched path that bypasses MemoryOS's natural per-page
        STM->MTM eviction flow. The natural flow does ~3-5 disk I/O
        operations per page and dominates wall-clock (measured at ~0.26
        pages/sec on 4xA100 with GPU embeddings, ~10 hours for 9251
        pages). The bulk path:

          1. Batch-encodes all passage texts on GPU in a single
             SentenceTransformer.encode() call (~30 seconds for 9k pages).
          2. Chunks the passages into sessions of fixed size (matching the
             ~100 sessions of ~92 pages each that the natural flow would
             produce with diverse Wikipedia content at 0.6 cosine
             similarity threshold).
          3. Constructs session_obj dicts directly and writes to
             self.memoryos.mid_term_memory.sessions, saving the JSON
             file exactly once at the end.

        The attack-time `update()` calls remain unchanged (they go through
        the normal add_memory->STM->MTM flow with monkey-patched LLM
        utilities), so MemoryOS's hierarchical retrieval semantics are
        fully exercised at attack time.
        """
        import heapq
        import numpy as np
        with open(self.base_db_path) as f:
            db = json.load(f)

        texts, pids = [], []
        for pid, entry in db.items():
            t = entry.get("content", "")
            if t:
                texts.append(t)
                pids.append(pid)

        # --- Get a reference to the SentenceTransformer model that MemoryOS
        #     uses, by forcing it to load once via the standard get_embedding
        #     call. We then reuse the cached model for the batched encode.
        import utils as _mos_utils  # noqa: E402
        _ = _mos_utils.get_embedding(
            "warmup",
            model_name=self.memoryos.embedding_model_name,
            **self.memoryos.embedding_model_kwargs,
        )
        model_init_key = json.dumps(
            {
                "model_name": self.memoryos.embedding_model_name,
                **{
                    k: v
                    for k, v in self.memoryos.embedding_model_kwargs.items()
                    if k not in ("batch_size", "max_length")
                },
            },
            sort_keys=True,
        )
        model = _mos_utils._model_cache[model_init_key]
        print(
            f"MemoryOSAdapter: batch-encoding {len(texts)} passages "
            f"(model={self.memoryos.embedding_model_name}, "
            f"device={getattr(model, 'device', '?')})..."
        )

        # --- Batch encode page embeddings.
        # Match the format that MemoryOS's natural add_session() flow uses
        # (mid_term.py:128):
        #     full_text = f"User: {user_input} Assistant: {agent_response}"
        # so base-corpus pages are embedded with the SAME wrapper that
        # injected entries get at attack time. Without this, base pages
        # would be embedded as raw passage text while injected entries
        # would be embedded with "User: ... Assistant: ..." prefix, giving
        # injected entries a systematic embedding-space offset relative to
        # the base distribution and biasing retrieval comparisons in a
        # mechanism-irrelevant way.
        wrapped_texts = [f"User: {t[:200]} Assistant: {t}" for t in texts]
        page_embeddings = model.encode(
            wrapped_texts,
            batch_size=256,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # --- Chunk into sessions. 100 pages per chunk is a reasonable size:
        #     larger than the typical short-coherent natural chunk, but
        #     still produces ~92 distinct topical segments for 9251 pages,
        #     which is enough hierarchical diversity for the attack
        #     experiment.
        chunk_size = 100
        n_chunks = (len(texts) + chunk_size - 1) // chunk_size
        print(
            f"MemoryOSAdapter: building {n_chunks} sessions "
            f"({chunk_size} pages each) from {len(texts)} passages..."
        )

        # Lazy-import compute_segment_heat from MemoryOS
        from mid_term import compute_segment_heat
        from utils import get_timestamp, generate_id

        # --- Generate session summaries. If a batched LLM callable is
        #     installed, batch-call it to get real LLM topic summaries for
        #     all chunks at once (closer to native MemoryOS). Otherwise
        #     fall back to first-passage prefix.
        chunk_first_texts = [texts[i * chunk_size][:512] for i in range(n_chunks)]
        chunk_full_texts = [
            "\n".join(texts[i * chunk_size : i * chunk_size + chunk_size])[:8000]
            for i in range(n_chunks)
        ]

        if _LLM_BATCH_CALLABLE is not None:
            try:
                from prompts import (
                    MULTI_SUMMARY_SYSTEM_PROMPT,
                    MULTI_SUMMARY_USER_PROMPT,
                )
            except ImportError:
                import prompts as _p
                MULTI_SUMMARY_SYSTEM_PROMPT = _p.MULTI_SUMMARY_SYSTEM_PROMPT
                MULTI_SUMMARY_USER_PROMPT = _p.MULTI_SUMMARY_USER_PROMPT
            messages_list = [
                [
                    {"role": "system", "content": MULTI_SUMMARY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": MULTI_SUMMARY_USER_PROMPT.format(text=ct),
                    },
                ]
                for ct in chunk_full_texts
            ]
            print(
                f"MemoryOSAdapter: batch-calling LLM for {n_chunks} "
                f"chunk summaries..."
            )
            raw_summaries = _LLM_BATCH_CALLABLE(messages_list, 512)
            chunk_summaries = []
            for raw, fallback in zip(raw_summaries, chunk_first_texts):
                try:
                    m = re.search(r"\[\s*\{.*?\}\s*\]", raw, re.DOTALL)
                    parsed = json.loads(m.group(0)) if m else json.loads(raw)
                    if parsed and isinstance(parsed, list):
                        first = parsed[0]
                        content = first.get("content", "")[:512] or fallback
                        keywords = first.get("keywords") or _simple_keywords(content)
                    else:
                        content, keywords = fallback, _simple_keywords(fallback)
                except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
                    content, keywords = fallback, _simple_keywords(fallback)
                chunk_summaries.append((content, keywords))
            summary_texts_for_encode = [c for c, _ in chunk_summaries]
            summary_embeddings_all = model.encode(
                summary_texts_for_encode,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        else:
            chunk_summaries = [
                (ft, _simple_keywords(ft)) for ft in chunk_first_texts
            ]
            summary_texts_for_encode = chunk_first_texts
            summary_embeddings_all = model.encode(
                summary_texts_for_encode,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

        mtm = self.memoryos.mid_term_memory
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(texts))
            chunk_texts = texts[start:end]
            chunk_embs = page_embeddings[start:end]

            summary, summary_keywords = chunk_summaries[chunk_idx]
            summary_embedding = summary_embeddings_all[chunk_idx].tolist()

            details = []
            for j, (t, emb) in enumerate(zip(chunk_texts, chunk_embs)):
                details.append(
                    {
                        "page_id": f"preload_{start + j}",
                        "user_input": t[:200],
                        "agent_response": t,
                        "page_embedding": emb.tolist(),
                        "page_keywords": _simple_keywords(t),
                        "preloaded": True,
                        "analyzed": True,
                        "timestamp": "preload",
                        "pre_page": None,
                        "next_page": None,
                        "meta_info": None,
                    }
                )

            ts = get_timestamp()
            sid = generate_id("session")
            session_obj = {
                "id": sid,
                "summary": summary,
                "summary_keywords": summary_keywords,
                "summary_embedding": summary_embedding,
                "details": details,
                "L_interaction": len(details),
                "R_recency": 1.0,
                "N_visit": 0,
                "H_segment": 0.0,
                "timestamp": ts,
                "last_visit_time": ts,
                "access_count_lfu": 0,
            }
            session_obj["H_segment"] = compute_segment_heat(session_obj)
            mtm.sessions[sid] = session_obj
            mtm.access_frequency[sid] = 0
            heapq.heappush(mtm.heap, (-session_obj["H_segment"], sid))

        # Single save at the end -- huge wall-clock win vs per-session save.
        mtm.save()
        print(
            f"MemoryOSAdapter: bulk preloaded {len(texts)} passages "
            f"into {n_chunks} sessions in mid-term memory."
        )

    # -----------------------------------------------------------------------
    # QADataBase-compatible surface
    # -----------------------------------------------------------------------
    def retrieve(self, query: str) -> str:
        """Return retrieved context for the query.

        Two retrieval modes controlled by self.stm_mode:
          - "score" (legacy): score STM and MTM together, return top-knn.
            Used when matching flat-KV's top-1 retrieval semantics. STM
            is filtered by cosine to query.
          - "history" (native): retrieve MTM top-knn, AND prepend ALL STM
            entries verbatim as a "Recent History" block. Matches native
            MemoryOS's get_response prompt construction.
        """
        from utils import get_embedding, normalize_vector
        import numpy as np

        query_vec = get_embedding(
            query,
            model_name=self.memoryos.embedding_model_name,
            **self.memoryos.embedding_model_kwargs,
        )
        query_vec = normalize_vector(np.asarray(query_vec, dtype=np.float32))

        # --- MTM scoring via stock retriever ---------------------------------
        # retrieval_queue_capacity = knn*3 was set at __init__, so MTM returns
        # plenty of candidates and we re-rank with our own dot product.
        mtm_results = self.memoryos.retriever.retrieve_context(
            user_query=query,
            user_id=self.memoryos.user_id,
        )
        mtm_pages = mtm_results.get("retrieved_pages", []) or []
        mtm_candidates = []
        for page in mtm_pages:
            pe = page.get("page_embedding")
            ar = page.get("agent_response", "")
            if not ar:
                continue
            if pe is not None:
                pe_vec = normalize_vector(np.asarray(pe, dtype=np.float32))
                score = float(np.dot(query_vec, pe_vec))
            else:
                score = 0.0
            mtm_candidates.append((score, ar, page.get("user_input", "")))

        if self.stm_mode == "history":
            # NATIVE behaviour: include all STM entries verbatim as a
            # "Recent History" block. Then MTM top-knn as retrieved memory.
            mtm_candidates.sort(key=lambda x: x[0], reverse=True)
            mtm_top = mtm_candidates[: max(1, self.knn)]

            history_parts = []
            for qa in self.memoryos.short_term_memory.get_all() or []:
                ui = qa.get("user_input", "")
                ar = qa.get("agent_response", "")
                if not ar:
                    continue
                history_parts.append(f"User: {ui}\nAssistant: {ar}")

            retrieved_parts = [ar for _, ar, _ in mtm_top]

            blocks = []
            if history_parts:
                blocks.append(
                    "Recent History:\n" + "\n".join(history_parts)
                )
            if retrieved_parts:
                blocks.append(
                    "Retrieved Memory:\n" + "\n".join(retrieved_parts)
                )
            if not blocks:
                return "\n"
            return "\n\n".join(blocks) + "\n"

        # --- SCORE mode (legacy): STM gets cosine-scored and competes with MTM
        stm_candidates = []
        for qa in self.memoryos.short_term_memory.get_all() or []:
            ui = qa.get("user_input", "")
            ar = qa.get("agent_response", "")
            if not ar:
                continue
            ui_vec = get_embedding(
                ui or ar,
                model_name=self.memoryos.embedding_model_name,
                **self.memoryos.embedding_model_kwargs,
            )
            ui_vec = normalize_vector(np.asarray(ui_vec, dtype=np.float32))
            score = float(np.dot(query_vec, ui_vec))
            stm_candidates.append((score, ar))

        mtm_only = [(s, ar) for s, ar, _ in mtm_candidates]
        all_candidates = stm_candidates + mtm_only
        if not all_candidates:
            return "\n"
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        top = all_candidates[: max(1, self.knn)]
        return "\n".join(ar for _, ar in top) + "\n"

    # -----------------------------------------------------------------------
    # Bulk inject: vectorised native-style merging for Phase 1
    # -----------------------------------------------------------------------
    def bulk_insert_injects(
        self,
        qadv_list: List[str],
        response_list: List[str],
        sim_threshold: float = 0.6,
    ):
        """Insert many (q_adv, response) pairs at once with native-style
        topic merging.

        For each pair:
          1. Compute page_embedding = embedding("User: q_adv Assistant: response")
          2. Find best matching existing MTM session by summary embedding
          3. If best similarity >= threshold: append page to that session
             (no new session, no LLM call)
          4. Else: collect for batched new-session creation

        For new sessions, batch-call the LLM (if installed) to generate
        topic summaries, then create one session per orphan group.
        """
        import numpy as np
        import utils as _mos_utils
        from mid_term import compute_segment_heat
        from utils import get_timestamp, generate_id

        _ = _mos_utils.get_embedding(
            "warmup",
            model_name=self.memoryos.embedding_model_name,
            **self.memoryos.embedding_model_kwargs,
        )
        model_init_key = json.dumps(
            {
                "model_name": self.memoryos.embedding_model_name,
                **{
                    k: v
                    for k, v in self.memoryos.embedding_model_kwargs.items()
                    if k not in ("batch_size", "max_length")
                },
            },
            sort_keys=True,
        )
        model = _mos_utils._model_cache[model_init_key]

        N = len(qadv_list)
        if N == 0:
            return

        # --- 1. Batch compute page embeddings
        full_texts = [
            f"User: {q[:1500]} Assistant: {r[:1500]}"
            for q, r in zip(qadv_list, response_list)
        ]
        page_embeddings = model.encode(
            full_texts,
            batch_size=256,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        mtm = self.memoryos.mid_term_memory
        n_merged = 0
        orphans = []  # indices needing new sessions

        # --- 2. For each inject, find best matching session (vectorised)
        if mtm.sessions:
            sid_list = list(mtm.sessions.keys())
            sess_emb_matrix = np.array(
                [mtm.sessions[sid]["summary_embedding"] for sid in sid_list],
                dtype=np.float32,
            )
            # (N, dim) @ (dim, M) -> (N, M)
            sims_matrix = page_embeddings @ sess_emb_matrix.T
            best_sims = sims_matrix.max(axis=1)
            best_sids = [sid_list[i] for i in sims_matrix.argmax(axis=1)]
        else:
            best_sims = np.zeros(N)
            best_sids = [None] * N

        # --- 3. Merge into existing or mark as orphan
        for i in range(N):
            if best_sids[i] is not None and best_sims[i] >= sim_threshold:
                page = {
                    "page_id": f"inject_{i}_{generate_id('p')}",
                    "user_input": qadv_list[i][:1500],
                    "agent_response": response_list[i][:1500],
                    "page_embedding": page_embeddings[i].tolist(),
                    "page_keywords": _simple_keywords(qadv_list[i]),
                    "preloaded": False,
                    "analyzed": True,
                    "timestamp": "inject",
                    "pre_page": None,
                    "next_page": None,
                    "meta_info": qadv_list[i][:128],
                }
                mtm.sessions[best_sids[i]]["details"].append(page)
                n_merged += 1
            else:
                orphans.append(i)

        # --- 4. For orphans, batch-generate summaries (if LLM installed)
        #     then create one session per orphan (no further merging among
        #     orphans to keep this simple; the user-visible behaviour is
        #     "rare-topic injects each get their own segment").
        if orphans:
            import heapq
            print(
                f"MemoryOSAdapter: {n_merged} injects merged into existing "
                f"sessions, {len(orphans)} orphans -> new sessions."
            )
            if _LLM_BATCH_CALLABLE is not None:
                try:
                    from prompts import (
                        MULTI_SUMMARY_SYSTEM_PROMPT,
                        MULTI_SUMMARY_USER_PROMPT,
                    )
                except ImportError:
                    import prompts as _p
                    MULTI_SUMMARY_SYSTEM_PROMPT = _p.MULTI_SUMMARY_SYSTEM_PROMPT
                    MULTI_SUMMARY_USER_PROMPT = _p.MULTI_SUMMARY_USER_PROMPT
                messages_list = [
                    [
                        {"role": "system", "content": MULTI_SUMMARY_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": MULTI_SUMMARY_USER_PROMPT.format(
                                text=full_texts[i]
                            ),
                        },
                    ]
                    for i in orphans
                ]
                raw_summaries = _LLM_BATCH_CALLABLE(messages_list, 256)
            else:
                raw_summaries = [full_texts[i][:512] for i in orphans]

            # Parse summaries, embed them, create sessions
            orphan_summary_texts = []
            orphan_summary_keywords = []
            for i, raw in zip(orphans, raw_summaries):
                try:
                    m = re.search(r"\[\s*\{.*?\}\s*\]", raw, re.DOTALL)
                    parsed = json.loads(m.group(0)) if m else json.loads(raw)
                    if parsed and isinstance(parsed, list):
                        content = parsed[0].get("content", "")[:512] or full_texts[i][:512]
                        keywords = (
                            parsed[0].get("keywords")
                            or _simple_keywords(content)
                        )
                    else:
                        content, keywords = full_texts[i][:512], _simple_keywords(
                            qadv_list[i]
                        )
                except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
                    content, keywords = full_texts[i][:512], _simple_keywords(
                        qadv_list[i]
                    )
                orphan_summary_texts.append(content)
                orphan_summary_keywords.append(keywords)

            orphan_summary_embs = model.encode(
                orphan_summary_texts,
                batch_size=128,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            for k, i in enumerate(orphans):
                page = {
                    "page_id": f"inject_orphan_{i}",
                    "user_input": qadv_list[i][:1500],
                    "agent_response": response_list[i][:1500],
                    "page_embedding": page_embeddings[i].tolist(),
                    "page_keywords": _simple_keywords(qadv_list[i]),
                    "preloaded": False,
                    "analyzed": True,
                    "timestamp": "inject",
                    "pre_page": None,
                    "next_page": None,
                    "meta_info": qadv_list[i][:128],
                }
                ts = get_timestamp()
                sid = generate_id("session")
                session_obj = {
                    "id": sid,
                    "summary": orphan_summary_texts[k],
                    "summary_keywords": orphan_summary_keywords[k],
                    "summary_embedding": orphan_summary_embs[k].tolist(),
                    "details": [page],
                    "L_interaction": 1,
                    "R_recency": 1.0,
                    "N_visit": 0,
                    "H_segment": 0.0,
                    "timestamp": ts,
                    "last_visit_time": ts,
                    "access_count_lfu": 0,
                }
                session_obj["H_segment"] = compute_segment_heat(session_obj)
                mtm.sessions[sid] = session_obj
                mtm.access_frequency[sid] = 0
                heapq.heappush(mtm.heap, (-session_obj["H_segment"], sid))
        print(
            f"MemoryOSAdapter: bulk-inject complete. "
            f"{n_merged} merged, {len(orphans)} new-session orphans."
        )

    # -----------------------------------------------------------------------
    # STM control: replace contents with benign filler pairs
    # -----------------------------------------------------------------------
    def replace_stm_with_benign(self, n: int = 10, seed: int = 42):
        """Clear STM, refill with n random benign QA pairs from the base
        corpus. Simulates the "victim returns to agent after some unrelated
        benign interactions" scenario, where the most-recent injections
        have already aged out of STM.
        """
        import random
        with open(self.base_db_path) as f:
            db = json.load(f)
        passages = list(db.values())
        if not passages:
            return
        rng = random.Random(seed)
        sample = rng.sample(passages, min(n, len(passages)))

        stm = self.memoryos.short_term_memory
        # Clear current STM
        try:
            stm.qa_list = []  # default underlying storage in MemoryOS
        except AttributeError:
            pass
        if hasattr(stm, "memory"):
            try:
                stm.memory.clear()
            except Exception:
                pass

        for entry in sample:
            text = entry.get("content", "")[:1500]
            if not text:
                continue
            qa_pair = {
                "user_input": f"Tell me about: {text[:80]}",
                "agent_response": text,
                "timestamp": "benign_filler",
            }
            stm.add_qa_pair(qa_pair)
        print(
            f"MemoryOSAdapter: STM replaced with {len(sample)} benign filler "
            f"pairs (seed={seed})."
        )

    def update(self, response: str, question: Optional[str] = None) -> None:
        """Store a new interaction.

        QADataBase semantics:
            answer_based key   -> embedding(response), value=response
            question_based key -> embedding(question), value=response
        MemoryOS embeds the (user_input, agent_response) pair holistically,
        so the closest equivalent is:
            answer_based:   user_input=response[:200], agent_response=response
            question_based: user_input=question,       agent_response=response
        """
        if question is not None:
            user_input = question
        else:
            user_input = response[:200] if response else ""
        self.memoryos.add_memory(
            user_input=user_input,
            agent_response=response,
        )

    def reset(self) -> None:
        """Wipe all memory data on disk and reinitialise.

        Useful between attack strategies. Not called by the driver
        directly; the driver instead constructs a fresh adapter per
        strategy with wipe_existing=True.
        """
        if os.path.isdir(self.data_storage_path):
            shutil.rmtree(self.data_storage_path)
        self.__init__(
            base_db_path=self.base_db_path,
            data_storage_path=self.data_storage_path,
            embedding_model_name=self.memoryos.embedding_model_name,
            user_id=self.memoryos.user_id,
            assistant_id=self.memoryos.assistant_id,
            knn=self.knn,
            preload=True,
            wipe_existing=False,  # already wiped above
        )
