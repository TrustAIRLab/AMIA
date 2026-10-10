"""WebShop environment + RAP retrieval for the RAP memory-injection attack.

This module provides the pieces the attack runner (rap_gpt.py) reuses:
  - `webshopEnv`: a WebShop action interface (reset / search / click / think)
    that scrapes pages from a local mock WebShop server (mock_webshop.py) over
    HTTP and parses them into the text observations the agent reads.
  - `generate_embeddings` / `generate_examples`: RAP's memory retrieval — it
    embeds each stored trajectory's instruction (and optionally its actions) and
    returns the most similar past trajectories as in-context examples.
  - `initial_prompt`: the RAP few-shot WebShop prompt (from rap_assets/).

The vendored WebShop assets live under `rap_assets/`; the WebShop product data
itself is NOT shipped (see DATA_PREPARATION.md).
"""
import json
import os
from pathlib import Path
from urllib.parse import quote as _url_quote

import numpy as np
import torch
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from sentence_transformers.util import cos_sim

# ─── Paths / config ─────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
ASSETS_DIR = _HERE / "rap_assets"
WEBSHOP_INSTR_FILE = str(ASSETS_DIR / "webshop_instructions.json")
VICTIM_TARGET_FILE = str(ASSETS_DIR / "victim_target_pair" / "victim_target.json")
WEBSHOP_PROMPT_FILE = str(ASSETS_DIR / "prompts" / "webshop_prompt.py")
# Scratch file the mock server reads to resolve a session -> instruction.
MERGED_INSTR_FILE = str(_HERE / "results" / "merged_instructions.json")

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
WEBSHOP_URL = os.environ.get("WEBSHOP_URL", "http://localhost:3000")

# RAP retrieval config (mirrors the RAP base_config.yaml values).
RAP_CONFIG = {
    "num_retrieval": 3,
    "analogy_len": 10,
    "act_obs": True,
    "intra_task": True,
    "success": True,
    "max_init_prompt_len": 12800,
}

ACTION_TO_TEMPLATE = {
    "Description": "description_page.html",
    "Features": "features_page.html",
    "Reviews": "review_page.html",
    "Attributes": "attributes_page.html",
}


# ─── WebShop page scraping ────────────────────────────────────────────────────
def clean_str(p: str) -> str:
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element) -> bool:
    ignore = {"style", "script", "head", "title", "meta", "[document]"}
    return element.parent.name not in ignore and not isinstance(element, Comment)


def webshop_text(session, page_type, query_string="", page_num=1, asin="",
                 options={}, subpage="", merged_instructions_path=MERGED_INSTR_FILE,
                 **kwargs):
    """Fetch a WebShop page from the mock server and parse it into observation
    text plus structured info (clickable options, product ASINs, score)."""
    if page_type == "init":
        url = f"{WEBSHOP_URL}/{session}"
    elif page_type == "search":
        url = f"{WEBSHOP_URL}/search_results/{session}/{_url_quote(query_string)}/{page_num}"
    elif page_type == "item":
        url = f"{WEBSHOP_URL}/item_page/{session}/{asin}/{_url_quote(query_string) if query_string else ''}/{page_num}/{options}"
    elif page_type == "item_sub":
        url = f"{WEBSHOP_URL}/item_sub_page/{session}/{asin}/{_url_quote(query_string) if query_string else ''}/{page_num}/{subpage}/{options}"
    elif page_type == "end":
        url = f"{WEBSHOP_URL}/done/{session}/{asin}/{options}"
    else:
        url = f"{WEBSHOP_URL}/{session}"

    try:
        html = requests.get(url, timeout=30).text
    except Exception as e:
        return f"Error fetching {url}: {e}", {}

    html_obj = BeautifulSoup(html, "html.parser")
    visible_texts = list(filter(tag_visible, html_obj.find_all(string=True)))

    observation = ""
    option_type = ""
    options_dict = {}
    asins = []
    cnt = 0
    prod_cnt = 0
    just_prod = 0
    for t in visible_texts:
        if t == "\n":
            continue
        if t.replace("\n", "").replace("\\n", "").replace(" ", "") == "":
            continue
        if t.parent.name == "button":
            processed_t = f"\n[{t}] "
        elif t.parent.name == "label":
            processed_t = f"[[{t}]]" if f"'{t}'" in url else f"[{t}]"
            options_dict[str(t)] = option_type
        elif t.parent.get("class") == ["product-link"]:
            processed_t = f"\n[{t}] "
            if prod_cnt >= 3:
                processed_t = ""
            prod_cnt += 1
            asins.append(str(t))
            just_prod = 0
        else:
            processed_t = "\n" + str(t) + " "
            if cnt < 2 and page_type != "init":
                processed_t = ""
            if just_prod <= 2 and prod_cnt >= 4:
                processed_t = ""
            option_type = str(t)
            cnt += 1
        just_prod += 1
        observation += processed_t

    info = {}
    if options_dict:
        info["option_types"] = options_dict
    if asins:
        info["asins"] = asins
    score_texts = [str(t) for t in visible_texts if "Your score (min 0.0, max 1.0)" in str(t)]
    if score_texts:
        idx = next((i for i, t in enumerate(visible_texts)
                    if "Your score (min 0.0, max 1.0)" in str(t)), -1)
        score_val = 0.0
        if idx >= 0:
            for candidate in visible_texts[idx + 1:idx + 5]:
                try:
                    score_val = float(str(candidate).strip())
                    break
                except (ValueError, TypeError):
                    continue
        info["reward"] = score_val
        observation = f"Your score (min 0.0, max 1.0): {score_val}"

    instruction = None
    if os.path.exists(merged_instructions_path):
        try:
            data = json.loads(Path(merged_instructions_path).read_text(encoding="utf-8"))
            for entry in data:
                if entry.get("session") == session:
                    instruction = entry.get("instruction")
                    break
        except Exception:
            pass
    info["instruction"] = instruction

    try:
        observation = clean_str(observation)
    except Exception:
        pass
    if page_type == "init" and "Instruction: " in observation:
        observation = f"Webshop\nInstruction:\n{instruction}\n[Search]"
    return observation, info


class webshopEnv:
    """WebShop action interface over the mock server. Mirrors the RAP/WebShop
    reference environment: reset -> search[query] -> click[...] -> Buy Now."""

    def __init__(self, merged_instructions_path: str = MERGED_INSTR_FILE):
        self.sessions = {}
        self.merged_path = merged_instructions_path

    def step(self, session, action):
        done = False
        observation_ = None
        if action == "reset":
            self.sessions[session] = {"session": session, "page_type": "init"}
        elif action.startswith("think["):
            observation = "OK."
        elif action.startswith("search["):
            assert self.sessions[session]["page_type"] == "init"
            query = action[7:-1]
            self.sessions[session] = {"session": session, "page_type": "search",
                                      "query_string": query, "page_num": 1}
        elif action.startswith("click["):
            button = action[6:-1]
            if button == "Buy Now":
                assert self.sessions[session]["page_type"] == "item"
                if "options" in self.sessions[session]:
                    for opt_type in self.sessions[session]["options"]:
                        self.sessions[session]["options"][opt_type] = _url_quote(
                            self.sessions[session]["options"][opt_type])
                self.sessions[session]["page_type"] = "end"
                done = True
            elif button == "Back to Search":
                self.sessions[session] = {"session": session, "page_type": "init"}
            elif button == "Next >":
                assert self.sessions[session]["page_type"] == "search"
                self.sessions[session]["page_num"] += 1
            elif button == "< Prev":
                if self.sessions[session]["page_type"] == "item_sub":
                    self.sessions[session]["page_type"] = "item"
                elif self.sessions[session]["page_type"] == "item":
                    self.sessions[session]["page_type"] = "search"
                    self.sessions[session]["options"] = {}
            elif button in ACTION_TO_TEMPLATE:
                self.sessions[session]["page_type"] = "item_sub"
                self.sessions[session]["subpage"] = button
            else:
                if self.sessions[session]["page_type"] == "search":
                    if button in self.sessions[session].get("asins", []):
                        self.sessions[session]["page_type"] = "item"
                        self.sessions[session]["asin"] = button
                elif self.sessions[session]["page_type"] == "item":
                    if "option_types" in self.sessions[session] and \
                            button in self.sessions[session]["option_types"]:
                        opt_type = self.sessions[session]["option_types"][button]
                        self.sessions[session].setdefault("options", {})[opt_type] = button
                        observation_ = f"You have clicked {button}."
        else:
            raise AssertionError(f"Unknown action: {action}")

        observation, info = webshop_text(**self.sessions[session],
                                         merged_instructions_path=self.merged_path)
        if observation_:
            observation = observation_
        self.sessions[session].update(info)
        reward = info.get("reward", 0.0)
        return observation, reward, done, info


# ─── RAP memory retrieval ─────────────────────────────────────────────────────
def generate_embeddings(memory, model_embedding):
    """Embed the retrievable fields of each successful past trajectory."""
    memory = [m for m in memory if m["Reward"] > 0.0]
    if RAP_CONFIG.get("success", False):
        memory = [m for m in memory if m["Success"]]
    embeddings = {}
    for key in ["Instruction", "Reward", "Actions"]:
        if key == "Actions" and "Actions" in memory[0]:
            retrieve_info = [m[key][1:].copy() for m in memory]
            for i in range(len(retrieve_info)):
                for j in range(len(retrieve_info[i])):
                    retrieve_info[i][j] = retrieve_info[i][j].strip()
            embeddings[key] = [model_embedding.encode(r) for r in retrieve_info]
            continue
        retrieve_info = [m[key] for m in memory]
        if key == "Reward":
            embeddings[key] = retrieve_info
            continue
        embeddings[key] = model_embedding.encode(retrieve_info)
    return memory, embeddings


def generate_examples(info, actions, memory, embeddings, model_embedding,
                      reasoning="", k=3, act_len=0, use_act_obs=False):
    """Retrieve in-context examples from memory by instruction (and optionally
    action) similarity to the current task state."""
    cos_scores = None
    if info.get("instruction") is not None:
        instruction = info["instruction"]
        with torch.no_grad():
            instruction_embedding = model_embedding.encode([instruction])
        cos_scores = cos_sim(instruction_embedding, embeddings["Instruction"])[0]

    if len(actions) > 2 and (
        actions[-2].replace("Action: ", "").startswith("think")
        or actions[-2].replace("Action: ", "").startswith("search")
    ):
        reasoning = actions[-2].replace("Action: ", "")

    init_prompt = ""
    if cos_scores is not None:
        if act_len > 0 and reasoning != "" and "Actions" in embeddings:
            ret_scores, ret_index, intra_scores = [], [], []
            query_embedding = model_embedding.encode([reasoning])
            for a, emb in enumerate(embeddings["Actions"]):
                if use_act_obs:
                    if actions[-2].replace("Action: ", "").startswith("think"):
                        q_emb = model_embedding.encode([actions[-2].replace("Action: ", "")])
                        cs = cos_sim(q_emb, emb[::2]).numpy()
                        ret_scores.append(np.max(cs))
                        ret_index.append(np.argmax(cs) * 2)
                    else:
                        q_emb = model_embedding.encode([actions[-1].replace("Observation: ", "")])
                        cs = cos_sim(q_emb, emb[1::2]).numpy()
                        ret_scores.append(np.max(cs))
                        ret_index.append(np.argmax(cs) * 2 + 1)
                else:
                    cs = cos_sim(query_embedding, emb[::2]).numpy()
                    ret_scores.append(np.max(cs))
                    ret_index.append(np.argmax(cs) * 2)
                if RAP_CONFIG.get("intra_task", False):
                    intra_scores.append(
                        cos_sim(embeddings["Instruction"][a], emb[np.argmax(cs) * 2]).item())
            ret_scores = torch.FloatTensor(ret_scores)
            if RAP_CONFIG.get("intra_task", False):
                intra_scores = torch.FloatTensor(intra_scores)
                _, hits = torch.topk(ret_scores + cos_scores + intra_scores, k=k)
            else:
                _, hits = torch.topk(ret_scores + cos_scores, k=k)
            h_num = 0
            for h in hits:
                h_num += 1
                part = [max(1, ret_index[h] - act_len + 2),
                        min(len(memory[h]["Actions"]), ret_index[h] + act_len + 2)]
                retrieve_prompt = (
                    f"This is example{h_num}:\nInstruction: {memory[h]['Instruction']}\n"
                    f"These are actions and observations based on the task:\n"
                    f"{chr(10).join(memory[h]['Actions'][part[0]:part[1]])}\n\n")
                if len(init_prompt) + len(retrieve_prompt) > RAP_CONFIG["max_init_prompt_len"]:
                    break
                init_prompt += "\n" + retrieve_prompt
        else:
            _, hits = torch.topk(cos_scores, k=k)
            ret_examples = []
            h_num = 0
            for h in hits:
                h_num += 1
                ret_examples.append(
                    f"\nThis is example {h_num}:\nInstruction: {memory[h]['Instruction']}\n"
                    f"These are actions and observations based on the task:\n")
                ret_examples.append("\n".join(memory[h]["Actions"]))
                if len("\n".join(ret_examples)) > RAP_CONFIG["max_init_prompt_len"]:
                    ret_examples = ret_examples[:-2]
                    break
            init_prompt = "\n".join(ret_examples)
    return init_prompt, reasoning


# ─── RAP few-shot WebShop prompt (from vendored assets) ───────────────────────
with open(WEBSHOP_PROMPT_FILE) as _f:
    exec(_f.read())
initial_prompt = INITIAL_PROMPTS["PROMPT1"]  # type: ignore[name-defined]  # noqa: F821
