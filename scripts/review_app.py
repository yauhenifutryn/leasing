import json
import os
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from json import JSONDecodeError
from openai import OpenAI

load_dotenv()

KB_PATH = Path("knowledge_base/kb_faq_ru.json")
DEDUP_PATH = Path("insights_global/global_faq_clusters_dedup.json")
NLU_PATH = Path("nlu_output/nlu_pairs.jsonl")
CORR_PATH = Path("corrections/corrections.jsonl")
INSIGHTS_DIR = Path("insights_per_call")
OPENAI_MODEL = os.getenv("REVIEW_OPENAI_MODEL", "gpt-5.1")
OPENAI_READY = bool(os.getenv("OPENAI_API_KEY"))
MAX_REWRITES = 20
DETECTION_BATCH_SIZE = 8
ELEVEN_CONVAI_AGENT_ID = os.getenv("ELEVEN_CONVAI_AGENT_ID", "agent_6901kbht9aadfe69wts0nvpfdbst")
ELEVEN_WIDGET_ENABLED = os.getenv("ELEVEN_CONVAI_WIDGET", "1") not in {"0", "false", "False"}

_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI | None:
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    try:
        _openai_client = OpenAI()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Не удалось инициализировать OpenAI клиента: {exc}")
        return None
    return _openai_client


def call_openai(messages: List[Dict[str, str]], temperature: float | None = None) -> str | None:
    client = get_openai_client()
    if client is None:
        return None
    kwargs: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": messages,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    elif not OPENAI_MODEL.lower().startswith("gpt-5"):
        kwargs["temperature"] = 0
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def extract_json_structure(text: str) -> Any | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except JSONDecodeError:
            pass
    return None


def chunk_list(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def load_kb() -> List[Dict[str, Any]]:
    if not KB_PATH.exists():
        st.error(f"Файл {KB_PATH} не найден. Выполните `make kb` перед запуском приложения.")
        st.stop()
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def load_clusters() -> Dict[str, Dict[str, Any]]:
    if not DEDUP_PATH.exists():
        st.error(
            f"Файл {DEDUP_PATH} не найден. "
            "Запустите `make dedup` и `make kb` прежде чем использовать приложение."
        )
        st.stop()
    clusters = json.loads(DEDUP_PATH.read_text(encoding="utf-8"))
    return {cluster["canonical_q"]: cluster for cluster in clusters}


def load_nlu_rows() -> List[Dict[str, Any]]:
    if not NLU_PATH.exists():
        st.error(f"Файл {NLU_PATH} не найден. Выполните `make nlu-export` перед проверкой.")
        st.stop()
    rows: List[Dict[str, Any]] = []
    with open(NLU_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_kb(entries: List[Dict[str, Any]]) -> None:
    KB_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def save_clusters(cluster_map: Dict[str, Dict[str, Any]]) -> None:
    clusters = list(cluster_map.values())
    DEDUP_PATH.write_text(json.dumps(clusters, ensure_ascii=False, indent=2), encoding="utf-8")


def save_nlu_rows(rows: List[Dict[str, Any]]) -> None:
    with open(NLU_PATH, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def append_correction_log(record: Dict[str, Any]) -> None:
    CORR_PATH.parent.mkdir(parents=True, exist_ok=True)
    last_record = read_last_log_entry()
    if last_record == record:
        return
    with open(CORR_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


def list_audio_sources(conversation_ids: List[str]) -> List[str]:
    sources: List[str] = []
    for conv_id in conversation_ids:
        audio_fp = Path("audio") / f"{conv_id}.wav"
        if audio_fp.exists():
            sources.append(str(audio_fp))
        else:
            sources.append(f"(нет файла) {conv_id}")
    return sources


def read_last_log_entry() -> Dict[str, Any] | None:
    if not CORR_PATH.exists():
        return None
    try:
        with open(CORR_PATH, "r", encoding="utf-8") as handle:
            last_line = deque(handle, maxlen=1)[0].strip()
    except (IndexError, FileNotFoundError):
        return None
    if not last_line:
        return None
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return None


def load_corrections_for(question: str) -> List[Dict[str, Any]]:
    if not CORR_PATH.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with open(CORR_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("canonical_question") == question:
                entries.append(record)
    return entries


def find_last_correction(question: str) -> Dict[str, Any] | None:
    entries = load_corrections_for(question)
    for record in reversed(entries):
        if record.get("type") == "corrected":
            return record
    return None


def detect_inconsistencies(
    rows: List[Dict[str, Any]],
    canonical_answer: str,
    comment: str,
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {
        row["idx"]: {"needs_edit": False, "reason": "no_change_needed", "snippet": ""}
        for row in rows
    }
    if not rows or not OPENAI_READY:
        return results

    instruction = (
        "У тебя есть канонический ответ и комментарий ревизора. "
        "Для каждого ответа оператора найди фактические несоответствия (числа, проценты, валюты, НДС/НДФЛ и т.п.). "
        "Игнорируй стилистику. Верни JSON-массив элементов вида "
        "{\\\"id\\\": <id>, \\\"needs_edit\\\": true|false, \\\"reason\\\": \\\"...\\\", \\\"snippet\\\": \\\"...\\\"}."
    )

    for chunk in chunk_list(rows, DETECTION_BATCH_SIZE):
        payload = [
            {"id": row["idx"], "question": row["question"], "answer": row["answer"]}
            for row in chunk
        ]
        message = (
            f"{instruction}\n\n"
            f"Канонический ответ: {canonical_answer}\n"
            f"Комментарий ревизора: {comment}\n"
            f"Ответы операторов: {json.dumps(payload, ensure_ascii=False)}\n"
            "Ответь только JSON-массивом."
        )
        content = call_openai([{"role": "user", "content": message}])
        parsed = extract_json_structure(content)
        if isinstance(parsed, list):
            for item in parsed:
                idx = item.get("id")
                if idx is None or idx not in results:
                    continue
                results[idx] = {
                    "needs_edit": bool(item.get("needs_edit")),
                    "reason": item.get("reason", ""),
                    "snippet": item.get("snippet", ""),
                }
        else:
            for row in chunk:
                results[row["idx"]] = {
                    "needs_edit": False,
                    "reason": "parse_error",
                    "snippet": "",
                }
    return results


def rewrite_snippet(
    original_answer: str,
    snippet: str,
    canonical_answer: str,
    comment: str,
) -> tuple[str, bool, str]:
    if not snippet or not OPENAI_READY:
        return (original_answer, False, "no_snippet")

    instruction = (
        "Замени только указанный фрагмент ответа корректной формулировкой, не добавляя новых фактов. "
        "Верни JSON {\\\"replacement\\\": \\\"...\\\"}."
    )
    message = (
        f"{instruction}\n\n"
        f"Исходный ответ: {original_answer}\n"
        f"Фрагмент: {snippet}\n"
        f"Канонический ответ: {canonical_answer}\n"
        f"Комментарий ревизора: {comment}\n"
    )
    content = call_openai([{"role": "user", "content": message}])
    parsed = extract_json_structure(content)
    if not isinstance(parsed, dict):
        return (original_answer, False, "parse_error_rewrite")

    replacement = (parsed.get("replacement") or "").strip()
    if not replacement:
        return (original_answer, False, "empty_rewrite")

    if snippet in original_answer:
        return (original_answer.replace(snippet, replacement, 1), True, "fixed")

    return (original_answer, False, "snippet_not_found")


def detect_inconsistencies(
    rows: List[Dict[str, Any]],
    canonical_answer: str,
    comment: str,
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {
        row["idx"]: {"needs_edit": False, "reason": "no_change_needed", "snippet": ""}
        for row in rows
    }
    if not rows:
        return results
    instruction = (
        "У тебя есть канонический ответ и комментарий ревизора. "
        "Для каждого ответа оператора найди фактические противоречия (числа, проценты, валюты, НДС/НДФЛ и т.п.). "
        "Игнорируй стиль. Верни JSON-массив элементов вида {\"id\": ..., \"needs_edit\": ..., \"reason\": ..., \"snippet\": ...}."
    )
    for chunk in chunk_list(rows, DETECTION_BATCH_SIZE):
        payload = [
            {"id": row["idx"], "question": row["question"], "answer": row["answer"]}
            for row in chunk
        ]
        message = (
            f"{instruction}\n\n"
            f"Канонический ответ: {canonical_answer}\n"
            f"Комментарий ревизора: {comment}\n"
            f"Ответы операторов: {json.dumps(payload, ensure_ascii=False)}\n"
            "Ответь только JSON-массивом."
        )
        content = call_openai([{"role": "user", "content": message}])
        parsed = extract_json_structure(content)
        if isinstance(parsed, list):
            for item in parsed:
                idx = item.get("id")
                if idx is None or idx not in results:
                    continue
                results[idx] = {
                    "needs_edit": bool(item.get("needs_edit")),
                    "reason": item.get("reason", ""),
                    "snippet": item.get("snippet", ""),
                }
        else:
            for row in chunk:
                results[row["idx"]] = {
                    "needs_edit": False,
                    "reason": "parse_error",
                    "snippet": "",
                }
    return results



def update_insights_pair(call_id: str, pair_index: int, new_answer: str) -> bool:
    path = INSIGHTS_DIR / f"{call_id}.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    qa_pairs = data.get("verbatim_QA_pairs") or []
    idx = pair_index - 1
    if 0 <= idx < len(qa_pairs):
        qa_pairs[idx]["a"] = new_answer
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    return False


def regenerate_nlu_export() -> bool:
    try:
        subprocess.run(
            ["make", "nlu-export"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        st.error(f"Не удалось обновить nlu_output: {exc.stderr or exc.stdout}")
        return False


def undo_last_correction(
    canonical_question: str,
    kb_entries: List[Dict[str, Any]],
    cluster_map: Dict[str, Dict[str, Any]],
    nlu_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    record = find_last_correction(canonical_question)
    if not record:
        raise ValueError("Нет исправлений, которые можно отменить.")

    prev_answer = record.get("previous_kb_answer")
    if prev_answer is None:
        raise ValueError("Не сохранён предыдущий ответ для отката.")

    # Restore KB entry
    for entry in kb_entries:
        if entry["canonical_question"] == canonical_question:
            entry["best_answer"] = prev_answer
            entry["pending_review"] = True
            entry["last_reviewed_at"] = record.get("reviewed_at")
            entry["last_reviewer"] = record.get("reviewer")
            entry["review_comment"] = f"Откат к версии от {record.get('reviewed_at')}"
            break
    save_kb(kb_entries)

    if canonical_question in cluster_map:
        cluster_map[canonical_question]["best_answer"] = prev_answer
        save_clusters(cluster_map)

    prev_rows = record.get("previous_rows") or []
    note = f"Откат правки от {record.get('reviewed_at')} ({record.get('reviewer')})"
    for prev in prev_rows:
        call_id = prev.get("call_id")
        pair_index = prev.get("pair_index")
        prev_answer_text = prev.get("previous_answer", "")
        if not call_id or not pair_index:
            continue
        for row in nlu_rows:
            if row.get("call_id") == call_id and row.get("pair_index") == pair_index:
                row["answer"] = prev_answer_text
                row["needs_review"] = True
                row["review_notes"] = note
                update_insights_pair(call_id, pair_index, prev_answer_text)
                break
    if prev_rows:
        save_nlu_rows(nlu_rows)
        regenerate_nlu_export()

    undo_entry = {
        "canonical_question": canonical_question,
        "undone_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "undone_record": record.get("reviewed_at"),
        "type": "undo",
    }
    append_correction_log(undo_entry)
    return undo_entry


def find_candidate_rows(
    canonical_question: str,
    cluster_map: Dict[str, Dict[str, Any]],
    nlu_rows: List[Dict[str, Any]],
) -> List[int]:
    cluster = cluster_map.get(canonical_question)
    if not cluster:
        return []

    call_ids = set(cluster.get("source_conversation_ids", []))
    near_phrases = {
        canonical_question.lower(),
        *(phrase.lower() for phrase in cluster.get("near_duplicates", []) if phrase),
    }

    indices: List[int] = []
    for idx, row in enumerate(nlu_rows):
        include = False
        if row.get("call_id") in call_ids:
            include = True
        else:
            question = (row.get("question") or "").lower()
            if any(phrase and phrase in question for phrase in near_phrases):
                include = True
        if include:
            indices.append(idx)
    return indices


def format_candidate(row: Dict[str, Any]) -> str:
    question = (row.get("question") or "").replace("\n", " ")
    question = question[:90] + ("…" if len(question) > 90 else "")
    return f"{row.get('call_id')} · #{row.get('pair_index')} · {question}"


def format_candidate_details(row: Dict[str, Any]) -> str:
    question = row.get("question") or ""
    answer = row.get("answer") or ""
    return f"**Вопрос:** {question}\n\n**Текущий ответ:** {answer}"


def update_records(
    kb_entries: List[Dict[str, Any]],
    kb_index: int,
    new_answer: str,
    reviewer: str,
    comment: str,
    selected_row_indices: List[int],
    cluster_map: Dict[str, Dict[str, Any]],
    nlu_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    canonical_question = kb_entries[kb_index]["canonical_question"]
    previous_kb_answer = kb_entries[kb_index].get("best_answer", "")

    kb_entries[kb_index]["best_answer"] = new_answer.strip()
    kb_entries[kb_index]["last_reviewed_at"] = timestamp
    kb_entries[kb_index]["last_reviewer"] = reviewer
    kb_entries[kb_index]["review_comment"] = comment.strip()
    kb_entries[kb_index]["pending_review"] = False
    save_kb(kb_entries)

    if canonical_question in cluster_map:
        cluster_map[canonical_question]["best_answer"] = new_answer.strip()
        save_clusters(cluster_map)

    updated_rows = []
    previous_rows = []
    note = f"Исправлено {timestamp} ({reviewer}): {comment.strip()}"
    llm_used = False
    logs_for_user = []

    batch_rows = []
    for idx in selected_row_indices:
        row = nlu_rows[idx]
        batch_rows.append(
            {
                "idx": idx,
                "question": row.get("question", ""),
                "answer": row.get("answer") or "",
            }
        )

    detection_map = detect_inconsistencies(batch_rows, new_answer, comment)
    rewrites_done = 0
    for idx in selected_row_indices:
        row = nlu_rows[idx]
        if rewrites_done >= MAX_REWRITES:
            logs_for_user.append(
                {
                    "call_id": row.get("call_id"),
                    "pair_index": row.get("pair_index"),
                    "question": row.get("question"),
                    "old_answer": row.get("answer"),
                    "new_answer": row.get("answer"),
                    "changed": False,
                    "llm_used": False,
                    "reason": "limit_reached",
                }
            )
            continue
        original_answer = row.get("answer") or ""
        previous_rows.append(
            {
                "call_id": row.get("call_id"),
                "pair_index": row.get("pair_index"),
                "previous_answer": original_answer,
            }
        )

        detect_info = detection_map.get(idx, {})
        needs_edit = detect_info.get("needs_edit", False)
        llm_reason = detect_info.get("reason", "")
        snippet = detect_info.get("snippet", "")
        revised_answer = original_answer

        if needs_edit and snippet:
            if rewrites_done < MAX_REWRITES:
                revised_answer, changed_flag, rewrite_reason = rewrite_snippet(
                    original_answer,
                    snippet,
                    new_answer,
                    comment,
                )
                needs_edit = changed_flag
                llm_reason = rewrite_reason
                rewrites_done += 1 if changed_flag else 0
            else:
                llm_reason = "limit_reached"
                needs_edit = False
        elif needs_edit and not snippet:
            llm_reason = "no_snippet"
            needs_edit = False
        llm_used = llm_used or (needs_edit and revised_answer != original_answer)
        call_id = row.get("call_id")
        pair_index = row.get("pair_index")

        changed = needs_edit and revised_answer != original_answer

        if changed:
            row["answer"] = revised_answer
            row["needs_review"] = False
            row["review_notes"] = note
            if call_id and pair_index:
                update_insights_pair(call_id, pair_index, revised_answer)
            updated_rows.append(
                {
                    "call_id": call_id,
                    "pair_index": pair_index,
                }
            )
        llm_used = llm_used or changed

        logs_for_user.append(
            {
                "call_id": call_id,
                "pair_index": pair_index,
                "question": row.get("question"),
                "old_answer": original_answer,
                "new_answer": revised_answer,
                "changed": changed,
                "llm_used": changed,
                "reason": llm_reason,
            }
        )
    if updated_rows:
        save_nlu_rows(nlu_rows)
        regenerate_nlu_export()

    log_entry = {
        "canonical_question": canonical_question,
        "reviewed_at": timestamp,
        "reviewer": reviewer,
        "comment": comment.strip(),
        "corrected_answer": new_answer.strip(),
        "updated_rows": updated_rows,
        "type": "corrected",
        "nlu_regenerated": bool(updated_rows),
        "previous_kb_answer": previous_kb_answer,
        "previous_rows": previous_rows,
        "llm_used": llm_used,
        "row_diffs": logs_for_user,
    }
    append_correction_log(log_entry)
    return log_entry


def confirm_entry(
    kb_entries: List[Dict[str, Any]],
    kb_index: int,
    reviewer: str,
    comment: str,
) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = kb_entries[kb_index]
    entry["pending_review"] = False
    entry["last_reviewed_at"] = timestamp
    entry["last_reviewer"] = reviewer
    entry["review_comment"] = comment.strip()
    save_kb(kb_entries)

    log_entry = {
        "canonical_question": entry["canonical_question"],
        "reviewed_at": timestamp,
        "reviewer": reviewer,
        "comment": comment.strip(),
        "type": "confirmed",
    }
    append_correction_log(log_entry)
    return log_entry


def main() -> None:
    st.set_page_config(page_title="KB Accuracy Review", layout="wide")
    st.title("Проверка качества знаний")
    st.caption(
        "Оцените корректность ответов в `knowledge_base/kb_faq_ru.json`. "
        "Если ответ неверный, укажите корректную формулировку и комментарий — система обновит KB, "
        "источник в `insights_global` и связанные Q&A в `nlu_output/nlu_pairs.jsonl`."
    )

    if ELEVEN_WIDGET_ENABLED and ELEVEN_CONVAI_AGENT_ID:
        widget_js = """
<script>
(() => {{
  const AGENT_ID = "{agent}";
  const ensureWidget = () => {{
    if (!document.getElementById('convai-script')) {{
      const s = document.createElement('script');
      s.id = 'convai-script';
      s.src = 'https://unpkg.com/@elevenlabs/convai-widget-embed';
      s.async = true;
      s.type = 'text/javascript';
      document.body.appendChild(s);
    }}
    let wrapper = document.getElementById('convai-floating');
    if (!wrapper) {{
      wrapper = document.createElement('div');
      wrapper.id = 'convai-floating';
      wrapper.style.position = 'fixed';
      wrapper.style.bottom = '24px';
      wrapper.style.right = '24px';
      wrapper.style.zIndex = '999999';
      wrapper.style.width = '420px';
      wrapper.style.maxWidth = '95vw';
      wrapper.style.height = '640px';
      wrapper.style.pointerEvents = 'auto';
      document.body.appendChild(wrapper);
    }}
    wrapper.innerHTML = `<elevenlabs-convai agent-id="${{AGENT_ID}}" style="width:100%;height:100%;display:block;"></elevenlabs-convai>`;
  }};
  ensureWidget();
  window._convai_keepalive = window._convai_keepalive || setInterval(ensureWidget, 2000);
})();
</script>
""".format(agent=ELEVEN_CONVAI_AGENT_ID)
        st.markdown(widget_js, unsafe_allow_html=True)

    kb_entries = load_kb()
    cluster_map = load_clusters()
    nlu_rows = load_nlu_rows()

    options = [
        f"{idx + 1}. {entry['canonical_question']}"
        for idx, entry in enumerate(kb_entries)
    ]
    selected_idx = st.sidebar.selectbox(
        "Вопрос (canonical_question)",
        options=list(range(len(kb_entries))),
        format_func=lambda idx: options[idx],
    )
    entry = kb_entries[selected_idx]
    correction_history = load_corrections_for(entry["canonical_question"])

    with st.expander("Информация о записи", expanded=True):
        st.subheader(entry["canonical_question"])
        st.markdown("**Intent**: " + entry.get("intent", ""))
        status = "НЕ проверено" if entry.get("pending_review", True) else "Проверено"
        st.markdown(f"**Статус:** {status}")
        cluster = cluster_map.get(entry["canonical_question"])
        if cluster:
            st.markdown("**Связанные звонки (WAV):**")
            sources = list_audio_sources(cluster.get("source_conversation_ids", []))
            st.write(sources)
        st.markdown("**Текущий ответ:**")
        st.write(entry.get("best_answer", ""))
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Правила**")
            st.write(entry.get("eligibility_rules", []))
            st.markdown("**Обязательные поля**")
            st.write(entry.get("required_fields", []))
        with col_b:
            st.markdown("**Compliance**")
            st.write(entry.get("compliance_notes", []))
            st.markdown("**Эмпатия**")
            st.write(entry.get("empathy_patterns", []))
            st.markdown("**Follow-ups**")
            st.write(entry.get("followups", []))
        with st.expander("История правок", expanded=False):
            if correction_history:
                for record in reversed(correction_history[-5:]):
                    st.markdown(
                        f"- {record.get('reviewed_at')} — {record.get('reviewer','?')} "
                        f"({record.get('type','')}) — {record.get('comment','')}"
                    )
                if any(r.get("type") == "corrected" for r in correction_history):
                    if st.button("Отменить последнюю правку"):
                        try:
                            undo_entry = undo_last_correction(
                                entry["canonical_question"],
                                kb_entries,
                                cluster_map,
                                nlu_rows,
                            )
                            st.success("Последняя правка отменена.")
                            st.json(undo_entry)
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Не удалось отменить правку: {exc}")
                last_record = correction_history[-1]
                diffs = last_record.get("row_diffs") or []
                if diffs:
                    st.markdown("**Последняя правка (разница):**")
                    for diff in diffs:
                        with st.expander(
                            f"{diff.get('call_id')} · #{diff.get('pair_index')} "
                            f"{'(изм.)' if diff.get('changed') else '(без изменений)'}",
                            expanded=False,
                        ):
                            st.markdown(f"**Вопрос:** {diff.get('question')}")
                            st.markdown(f"**Старый ответ:** {diff.get('old_answer')}")
                            st.markdown(f"**Новый ответ:** {diff.get('new_answer')}")
            else:
                st.caption("История пока пуста.")

    reviewer = st.text_input("Ваше имя", value=st.session_state.get("reviewer", ""))
    if reviewer:
        st.session_state["reviewer"] = reviewer

    if not OPENAI_READY:
        st.warning(
            "OPENAI_API_KEY не найден в окружении. Ответы в связанных Q&A будут "
            "просто заменены на канонический текст без LLM-редактирования."
        )

    verdict = st.radio(
        "Ответ корректный?",
        options=["Да", "Нет"],
        horizontal=True,
        index=0,
    )

    if verdict == "Да":
        with st.form("confirm_form"):
            confirm_comment = st.text_area(
                "Комментарий (необязательно)",
                value=entry.get("review_comment", ""),
                height=120,
            )
            submitted = st.form_submit_button("Подтвердить корректность")
            if submitted:
                if not reviewer.strip():
                    st.error("Укажите имя ревизора.")
                    st.stop()
                try:
                    log_entry = confirm_entry(
                        kb_entries,
                        selected_idx,
                        reviewer,
                        confirm_comment,
                    )
                    st.success("Запись помечена как проверенная.")
                    st.json(log_entry)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Ошибка при подтверждении: {exc}")
        return

    with st.form("correction_form"):
        corrected_answer = st.text_area(
            "Корректный ответ",
            value=entry.get("best_answer", ""),
            height=200,
        )
        comment = st.text_area("Комментарий / что не так", height=120)
        submit_fix = st.form_submit_button("Сохранить исправление")

    candidate_indices = find_candidate_rows(
        entry["canonical_question"],
        cluster_map,
        nlu_rows,
    )
    if candidate_indices:
        st.markdown("### Связанные Q&A (будут обновлены автоматически)")
        for idx in candidate_indices:
            row = nlu_rows[idx]
            with st.expander(format_candidate(row), expanded=False):
                st.markdown(format_candidate_details(row))
    else:
        st.warning(
            "Не найдено связанных Q&A в nlu_output. Коррекция затронет только запись KB."
        )

    if submit_fix:
        if not reviewer.strip():
            st.error("Укажите имя ревизора.")
            st.stop()
        if not corrected_answer.strip():
            st.error("Введите корректный ответ.")
            st.stop()
        if not comment.strip():
            st.error("Добавьте комментарий.")
            st.stop()

        try:
            with st.spinner("Применяем исправление и обновляем связанные ответы..."):
                log_entry = update_records(
                    kb_entries,
                    selected_idx,
                    corrected_answer,
                    reviewer,
                    comment,
                    candidate_indices,
                    cluster_map,
                    nlu_rows,
                )
            st.success("Исправление сохранено.")
            st.json(log_entry)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ошибка при сохранении исправления: {exc}")


if __name__ == "__main__":
    main()
def list_audio_sources(conversation_ids: List[str]) -> List[str]:
    sources: List[str] = []
    for conv_id in conversation_ids:
        potential = Path("audio") / f"{conv_id}.wav"
        if potential.exists():
            sources.append(str(potential))
        else:
            sources.append(f"(нет файла) {conv_id}")
    return sources
