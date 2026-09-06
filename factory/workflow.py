"""Auditable factory decision support. Simulations are illustrative, never control machinery."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
import urllib.request
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def retrieve(query: str, extra_documents: list[dict] | None = None) -> list[dict]:
    """Return up to three positive TF-IDF cosine matches; documents are untrusted evidence."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    documents = []
    for path in sorted((ROOT / "data" / "manuals").glob("*.txt")):
        documents.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    documents.extend(extra_documents or [])
    # Chunk paragraphs for specific citations while retaining file provenance.
    chunks = [{"source": str(d.get("source", "uploaded document")), "text": part.strip()}
              for d in documents for part in re.split(r"\n\s*\n", str(d.get("text", "")))
              if part.strip()]
    if not chunks or not query.strip():
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([d["text"] for d in chunks])
        scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    except ValueError:
        return []
    return [dict(chunks[i], score=round(float(scores[i]), 5))
            for i in sorted(range(len(chunks)), key=lambda i: -scores[i])[:3]
            if scores[i] > 0]


def analyze_note(text: str) -> dict:
    """Rule baseline: negate symptoms in the same short clause; not a clinical NLP model."""
    symptoms = []
    negated = []
    patterns = {"vibration": r"vibrat\w*", "overheating": r"overheat\w*|hot|temperature",
                "noise": r"noise|noisy|grind\w*|rattl\w*", "leak": r"leak\w*",
                "wear": r"wear|worn", "crack": r"crack\w*"}
    for name, pattern in patterns.items():
        for match in re.finditer(r"\b(?:" + pattern + r")\b", text.lower()):
            prefix = re.split(r"[.;,!\n]|\bbut\b", text.lower()[:match.start()])[-1]
            suffix = text.lower()[match.end():match.end() + 25]
            is_negated = bool(re.search(r"\b(no|not|without|denies|absent)\b(?:\W+\w+){0,4}\W*$", prefix)
                              or re.match(r"\s+(?:is\s+)?(?:absent|not present)\b", suffix))
            target = negated if is_negated else symptoms
            if name not in target:
                target.append(name)
    urgent = bool(re.search(r"\b(urgent|critical|smoke|fire|seizure)\b", text.lower()))
    return {"method": "rule_based_baseline", "symptoms": symptoms, "negated_symptoms": negated,
            "urgency": "high" if urgent or len(symptoms) >= 3 else "medium" if symptoms else "low",
            "text": text, "limitations": "Keyword and local-negation rules; supervisor must verify context."}


def _probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probabilities must be finite numbers between 0 and 1.")
    return value


def simulate(probability, defect_probability, hours=8, rate=120, observed_reject_rate=.05) -> list[dict]:
    """Illustrative expected-cost twin using a hypothetical constant-hazard conversion.

    Model failure probability has a six-hour horizon; scenario probability is
    1-(1-p)**(hours/6). Failure loses four hours and costs 2400 currency units;
    lost output costs 4 per unit. Reduced load uses 70% throughput and a 0.55
    risk multiplier. Maintenance takes two hours, costs 450, and multiplies risk
    by 0.15. Reject rate stays constant across actions: no unsupported repair
    effect on quality is assumed. Individual image scores never determine yield.
    The default 5% reject rate is an assumption unless production data supplies it.
    """
    probability, defect_probability = _probability(probability), _probability(defect_probability)
    hours, rate = float(hours), float(rate)
    if not math.isfinite(hours) or not math.isfinite(rate) or hours <= 0 or rate <= 0:
        raise ValueError("Hours and rate must be positive finite values.")
    reject_rate = _probability(observed_reject_rate)
    horizon_probability = 1 - (1 - probability) ** (hours / 6)
    results = []
    for action, speed, risk, downtime, expense in [
        ("continue", 1, 1, 0, 0), ("reduce_load", .70, .55, 0, 0),
        ("maintenance", 1, .15, min(2, hours), 450)]:
        p = horizon_probability * risk
        failure_down = p * min(4, hours - downtime)
        total_down = downtime + failure_down
        units = (hours - total_down) * rate * speed * (1 - reject_rate)
        cost = expense + p * 2400 + (hours * rate - units) * 4
        results.append({"action": action, "expected_units": round(units, 1),
                        "downtime_hours": round(total_down, 2), "failure_probability": round(p, 4),
                        "expected_cost": round(cost, 2), "currency": "illustrative currency units",
                        "assumptions": {"illustrative_only": True, "hours": hours, "rate": rate,
                         "unit_margin": 4, "failure_repair_cost": 2400, "failure_lost_hours": 4,
                         "model_horizon_hours": 6, "input_failure_probability": probability,
                         "horizon_failure_probability": horizon_probability,
                         "horizon_conversion": "1-(1-p)**(hours/6), hypothetical constant hazard",
                         "speed_multiplier": speed, "risk_multiplier": risk,
                         "planned_downtime": downtime, "planned_cost": expense,
                         "observed_reject_rate": reject_rate,
                         "image_score_used_for_yield": False}})
    return results


def run_agents(maintenance: dict, vision: dict, note: str, extra_documents=None) -> dict:
    """Four deterministic cooperating agents exchange structured evidence, not autonomous actuators."""
    p = _probability(maintenance.get("failure_probability", maintenance.get("probability", 0)))
    d = _probability(vision.get("defect_probability", vision.get("probability", 0)))
    note_result = analyze_note(note)
    signals = [str(item.get("feature", "")).replace("_", " ")
               for item in maintenance.get("features", [])[:3] if isinstance(item, dict)]
    quality_review = d >= .5
    query = " ".join(["bearing maintenance inspection", *note_result["symptoms"], *signals,
                      "surface defect quality inspection" if quality_review else "product quality"])
    evidence = retrieve(query, extra_documents)
    reject_rate = maintenance.get("observed_reject_rate", .05)
    scenarios = simulate(p, d, observed_reject_rate=reject_rate)
    for scenario in scenarios:
        scenario["assumptions"]["reject_rate_source"] = (
            maintenance.get("reject_rate_source", "production_records" if "observed_reject_rate" in maintenance else "default assumption"))
    best = min(scenarios, key=lambda s: s["expected_cost"])
    recommendation = {"action": best["action"], "expected_cost": best["expected_cost"],
                      "reason": "Lowest expected cost among three illustrative scenarios; human approval required."
                      + (" Image classifier flagged this item: supervisor quality review is required before release."
                         if quality_review else ""),
                      "quality_review_required": quality_review,
                      "quality_evidence": {"individual_image_defect_probability": d,
                                           "observed_reject_rate": reject_rate,
                                           "image_score_is_batch_reject_rate": False},
                      "sources": sorted(set(e["source"] for e in evidence)),
                      "evidence_available": bool(evidence)}
    messages = [
        {"agent": "maintenance", "to": "planner", "payload": {"prediction": maintenance, "note_analysis": note_result}},
        {"agent": "vision", "to": "planner", "payload": vision},
        {"agent": "knowledge", "to": "planner", "payload": {"query": query, "evidence": evidence, "abstained": not evidence}},
        {"agent": "planner", "to": "supervisor", "payload": {"scenarios": scenarios, "recommendation": recommendation}},
    ]
    return {"incident_id": str(uuid4()), "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "pending", "maintenance": maintenance, "vision": vision,
            "note_analysis": note_result, "evidence": evidence, "scenarios": scenarios,
            "recommendation": recommendation, "messages": messages,
            "agent_mode": "deterministic structured orchestration"}


def save_decision(incident, decision, reason, modified_action=None, db_path="artifacts/decisions.sqlite") -> dict:
    """Append a human decision plus full incident snapshot to a local SQLite audit log."""
    if decision not in {"approve", "reject", "modify"}:
        raise ValueError("Decision must be approve, reject, or modify.")
    if decision == "modify" and (modified_action not in {"continue", "reduce_load", "maintenance"} or not reason.strip()):
        raise ValueError("A modification needs a valid action and a reason.")
    path = Path(db_path)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {"decision_id": str(uuid4()), "incident_id": incident["incident_id"],
              "timestamp": datetime.now(timezone.utc).isoformat(), "decision": decision,
              "reason": reason, "action": modified_action if decision == "modify" else
              incident["recommendation"]["action"] if decision == "approve" else None,
              "status": {"approve": "approved", "reject": "rejected", "modify": "modified"}[decision]}
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS decisions (decision_id TEXT PRIMARY KEY, incident_id TEXT, timestamp TEXT, decision TEXT, reason TEXT, action TEXT, incident_json TEXT)")
        db.execute("INSERT INTO decisions VALUES (?,?,?,?,?,?,?)", (result["decision_id"], result["incident_id"],
                   result["timestamp"], decision, reason, result["action"], json.dumps(incident)))
        db.commit()
    return result


def generate_explanation(incident, api_key='', model='gpt-4o-mini'):
    from factory.llm import generate_explanation as explain
    return explain(incident, api_key, model)
