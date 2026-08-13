"""
Domain detection for NER/grounding: keyword-based (fast, no LLM).
Used by batch global soft-gate, domain boosting, and parallel prefetch.
Keys align with DOMAIN_ANCHORS in kb_domain_affinity for affinity/boost parity.
Kept in a separate module to avoid circular imports (e.g. kb_ner_global_search).
"""

from typing import Any, Dict, List, Optional

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "orthopedic": ["hip", "stifle", "lameness", "ortolani", "norberg", "cruciate", "tplo", "fho", "dysplasia", "patella", "luxat", "joint", "limping", "ortho", "femoral head", "ostectomy"],
    "dermatology": ["skin", "pruritus", "itch", "dermatitis", "otitis", "alopecia", "cytology", "scraping", "pyoderma"],
    "cardiology": ["heart", "murmur", "arrhythmia", "chf", "echo", "ecg", "cardiomyopathy"],
    "neurology": ["seizure", "ataxia", "paresis", "paralysis", "disc", "ivdd", "neuro"],
    "gastroenterology": ["vomit", "diarrhea", "ibd", "pancreatitis", "hge", "gi ", "abdomen"],
    "urology_nephrology": ["renal", "kidney", "ckd", "aki", "cystitis", "flutd", "urinalysis", "urolith"],
    "pulmonology": ["cough", "dyspnea", "pneumonia", "respiratory", "asthma"],
    "endocrinology": ["diabetes", "cushing", "addison", "thyroid", "insulin", "acth"],
    "dentistry": ["dental", "tooth", "periodontal", "gingivitis", "extraction", "tartar"],
    "oncology": ["cancer", "lymphoma", "mass", "biopsy", "fna", "chemotherapy"],
    "nutrition": ["obesity", "diet", "prescription diet", "renal diet", "weight"],
    "preventative_wellness": ["vaccine", "deworm", "flea", "tick", "heartworm", "wellness"],
}
DOMAIN_PRIMERS: Dict[str, List[str]] = {
    "orthopedic": ["Ortolani test", "Norberg angle", "lameness", "hip dysplasia", "Femoral Head and Neck Ostectomy", "TPLO", "cruciate"],
    "dermatology": ["otitis externa", "pruritus", "skin cytology"],
    "general": [],
}


def detect_domain(
    transcript: Optional[str],
    return_multiple: bool = False,
) -> Any:
    """
    Detect clinical domain(s) from transcript using keyword matching (first 1000 chars).
    Returns single domain string (or None) when return_multiple=False; returns list of domain strings when return_multiple=True.
    """
    if not (transcript or "").strip():
        return [] if return_multiple else None
    text = (transcript or "")[:1000].strip().lower()
    if not text:
        return [] if return_multiple else None
    found: List[str] = []
    for domain_key, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            found.append(domain_key)
    if return_multiple:
        return found if found else ["general"]
    return found[0] if found else "general"
