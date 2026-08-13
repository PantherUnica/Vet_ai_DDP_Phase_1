"""
Embedding-based domain affinity using Veterinary Master Taxonomy (semantic anchors).

Replaces hardcoded word lists with vector-to-domain similarity:
- DOMAIN_ANCHORS: high-density semantic descriptions per specialty (AVMA/RCVS-aligned).
- Candidate is domain-relevant when cosine_similarity(candidate_embedding, domain_embedding) >= threshold.
- Optional cold-start: load precomputed domain vectors from pickle at boot to avoid runtime embedding.

Scaling: add one anchor description per new specialty; no per-term lists. Optional dynamic discovery
via get_top_domain() for novel/unlinked entities (score < 0.75 => generic/novel).
"""

import math
import logging
import os
import threading
from typing import List, Optional, Any, Dict, Tuple

# ---------------------------------------------------------------------------
# VETERINARY MASTER SEMANTIC ANCHORS
# High-density descriptions for vector similarity; add one entry per specialty.
# Aliases (e.g. respiratory -> pulmonology) allow detect_domain() keys to match.
# ---------------------------------------------------------------------------
DOMAIN_ANCHORS: Dict[str, str] = {
    # --- ORGAN SYSTEMS ---
    "cardiology": (
        "Cardiovascular system, heart, and vasculature. Key concepts: murmurs, arrhythmias, "
        "congestive heart failure (CHF), echocardiography, ECG, hypertension, valve disease, "
        "cardiomyopathy (DCM/HCM), syncope, and cardiac biomarkers like NT-proBNP."
    ),
    "dermatology": (
        "Integumentary system, skin, ears, hair, and claws. Key concepts: pruritus (itching), "
        "alopecia, dermatitis, otitis externa, allergies, skin scraping, cytology, fungal "
        "infections, pyoderma, and parasites like Demodex or Sarcoptes."
    ),
    "neurology": (
        "Central and peripheral nervous systems, brain, and spinal cord. Key concepts: seizures, "
        "ataxia, paresis, paralysis, disc disease (IVDD), encephalitis, neuro-localization, "
        "proprioceptive deficits, head tilt, and MRI/CT imaging of the CNS."
    ),
    "ophthalmology": (
        "Ocular structures and vision. Key concepts: corneal ulcers, glaucoma, cataracts, "
        "uveitis, tonometry (IOP), Schirmer tear test, retinal detachment, blepharospasm, "
        "and conjunctivitis."
    ),
    "gastroenterology": (
        "Digestive tract from esophagus to colon. Key concepts: emesis (vomiting), diarrhea, "
        "IBD, pancreatitis, HGE, foreign body ingestion, endoscopy, malabsorption, "
        "and liver/gallbladder health (cholangitis)."
    ),
    "urology_nephrology": (
        "Urinary tract and kidneys. Key concepts: renal failure (CKD/AKI), urolithiasis (stones), "
        "cystitis, FLUTD, proteinuria, azotemia, urinalysis, and urinary incontinence."
    ),
    "pulmonology": (
        "Respiratory system and lungs. Key concepts: dyspnea, coughing, asthma, pneumonia, "
        "bronchoscopy, tracheal collapse, stertor, stridor, and pleural effusion."
    ),
    "endocrinology": (
        "Hormonal and metabolic disorders. Key concepts: Diabetes mellitus, Cushing's disease "
        "(hyperadrenocorticism), Addison's (hypoadrenocorticism), Hyperthyroidism, Hypothyroidism, "
        "insulin, glucose curves, and ACTH stimulation."
    ),
    "dentistry": (
        "Oral cavity and dental health. Key concepts: periodontal disease, tartar, gingivitis, "
        "extractions, dental prophylaxis, FORL, oral masses, and malocclusions."
    ),
    # --- CLINICAL SPECIALTIES ---
    "orthopedic": (
        "Musculoskeletal surgery and bone health. Key concepts: lameness, Ortolani test, "
        "cruciate ligament (TPLO/TTA), luxating patella, fractures, osteoarthritis, "
        "joint laxity, and bone grafting."
    ),
    "oncology": (
        "Neoplasia and cancer management. Key concepts: chemotherapy, staging, masses, "
        "biopsy, FNA, lymphoma, osteosarcoma, mast cell tumors, and paraneoplastic syndromes."
    ),
    "toxicology": (
        "Poisons, toxins, and envenomation. Key concepts: antidote, ingestion, chocolate, "
        "xylitol, rodenticide, ethylene glycol, lilies, gastric lavage, and activated charcoal."
    ),
    "theriogenology": (
        "Reproductive medicine. Key concepts: pregnancy, dystocia, pyometra, artificial "
        "insemination, heat cycles, neonatal care, and brucellosis."
    ),
    "emergency_critical_care": (
        "Acute trauma and life support. Key concepts: shock, triage, GDV, fluid resuscitation, "
        "transfusions, oxygen therapy, and sepsis."
    ),
    "internal_medicine": (
        "Complex multi-systemic metabolic and endocrine disorders. Clinical signs: PU/PD (polyuria/polydipsia), "
        "vomiting, diarrhea, weight loss, jaundice. Tests: CBC, Chemistry, Urinalysis, Bile acids. "
        "Conditions: Diabetes mellitus, Cushing's (hyperadrenocorticism), Renal failure, IBD, Pancreatitis."
    ),
    # --- ANCILLARY & DIAGNOSTIC ---
    "nutrition": (
        "Dietary management. Key concepts: obesity, prescription diets, calories, "
        "protein restriction, hypoallergenic food, and enteral feeding."
    ),
    "behavior": (
        "Mental and behavioral health. Key concepts: anxiety, aggression, phobias, "
        "compulsive behaviors, and psychotropic medications."
    ),
    "exotic_avian_medicine": (
        "Non-traditional species. Key concepts: reptiles, birds, small mammals, husbandry, "
        "metabolic bone disease in lizards, and feather plucking."
    ),
    "anesthesia_analgesia": (
        "Pain management and sedation. Key concepts: local blocks, intubation, opioids, "
        "NSAIDs, multi-modal analgesia, and sedative protocols."
    ),
    "radiology_imaging": (
        "Diagnostic imaging. Key concepts: X-ray, ultrasound, CT, MRI, contrast studies, "
        "and position/technique."
    ),
    # --- OPERATIONAL / AUDIT-AWARE (reduce general bucket, separate clinical from retail) ---
    "preventative_wellness": (
        "Preventive care. Key concepts: vaccines, vaccination, deworming, flea and tick control, "
        "heartworm prevention, wellness exams, and parasite prevention."
    ),
    "diagnostic_lab": (
        "Laboratory and pathology. Key concepts: blood work, CBC, chemistry panel, urinalysis, "
        "histopathology, cytology, culture, PCR, and in-house or external lab tests."
    ),
    "soft_tissue_surgery": (
        "Soft tissue and general surgery (non-orthopedic). Key concepts: spay, neuter, "
        "mass removal, laparotomy, wound repair, and abdominal or thoracic surgery."
    ),
    "administrative_retail": (
        "Non-clinical retail and administrative items. Key concepts: collars, leashes, "
        "pet boarding fees (non-medical), grooming services (bathing, trimming), accessories, "
        "and items that should not match clinical symptoms. Excludes: medical consultations, "
        "clinical boarding (inpatient care), medical procedures, certificates issued by vets."
    ),
    "operational_consumables": (
        "Operational and consumable supplies. Key concepts: general medical supplies, "
        "surgical supplies, lab supplies, mortuary, and consumables (low clinical-match priority)."
    ),
    "retail_pet_shop": (
        "Pet shop and non-clinical retail. Key concepts: toys, accessories, pet supplies, "
        "grooming products, and items with high phonetic noise for clinical search."
    ),
    # Aliases for detect_domain() / existing callers
    "respiratory": (
        "Respiratory system and lungs. Key concepts: dyspnea, coughing, asthma, pneumonia, "
        "bronchoscopy, tracheal collapse, stertor, stridor, and pleural effusion."
    ),
    "urology": (
        "Urinary tract and kidneys. Key concepts: renal failure (CKD/AKI), urolithiasis (stones), "
        "cystitis, FLUTD, proteinuria, azotemia, urinalysis, and urinary incontinence."
    ),
    "reproductive": (
        "Reproductive medicine. Key concepts: pregnancy, dystocia, pyometra, artificial "
        "insemination, heat cycles, neonatal care, and brucellosis."
    ),
}

# Backward compatibility: same as DOMAIN_ANCHORS
DOMAIN_DESCRIPTIONS: Dict[str, str] = DOMAIN_ANCHORS

_domain_embedding_cache: Dict[str, List[float]] = {}
_domain_cache_lock = threading.Lock()
_default_engine: Optional["VetDomainEngine"] = None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors. Pure Python; no numpy. Returns 0 if invalid."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _get_embedding_for_anchor(
    text: str,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[List[float]]:
    """Use project embedding API (OpenAI text-embedding-3-small)."""
    try:
        from kb_ner_embeddings import embed_text
        return embed_text(text.strip(), client=client, logger=logger)
    except Exception as e:
        if logger:
            logger.debug(f"Embedding failed: {e}")
        return None


def get_domain_embedding(
    domain_key: str,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[List[float]]:
    """
    Return cached embedding for a domain (e.g. 'orthopedic').
    Embeds DOMAIN_ANCHORS[domain_key] if not cached. Optionally loads from pickle if set.
    """
    desc = DOMAIN_ANCHORS.get(domain_key)
    if not desc:
        return None
    with _domain_cache_lock:
        if domain_key in _domain_embedding_cache:
            return _domain_embedding_cache[domain_key]
    # Optional: load precomputed vectors from pickle (cold-start)
    _try_load_precomputed_cache(logger)
    with _domain_cache_lock:
        if domain_key in _domain_embedding_cache:
            return _domain_embedding_cache[domain_key]
    # Embed on first use
    try:
        vec = _get_embedding_for_anchor(desc, client=client, logger=logger)
        if vec:
            with _domain_cache_lock:
                _domain_embedding_cache[domain_key] = vec
            return vec
    except Exception as e:
        if logger:
            logger.debug(f"Domain embedding failed for '{domain_key}': {e}")
    return None


def _try_load_precomputed_cache(logger: Optional[logging.Logger] = None) -> None:
    """Load domain vectors from pickle if KB_DOMAIN_VECTORS_PATH is set and file exists."""
    path = os.getenv("KB_DOMAIN_VECTORS_PATH", "").strip()
    if not path or not os.path.isfile(path):
        return
    try:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            with _domain_cache_lock:
                for k, v in data.items():
                    if isinstance(v, list) and v and isinstance(v[0], (int, float)):
                        _domain_embedding_cache[k] = [float(x) for x in v]
            if logger:
                logger.debug(f"Loaded {len(data)} domain vectors from {path}")
    except Exception as e:
        if logger:
            logger.debug(f"Precomputed domain vectors load failed: {e}")


def candidate_domain_affinity(
    candidate_embedding: Optional[List[float]],
    domain_key: str,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Affinity of a candidate (by its embedding) to a domain. Returns 0 if missing inputs."""
    if not candidate_embedding:
        return 0.0
    domain_vec = get_domain_embedding(domain_key, client=client, logger=logger)
    if not domain_vec:
        return 0.0
    return cosine_similarity(candidate_embedding, domain_vec)


def is_candidate_domain_relevant(
    candidate_name: str,
    domain_key: str,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    threshold: float = 0.82,
) -> bool:
    """True if embedding of candidate_name is sufficiently similar to domain (semantic anchor)."""
    if not (candidate_name and candidate_name.strip()) or not client:
        return False
    try:
        from kb_ner_embeddings import embed_text
        vec = embed_text(candidate_name.strip(), client=client, logger=logger)
        if not vec:
            return False
        aff = candidate_domain_affinity(vec, domain_key, client=client, logger=logger)
        return aff >= threshold
    except Exception as e:
        if logger:
            logger.debug(f"Domain affinity check failed for candidate '{candidate_name[:50]}': {e}")
        return False


def is_session_domain_relevant(
    session_text: Optional[str],
    domain_key: str,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    threshold: float = 0.75,
) -> bool:
    """True if session context (e.g. suspected_condition) is relevant to domain. Cross-entity reinforcement."""
    if not (session_text and session_text.strip()) or not client:
        return False
    try:
        from kb_ner_embeddings import embed_text
        vec = embed_text(session_text.strip()[:500], client=client, logger=logger)
        if not vec:
            return False
        aff = candidate_domain_affinity(vec, domain_key, client=client, logger=logger)
        return aff >= threshold
    except Exception as e:
        if logger:
            logger.debug(f"Session domain check failed: {e}")
        return False


def is_domain_relevant_for_phonetic_threshold(
    candidate_name: str,
    suspected_condition: Optional[str],
    domain_key: str,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    candidate_threshold: float = 0.80,  # Orthopedic gate 0.80 for "ultralining" → Ortolani auto-confirm
    session_threshold: float = 0.75,
    candidate_threshold_with_session: float = 0.80,
) -> bool:
    """
    Production rule for "use lower phonetic link threshold (0.80) for this candidate":
    - Candidate is domain-relevant by embedding (affinity >= candidate_threshold), OR
    - Session is domain-relevant AND candidate affinity >= candidate_threshold_with_session.
    """
    if not client:
        return False
    try:
        from kb_ner_embeddings import embed_text
        c_vec = embed_text((candidate_name or "").strip(), client=client, logger=logger)
        if not c_vec:
            return False
        aff = candidate_domain_affinity(c_vec, domain_key, client=client, logger=logger)
        if aff >= candidate_threshold:
            return True
        if suspected_condition and (suspected_condition or "").strip():
            if not is_session_domain_relevant(suspected_condition, domain_key, client=client, logger=logger, threshold=session_threshold):
                return False
            return aff >= candidate_threshold_with_session
        return False
    except Exception as e:
        if logger:
            logger.debug(f"Domain relevance for phonetic threshold failed: {e}")
        return False


def get_top_domain(
    term: str,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    min_score: float = 0.75,
) -> Tuple[Optional[str], float]:
    """
    Categorize a term into its most likely veterinary specialty (vector-based).
    Returns (domain_key, score). If best score < min_score, returns (None, score) for "generic/novel".
    Use for dynamic discovery: unlinked entities with low similarity can be flagged or sent to LLM for new anchor generation.
    """
    if not (term and term.strip()) or not client:
        return (None, 0.0)
    try:
        from kb_ner_embeddings import embed_text
        term_vec = embed_text(term.strip(), client=client, logger=logger)
        if not term_vec:
            return (None, 0.0)
        best_domain: Optional[str] = None
        best_score = 0.0
        for domain_key in DOMAIN_ANCHORS:
            domain_vec = get_domain_embedding(domain_key, client=client, logger=logger)
            if not domain_vec:
                continue
            score = cosine_similarity(term_vec, domain_vec)
            if score > best_score:
                best_score = score
                best_domain = domain_key
        if best_score < min_score:
            return (None, best_score)
        return (best_domain, best_score)
    except Exception as e:
        if logger:
            logger.debug(f"get_top_domain failed: {e}")
        return (None, 0.0)


# ---------------------------------------------------------------------------
# VetDomainEngine: class-based API for precomputed anchors and affinity
# ---------------------------------------------------------------------------

class VetDomainEngine:
    """
    Engine that pre-embeds DOMAIN_ANCHORS on init and exposes affinity scores.
    Use initialize_vectors() once (e.g. at app boot) so domain lookups are in-memory only.
    """

    def __init__(self, embedding_client: Optional[Any] = None):
        """
        embedding_client: optional; if None, uses kb_ner_embeddings.embed_text (resolves own client).
        """
        self._client = embedding_client
        self.domain_vectors: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def _embed(self, text: str, logger: Optional[logging.Logger] = None) -> Optional[List[float]]:
        if self._client and hasattr(self._client, "embeddings") and hasattr(self._client.embeddings, "create"):
            try:
                resp = self._client.embeddings.create(model="text-embedding-3-small", input=text.strip())
                if resp.data and len(resp.data) > 0:
                    return getattr(resp.data[0], "embedding", None)
            except Exception as e:
                if logger:
                    logger.debug(f"Engine embed failed: {e}")
        return _get_embedding_for_anchor(text, client=self._client, logger=logger)

    def initialize_vectors(
        self,
        client: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Pre-embed all DOMAIN_ANCHORS and store in domain_vectors (one-time boot)."""
        c = client or self._client
        for domain_key, description in DOMAIN_ANCHORS.items():
            if domain_key in self.domain_vectors:
                continue
            vec = _get_embedding_for_anchor(description, client=c, logger=logger)
            if vec:
                with self._lock:
                    self.domain_vectors[domain_key] = vec

    def get_affinity_score(
        self,
        term: str,
        domain: str,
        client: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> float:
        """Returns semantic similarity (0.0 to 1.0) between a term and a domain."""
        with self._lock:
            domain_vec = self.domain_vectors.get(domain)
        if not domain_vec:
            # Lazy-load this domain
            domain_vec = get_domain_embedding(domain, client=client or self._client, logger=logger)
            if domain_vec:
                with self._lock:
                    self.domain_vectors[domain] = domain_vec
        if not domain_vec:
            return 0.0
        term_vec = self._embed(term, logger=logger) or _get_embedding_for_anchor(term, client=client or self._client, logger=logger)
        if not term_vec:
            return 0.0
        return cosine_similarity(term_vec, domain_vec)

    def is_domain_relevant(
        self,
        name: str,
        domain: str,
        threshold: float = 0.82,
        client: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> bool:
        """Gatekeeper for domain-based phonetic threshold (e.g. 0.80 boost)."""
        return self.get_affinity_score(name, domain, client=client, logger=logger) >= threshold

    def get_top_domain(
        self,
        term: str,
        client: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        min_score: float = 0.75,
    ) -> Optional[str]:
        """Returns the domain with highest affinity, or None if below min_score (generic/novel)."""
        scores: Dict[str, float] = {}
        with self._lock:
            keys = list(self.domain_vectors.keys())
        if not keys:
            self.initialize_vectors(client=client, logger=logger)
            with self._lock:
                keys = list(self.domain_vectors.keys())
        term_vec = self._embed(term, logger=logger) or _get_embedding_for_anchor(term, client=client or self._client, logger=logger)
        if not term_vec:
            return None
        for d in keys:
            with self._lock:
                dv = self.domain_vectors.get(d)
            if dv:
                scores[d] = cosine_similarity(term_vec, dv)
        if not scores:
            return None
        top = max(scores, key=scores.get)
        return top if scores[top] >= min_score else None


def get_engine(embedding_client: Optional[Any] = None) -> VetDomainEngine:
    """Return a shared VetDomainEngine instance (lazy init)."""
    global _default_engine
    if _default_engine is None:
        _default_engine = VetDomainEngine(embedding_client=embedding_client)
    return _default_engine
