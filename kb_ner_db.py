"""
Database connection and utility functions for KB NER linker.

This module handles:
- PostgreSQL connection management
- Database extension setup (pg_trgm, fuzzystrmatch)
- Learned aliases table management
- Entity mailbox operations for deferred learning
"""

import os
import logging
import threading
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict, Any

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logging.warning("psycopg2 not available. KB linking will be disabled.")

# Optional: thread-safe connection pool (recommended for high-concurrency workloads)
try:
    from psycopg2.pool import ThreadedConnectionPool  # type: ignore
    _POOL_AVAILABLE = True
except Exception:
    ThreadedConnectionPool = None  # type: ignore
    _POOL_AVAILABLE = False

# Connection cache for reuse
_connection_cache = None
_connection_lock = threading.Lock()

# Global pool (lazy-init)
_pg_pool = None
_pg_pool_lock = threading.Lock()


def _pg_conn_kwargs() -> Dict[str, Any]:
    """Build psycopg2 connect() kwargs from env vars."""
    host = os.getenv("PGHOST", "127.0.0.1")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE", "vetinstant")
    user = os.getenv("PGUSER", "vivek")
    password = os.getenv("PGPASSWORD", None)
    return {
        "host": host,
        "port": port,
        "dbname": dbname,
        "user": user,
        "password": password,
    }


def _pool_min_max() -> Tuple[int, int]:
    """
    Pool sizing:
    - KB_PG_POOL_MIN (default 1)
    - KB_PG_POOL_MAX (default 16)
    """
    try:
        minc = int(os.getenv("KB_PG_POOL_MIN", "1"))
    except Exception:
        minc = 1
    try:
        maxc = int(os.getenv("KB_PG_POOL_MAX", "16"))
    except Exception:
        maxc = 16
    if minc < 1:
        minc = 1
    if maxc < minc:
        maxc = minc
    return minc, maxc


def init_pg_pool(logger: Optional[logging.Logger] = None):
    """
    Initialize a thread-safe psycopg2 connection pool.
    Safe to call multiple times.
    """
    global _pg_pool
    if not PSYCOPG2_AVAILABLE or not _POOL_AVAILABLE:
        return None
    if os.getenv("KB_USE_PG_POOL", "true").lower() != "true":
        return None
    with _pg_pool_lock:
        if _pg_pool is not None:
            return _pg_pool
        minc, maxc = _pool_min_max()
        kwargs = _pg_conn_kwargs()

        # Allow the repo's default password fallback (vivek/bruno@9588) if PGPASSWORD unset
        if kwargs.get("user") == "vivek" and not kwargs.get("password"):
            kwargs = {**kwargs, "password": "bruno@9588"}

        try:
            _pg_pool = ThreadedConnectionPool(minc, maxc, **kwargs)  # type: ignore[misc]
            if logger:
                logger.info(f"✅ Initialized Postgres pool (min={minc}, max={maxc})")
        except Exception as e:
            _pg_pool = None
            if logger:
                logger.warning(f"⚠️  Failed to initialize Postgres pool: {e}")
        return _pg_pool


def acquire_pg_conn(logger: Optional[logging.Logger] = None):
    """
    Acquire a connection. Prefers the pool when enabled; falls back to direct connection.
    """
    pool = init_pg_pool(logger=logger)
    if pool is not None:
        try:
            conn = pool.getconn()
            conn.autocommit = True
            return conn
        except Exception as e:
            if logger:
                logger.warning(f"⚠️  Pool getconn failed, falling back to direct connect: {e}")
    return get_pg_conn(reuse=False)


def release_pg_conn(conn) -> None:
    """
    Return a connection to the pool (if it originated from the pool); otherwise close it.
    """
    global _pg_pool
    if conn is None:
        return
    try:
        if _pg_pool is not None:
            _pg_pool.putconn(conn)
            return
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass


@contextmanager
def pg_conn_ctx(logger: Optional[logging.Logger] = None):
    """
    Context manager around acquire/release.
    """
    conn = acquire_pg_conn(logger=logger)
    try:
        yield conn
    finally:
        release_pg_conn(conn)


def close_pg_pool(logger: Optional[logging.Logger] = None) -> None:
    """Close all pooled connections (optional cleanup for long-running apps)."""
    global _pg_pool
    with _pg_pool_lock:
        if _pg_pool is not None:
            try:
                _pg_pool.closeall()
                if logger:
                    logger.info("✅ Closed Postgres pool")
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️  Failed to close Postgres pool: {e}")
            _pg_pool = None


def get_pg_conn(reuse=True):
    """
    Connect to Postgres using env vars:
    PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
    
    Defaults match KB pipeline configuration:
    - PGHOST: 127.0.0.1 (or localhost)
    - PGPORT: 5432
    - PGDATABASE: vetinstant
    - PGUSER: vivek
    - PGPASSWORD: (from env var, no default)
    
    Falls back to sql_embeddings.connect_to_postgres if available.
    
    Args:
        reuse: If True, attempts to reuse cached connection if still valid (default: True)
    """
    global _connection_cache
    
    # Try to reuse cached connection if available and valid (thread-safe)
    if reuse:
        with _connection_lock:
            if _connection_cache is not None:
                try:
                    # Quick check if connection is still alive
                    _connection_cache.cursor().execute("SELECT 1")
                    return _connection_cache
                except (psycopg2.InterfaceError, psycopg2.OperationalError):
                    # Connection is dead, clear cache
                    _connection_cache = None
    
    if not PSYCOPG2_AVAILABLE:
        # Try to use sql_embeddings connection (optional fallback dependency)
        try:
            from sql_embeddings import connect_to_postgres  # type: ignore[import-untyped]
            conn = connect_to_postgres()
            if reuse:
                with _connection_lock:
                    _connection_cache = conn
            return conn
        except ImportError:
            raise RuntimeError(
                "Database connection unavailable: psycopg2 not installed. "
                "Install with: pip install psycopg2-binary (e.g. in your venv: voice_tenor_analysis/venv313)."
            )
        except RuntimeError:
            # connect_to_postgres() raises RuntimeError when psycopg2 is missing; propagate its message
            raise
    
    # Use same defaults as KB pipeline (from concept_embeddings.py)
    host = os.getenv("PGHOST", "127.0.0.1")
    port = os.getenv("PGPORT", "5432")
    dbname = os.getenv("PGDATABASE", "vetinstant")
    user = os.getenv("PGUSER", "vivek")
    password = os.getenv("PGPASSWORD", None)  # No default password for security
    
    # Try connection with provided credentials
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
        conn.autocommit = True
        if reuse:
            with _connection_lock:
                _connection_cache = conn
        return conn
    except psycopg2.OperationalError as e:
        # If connection fails, try with password from KB pipeline default
        # (only if user is vivek and password not provided)
        if user == "vivek" and password is None:
            try:
                conn = psycopg2.connect(
                    host=host,
                    port=port,
                    dbname=dbname,
                    user=user,
                    password="bruno@9588",  # KB pipeline default
                )
                conn.autocommit = True
                if reuse:
                    with _connection_lock:
                        _connection_cache = conn
                return conn
            except psycopg2.OperationalError:
                pass
        # Re-raise original error if fallback also fails
        raise e


def ensure_pg_trgm(conn, logger: Optional[logging.Logger] = None) -> bool:
    """
    Best-effort enablement of pg_trgm extension.
    Returns True if available, False otherwise.
    CRITICAL: Required for phonetic-first hybrid search (trigram similarity).
    """
    # First check if it's already enabled
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm');")
            if cur.fetchone()[0]:
                # Extension already enabled - no need to log (called frequently)
                return True
    except Exception as e:
        if logger:
            logger.debug(f"Could not check pg_trgm extension: {e}")
    
    # Try to create it
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        conn.commit()
        if logger:
            logger.info("✅ Created pg_trgm extension")
    except Exception as e:
        if logger:
            logger.warning(f"⚠️  Could not CREATE EXTENSION pg_trgm: {e}")
            logger.warning("   This may require superuser privileges. Contact your database administrator.")
    
    # Verify it's now available
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm');")
            is_available = cur.fetchone()[0]
            if is_available and logger:
                logger.info("✅ pg_trgm extension verified and ready")
            return is_available
    except Exception as e:
        if logger:
            logger.debug(f"Could not check pg_trgm extension: {e}")
    return False


def ensure_vector_extension(conn, logger: Optional[logging.Logger] = None) -> bool:
    """
    Best-effort enablement of pgvector extension.
    Returns True if available, False otherwise.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
            if cur.fetchone()[0]:
                if logger:
                    logger.debug("✅ vector extension already enabled")
                return True
    except Exception as e:
        if logger:
            logger.debug(f"Could not check vector extension: {e}")

    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        if logger:
            logger.info("✅ Created vector (pgvector) extension")
    except Exception as e:
        if logger:
            logger.warning(f"⚠️  Could not CREATE EXTENSION vector: {e}")
            logger.warning("   This may require superuser privileges. Contact your database administrator.")

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
            is_available = cur.fetchone()[0]
            if is_available and logger:
                logger.info("✅ vector extension verified and ready")
            return is_available
    except Exception as e:
        if logger:
            logger.debug(f"Could not verify vector extension: {e}")
    return False


from typing import Dict

# Cache for _column_exists to avoid repeated DB queries
_column_exists_cache: Dict[str, bool] = {}

def _column_exists(conn, schema: str, table: str, column: str) -> bool:
    cache_key = f"{schema}.{table}.{column}"
    if cache_key in _column_exists_cache:
        return _column_exists_cache[cache_key]
    
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = %s
                  AND column_name = %s
            );
            """,
            (schema, table, column),
        )
        result = bool(cur.fetchone()[0])
        _column_exists_cache[cache_key] = result
        return result


def ensure_hnsw_index(
    conn,
    *,
    schema: str,
    table: str,
    column: str,
    index_name: str,
    opclass: str = "vector_cosine_ops",
    m: int = 16,
    ef_construction: int = 64,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Best-effort create an HNSW index for pgvector columns (idempotent).
    Returns True if created or already exists; False if it could not be created.
    """
    try:
        if not ensure_vector_extension(conn, logger=logger):
            return False
        if not _column_exists(conn, schema, table, column):
            if logger:
                logger.debug(f"⚠️  Skipping HNSW index {index_name}: {schema}.{table}.{column} does not exist")
            return False

        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE schemaname = %s AND indexname = %s);",
                (schema, index_name),
            )
            if cur.fetchone()[0]:
                if logger:
                    logger.debug(f"✅ Index already exists: {schema}.{index_name}")
                return True

            cur.execute(
                f"""
                CREATE INDEX {index_name}
                ON {schema}.{table}
                USING hnsw ({column} {opclass})
                WITH (m = {int(m)}, ef_construction = {int(ef_construction)});
                """
            )
        conn.commit()
        if logger:
            logger.info(f"✅ Created HNSW index: {schema}.{index_name} on {schema}.{table}({column})")
        return True
    except Exception as e:
        if logger:
            logger.warning(f"⚠️  Could not create HNSW index {schema}.{index_name}: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def ensure_trgm_gin_index(
    conn,
    *,
    schema: str,
    table: str,
    column: str,
    index_name: str,
    use_lower: bool = True,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Best-effort create a pg_trgm GIN index (idempotent).
    Uses an expression index on lower(column) by default since most queries do lower(col).
    """
    try:
        if not ensure_pg_trgm(conn, logger=logger):
            return False
        if not _column_exists(conn, schema, table, column):
            if logger:
                logger.debug(f"⚠️  Skipping trigram index {index_name}: {schema}.{table}.{column} does not exist")
            return False

        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE schemaname = %s AND indexname = %s);",
                (schema, index_name),
            )
            if cur.fetchone()[0]:
                if logger:
                    logger.debug(f"✅ Index already exists: {schema}.{index_name}")
                return True

            expr = f"lower({column})" if use_lower else column
            cur.execute(
                f"""
                CREATE INDEX {index_name}
                ON {schema}.{table}
                USING gin ({expr} gin_trgm_ops);
                """
            )
        conn.commit()
        if logger:
            logger.info(f"✅ Created trigram GIN index: {schema}.{index_name} on {schema}.{table}({column})")
        return True
    except Exception as e:
        if logger:
            logger.warning(f"⚠️  Could not create trigram index {schema}.{index_name}: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def ensure_btree_index(
    conn,
    *,
    schema: str,
    table: str,
    column: str,
    index_name: str,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Best-effort create a B-tree index (idempotent).
    Used for kind_key, domain_key etc. to keep soft-gate and kind-filter queries fast.
    """
    try:
        if not _column_exists(conn, schema, table, column):
            if logger:
                logger.debug(f"⚠️  Skipping B-tree index {index_name}: {schema}.{table}.{column} does not exist")
            return False

        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE schemaname = %s AND indexname = %s);",
                (schema, index_name),
            )
            if cur.fetchone()[0]:
                if logger:
                    logger.debug(f"✅ Index already exists: {schema}.{index_name}")
                return True

            cur.execute(
                f'CREATE INDEX IF NOT EXISTS "{index_name}" ON {schema}.{table} ("{column}")'
            )
        conn.commit()
        if logger:
            logger.info(f"✅ Created B-tree index: {schema}.{index_name} on {schema}.{table}({column})")
        return True
    except Exception as e:
        if logger:
            logger.warning(f"⚠️  Could not create B-tree index {schema}.{index_name}: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def ensure_btree_composite_index(
    conn,
    *,
    schema: str,
    table: str,
    columns: List[str],
    index_name: str,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Best-effort create a B-tree composite index (idempotent).
    E.g. (kind, domain_key) so the DB can filter by kind then domain in one index.
    """
    try:
        for col in columns:
            if not _column_exists(conn, schema, table, col):
                if logger:
                    logger.debug(f"⚠️  Skipping composite index {index_name}: {schema}.{table}.{col} does not exist")
                return False

        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE schemaname = %s AND indexname = %s);",
                (schema, index_name),
            )
            if cur.fetchone()[0]:
                if logger:
                    logger.debug(f"✅ Index already exists: {schema}.{index_name}")
                return True

            cols_expr = ", ".join(f'"{c}"' for c in columns)
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS "{index_name}" ON {schema}.{table} ({cols_expr})'
            )
        conn.commit()
        if logger:
            logger.info(f"✅ Created composite B-tree index: {schema}.{index_name} on ({', '.join(columns)})")
        return True
    except Exception as e:
        if logger:
            logger.warning(f"⚠️  Could not create composite index {schema}.{index_name}: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def ensure_soft_gate_indexes(conn, logger: Optional[logging.Logger] = None) -> None:
    """
    Best-effort indexes for soft-gate and domain-aware search (idempotent).

    - kb.concepts: index on kind, domain_key, and composite (kind, domain_key) for route + domain.
    - soap.inventory / soap.service_master: domain_key index and composite (domain_key, category).

    Safe to run repeatedly. Call after ensure_soap_*_domain_column and ensure_kb_concepts_domain_column.
    """
    # kb.concepts: ensure domain_key column then indexes (our schema uses "kind" not "kind_key")
    ensure_kb_concepts_domain_column(conn, logger=logger)
    ensure_btree_index(
        conn, schema="kb", table="concepts", column="kind",
        index_name="idx_kb_concepts_kind", logger=logger,
    )
    ensure_btree_index(
        conn, schema="kb", table="concepts", column="domain_key",
        index_name="idx_kb_concepts_domain_key", logger=logger,
    )
    ensure_btree_composite_index(
        conn, schema="kb", table="concepts", columns=["kind", "domain_key"],
        index_name="idx_kb_concepts_kind_domain", logger=logger,
    )
    # soap.inventory / soap.service_master: domain_key (with DEFAULT 'general') then indexes
    ensure_soap_inventory_domain_column(conn, logger=logger)
    ensure_soap_service_master_domain_column(conn, logger=logger)
    ensure_btree_index(
        conn, schema="soap", table="inventory", column="domain_key",
        index_name="idx_soap_inventory_domain_key", logger=logger,
    )
    ensure_btree_index(
        conn, schema="soap", table="service_master", column="domain_key",
        index_name="idx_soap_service_master_domain_key", logger=logger,
    )
    ensure_btree_composite_index(
        conn, schema="soap", table="inventory", columns=["domain_key", "category"],
        index_name="idx_soap_inventory_domain_category", logger=logger,
    )
    ensure_btree_composite_index(
        conn, schema="soap", table="service_master", columns=["domain_key", "category"],
        index_name="idx_soap_service_master_domain_category", logger=logger,
    )


def ensure_kb_search_indexes(conn, logger: Optional[logging.Logger] = None) -> None:
    """
    Best-effort “speed path” index setup for both:
    - semantic search (HNSW on pgvector embeddings)
    - lexical fuzzy search (pg_trgm GIN on names / aliases)

    This function is safe to call repeatedly; it will no-op if indexes exist.
    """
    # Ensure core extensions
    ensure_vector_extension(conn, logger=logger)
    ensure_pg_trgm(conn, logger=logger)

    # Trigram indexes for the columns we actively query with similarity(lower(...), lower(%s))
    ensure_trgm_gin_index(
        conn, schema="kb", table="concepts", column="preferred_name",
        index_name="idx_kb_concepts_preferred_name_trgm", logger=logger
    )
    ensure_trgm_gin_index(
        conn, schema="kb", table="concept_aliases", column="alias_text",
        index_name="idx_kb_concept_aliases_alias_text_trgm", logger=logger
    )
    ensure_trgm_gin_index(
        conn, schema="soap", table="inventory", column="item_name",
        index_name="idx_soap_inventory_item_name_trgm", logger=logger
    )
    ensure_trgm_gin_index(
        conn, schema="soap", table="inventory", column="trade_name",
        index_name="idx_soap_inventory_trade_name_trgm", logger=logger
    )
    ensure_trgm_gin_index(
        conn, schema="soap", table="service_master", column="procedure_name",
        index_name="idx_soap_service_master_procedure_name_trgm", logger=logger
    )

    # HNSW indexes: create for every discovered pgvector column in kb + soap.
    # This makes setup “fully implemented” even if new vector columns are added later.
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_schema, table_name, column_name
                FROM information_schema.columns
                WHERE table_schema IN ('kb', 'soap')
                  AND udt_name = 'vector'
                ORDER BY table_schema, table_name, ordinal_position;
                """
            )
            vec_cols = cur.fetchall()

        for schema, table, column in vec_cols:
            index_name = f"idx_{schema}_{table}_{column}_hnsw"
            # Keep index names reasonably short (Postgres max is 63 bytes)
            if len(index_name) > 60:
                index_name = index_name[:60]
            ensure_hnsw_index(
                conn,
                schema=schema,
                table=table,
                column=column,
                index_name=index_name,
                logger=logger,
            )
    except Exception as e:
        if logger:
            logger.warning(f"⚠️  Failed to introspect vector columns for HNSW setup: {e}")

    # Refresh planner stats (best-effort)
    try:
        with conn.cursor() as cur:
            cur.execute("ANALYZE kb.concepts;")
            cur.execute("ANALYZE kb.concept_aliases;")
            cur.execute("ANALYZE kb.kb_text_embeddings;")
            cur.execute("ANALYZE soap.inventory;")
            cur.execute("ANALYZE soap.service_master;")
        conn.commit()
    except Exception:
        pass

def ensure_fuzzystrmatch(conn, logger: Optional[logging.Logger] = None) -> bool:
    """
    Best-effort enablement of fuzzystrmatch extension.
    Returns True if available, False otherwise.
    CRITICAL: Required for phonetic matching (metaphone function).
    """
    # First check if it's already enabled
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'fuzzystrmatch');")
            if cur.fetchone()[0]:
                # Extension already enabled - no need to log (called frequently)
                return True
    except Exception as e:
        if logger:
            logger.debug(f"Could not check fuzzystrmatch extension: {e}")

    # Try to create it
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;")
        conn.commit()
        if logger:
            logger.info("✅ Created fuzzystrmatch extension")
    except Exception as e:
        if logger:
            logger.warning(f"⚠️  Could not CREATE EXTENSION fuzzystrmatch: {e}")
            logger.warning("   This may require superuser privileges. Contact your database administrator.")

    # Verify it's now available
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'fuzzystrmatch');")
            is_available = cur.fetchone()[0]
            if is_available and logger:
                logger.info("✅ fuzzystrmatch extension verified and ready")
            return bool(is_available)
    except Exception as e:
        if logger:
            logger.debug(f"Could not check fuzzystrmatch extension: {e}")
    return False


# ==============================================================================
# KB: Vitals Registry (optional but recommended)
# ==============================================================================

# Seed list: canonical metric_name + definition + common synonyms/phrases.
# This is intentionally compact and clinic-agnostic; clinics can extend it later.
VITALS_REGISTRY_SEED: List[Dict[str, Any]] = [
    # Core TPR
    {
        "metric_name": "Temperature",
        "category": "TPR",
        "definition": "Body temperature measured during the exam (often rectal). Used to assess fever or hypothermia.",
        "synonyms": ["temp", "t", "temperature is normal", "febrile", "afebrile"],
        "expected_unit": "°F/°C",
    },
    {
        "metric_name": "Heart Rate",
        "category": "TPR",
        "definition": "Heart rate in beats per minute. Often recorded as HR or pulse rate.",
        "synonyms": ["hr", "pulse", "pulse rate", "bpm"],
        "expected_unit": "bpm",
    },
    {
        "metric_name": "Respiratory Rate",
        "category": "TPR",
        "definition": "Respiration rate in breaths per minute (RR). Used for triage and respiratory assessment.",
        "synonyms": ["rr", "resp rate", "respiratory rate", "breaths per minute"],
        "expected_unit": "breaths/min",
    },
    # Perfusion / circulation
    {
        "metric_name": "CRT",
        "category": "Perfusion",
        "definition": "Capillary refill time (CRT). Perfusion indicator (often measured at gums).",
        "synonyms": ["capillary refill time", "capillary refill", "crt 1.5 sec", "crt 2 sec"],
        "expected_unit": "sec",
    },
    {
        "metric_name": "Mucous Membranes",
        "category": "Perfusion",
        "definition": "Mucous membrane color/moisture (MM). Perfusion and hydration indicator.",
        "synonyms": ["mm", "mucous membrane", "mucous membranes pink", "mm pink", "pale mm", "tacky gums"],
        "expected_unit": "",
    },
    {
        "metric_name": "Pulse Quality",
        "category": "Perfusion",
        "definition": "Quality of peripheral pulse (strong/weak/thready/bounding) and synchrony with heartbeats.",
        "synonyms": ["pulse quality strong", "thready pulse", "bounding pulse", "weak pulse"],
        "expected_unit": "",
    },
    {
        "metric_name": "Hydration Status",
        "category": "Perfusion",
        "definition": "Hydration estimate (e.g., % dehydrated) based on skin turgor, mucous membranes, and other signs.",
        "synonyms": ["hydration", "skin turgor", "skin tent", "dehydrated", "tacky"],
        "expected_unit": "%/qualitative",
    },
    # Physical condition
    {
        "metric_name": "Weight",
        "category": "Physical",
        "definition": "Body weight measured during the visit.",
        "synonyms": ["wt", "weight is", "kg", "lbs", "kilos", "pounds"],
        "expected_unit": "kg/lb",
    },
    {
        "metric_name": "BCS",
        "category": "Physical",
        "definition": "Body condition score (BCS), typically on a 1–9 or 1–5 scale. Used to assess obesity/underweight.",
        "synonyms": ["body condition score", "bcs 6/9", "overweight", "obese"],
        "expected_unit": "score",
    },
    {
        "metric_name": "MCS",
        "category": "Physical",
        "definition": "Muscle condition score (MCS), qualitative assessment of muscle mass.",
        "synonyms": ["muscle condition score", "muscle wasting", "sarcopenia"],
        "expected_unit": "qualitative",
    },
    # Auscultation
    {
        "metric_name": "Heart Auscultation",
        "category": "Auscultation",
        "definition": "Heart sounds on auscultation (normal/abnormal), including murmur or arrhythmia notes.",
        "synonyms": ["heart sounds", "murmur", "arrhythmia", "no murmur", "murmur present"],
        "expected_unit": "qualitative",
    },
    {
        "metric_name": "Lung Auscultation",
        "category": "Auscultation",
        "definition": "Lung sounds on auscultation (clear/crackles/wheezes).",
        "synonyms": ["lungs clear", "lung sounds", "crackles", "wheezes", "increased bronchovesicular sounds"],
        "expected_unit": "qualitative",
    },
    # Neuro / pain
    {
        "metric_name": "Mentation",
        "category": "Neuro",
        "definition": "Level of consciousness / mentation (BAR/QAR/dull/obtunded).",
        "synonyms": ["bar", "qar", "bright alert responsive", "quiet alert responsive", "obtunded", "dull"],
        "expected_unit": "categorical",
    },
    {
        "metric_name": "Pain Score",
        "category": "Pain",
        "definition": "Pain score recorded using a clinic scale (e.g., 0–4) to guide analgesia and nursing care.",
        "synonyms": ["pain", "pain 2/4", "pain score 3", "painful"],
        "expected_unit": "score",
    },
    # Advanced / triage
    {
        "metric_name": "Blood Pressure",
        "category": "Advanced",
        "definition": "Blood pressure (systolic/diastolic/MAP).",
        "synonyms": ["bp", "blood pressure", "systolic", "diastolic", "map"],
        "expected_unit": "mmHg",
    },
    {
        "metric_name": "SpO2",
        "category": "Advanced",
        "definition": "Oxygen saturation measured via pulse oximetry (SpO2).",
        "synonyms": ["spo2", "oxygen saturation", "pulse ox"],
        "expected_unit": "%",
    },
    {
        "metric_name": "ETCO2",
        "category": "Advanced",
        "definition": "End-tidal CO2 (ETCO2), often used under anesthesia/sedation or respiratory monitoring.",
        "synonyms": ["etco2", "end tidal co2", "capnography"],
        "expected_unit": "mmHg",
    },
    {
        "metric_name": "Blood Glucose",
        "category": "Advanced",
        "definition": "Blood glucose (BG), often in emergency triage or diabetic monitoring.",
        "synonyms": ["bg", "blood glucose", "glucose"],
        "expected_unit": "mg/dL or mmol/L",
    },
]


def ensure_kb_concepts_vetbert_embedding_column(
    conn,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Add embedding_vetbert vector(768) to kb.concepts for dual-embedding (OpenAI + VetBERT) search.
    Creates HNSW index for fast VetBERT similarity search.
    Safe to call repeatedly.
    """
    try:
        ensure_vector_extension(conn, logger=logger)
        with conn.cursor() as cur:
            cur.execute(
                "ALTER TABLE kb.concepts ADD COLUMN IF NOT EXISTS embedding_vetbert vector(768)"
            )
        ensure_hnsw_index(
            conn,
            schema="kb",
            table="concepts",
            column="embedding_vetbert",
            index_name="idx_kb_concepts_embedding_vetbert_hnsw",
            logger=logger,
        )
        if logger:
            logger.info("✅ kb.concepts.embedding_vetbert column and HNSW index ready")
        return True
    except Exception as e:
        if logger:
            logger.warning("⚠️  Could not ensure kb.concepts.embedding_vetbert: %s", e)
        return False


def ensure_kb_concepts_domain_column(
    conn,
    column_name: str = "domain_key",
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Add a domain/specialty column to kb.concepts if it does not exist.

    Used for clinical domain tagging (e.g. orthopedic, ophthalmology) so that
    search/linking can apply domain-match scoring and avoid implausible matches
    (e.g. 'nebula' in an orthopedic note). Safe to call repeatedly.

    Args:
        conn: PostgreSQL connection (autocommit recommended).
        column_name: Name of the column (default: domain_key). Must be a valid
            identifier (alphanumeric and underscores only).
        logger: Optional logger.

    Returns:
        True if the column exists or was added, False on error.
    """
    if not column_name or not column_name.replace("_", "").isalnum():
        if logger:
            logger.warning("Invalid domain column name: %r", column_name)
        return False
    col = column_name.strip()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f'ALTER TABLE kb.concepts ADD COLUMN IF NOT EXISTS "{col}" TEXT'
            )
        if logger:
            logger.info("✅ kb.concepts.%s column ready (exists or added)", col)
        return True
    except Exception as e:
        if logger:
            logger.warning("⚠️  Could not ensure kb.concepts.%s: %s", col, e)
        return False


def ensure_soap_inventory_domain_column(conn, logger: Optional[logging.Logger] = None) -> bool:
    """
    Add domain_key to soap.inventory for domain gating and clinical context.
    Safe to call repeatedly.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                "ALTER TABLE soap.inventory ADD COLUMN IF NOT EXISTS domain_key TEXT DEFAULT 'general'"
            )
        if logger:
            logger.info("✅ soap.inventory.domain_key column ready (exists or added)")
        return True
    except Exception as e:
        if logger:
            logger.warning("⚠️  Could not ensure soap.inventory.domain_key: %s", e)
        return False


def ensure_soap_service_master_domain_column(conn, logger: Optional[logging.Logger] = None) -> bool:
    """
    Add domain_key to soap.service_master for domain gating and clinical context.
    Safe to call repeatedly.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                "ALTER TABLE soap.service_master ADD COLUMN IF NOT EXISTS domain_key TEXT DEFAULT 'general'"
            )
        if logger:
            logger.info("✅ soap.service_master.domain_key column ready (exists or added)")
        return True
    except Exception as e:
        if logger:
            logger.warning("⚠️  Could not ensure soap.service_master.domain_key: %s", e)
        return False


def ensure_soap_inventory_vetbert_column(conn, logger: Optional[logging.Logger] = None) -> bool:
    """
    Add vector_embedding_vetbert vector(768) to soap.inventory for VetBERT-based local search.
    Creates HNSW index. Safe to call repeatedly.
    """
    try:
        ensure_vector_extension(conn, logger=logger)
        with conn.cursor() as cur:
            cur.execute(
                "ALTER TABLE soap.inventory ADD COLUMN IF NOT EXISTS vector_embedding_vetbert vector(768)"
            )
        ensure_hnsw_index(
            conn,
            schema="soap",
            table="inventory",
            column="vector_embedding_vetbert",
            index_name="idx_soap_inventory_vector_embedding_vetbert_hnsw",
            logger=logger,
        )
        if logger:
            logger.info("✅ soap.inventory.vector_embedding_vetbert column and HNSW index ready")
        return True
    except Exception as e:
        if logger:
            logger.warning("⚠️  Could not ensure soap.inventory.vector_embedding_vetbert: %s", e)
        return False


def ensure_soap_service_master_vetbert_column(conn, logger: Optional[logging.Logger] = None) -> bool:
    """
    Add vector_embedding_vetbert vector(768) to soap.service_master for VetBERT-based local search.
    Creates HNSW index. Safe to call repeatedly.
    """
    try:
        ensure_vector_extension(conn, logger=logger)
        with conn.cursor() as cur:
            cur.execute(
                "ALTER TABLE soap.service_master ADD COLUMN IF NOT EXISTS vector_embedding_vetbert vector(768)"
            )
        ensure_hnsw_index(
            conn,
            schema="soap",
            table="service_master",
            column="vector_embedding_vetbert",
            index_name="idx_soap_service_master_vector_embedding_vetbert_hnsw",
            logger=logger,
        )
        if logger:
            logger.info("✅ soap.service_master.vector_embedding_vetbert column and HNSW index ready")
        return True
    except Exception as e:
        if logger:
            logger.warning("⚠️  Could not ensure soap.service_master.vector_embedding_vetbert: %s", e)
        return False


def ensure_vitals_registry_table(conn, logger: Optional[logging.Logger] = None) -> bool:
    """
    Create an optional KB vitals registry table with definitions + synonyms + embeddings.
    This is useful for:
    - Consistent VitalSign taxonomy across the pipeline
    - Fuzzy search / vector search over vitals terms (future)
    - Stronger prompt grounding if you choose to include registry excerpts
    """
    try:
        ensure_vector_extension(conn, logger=logger)
        ensure_pg_trgm(conn, logger=logger)
        ensure_fuzzystrmatch(conn, logger=logger)
    except Exception:
        pass

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS kb.vitals_registry (
                    vital_id BIGSERIAL PRIMARY KEY,
                    metric_name TEXT NOT NULL UNIQUE,
                    category TEXT,
                    definition TEXT,
                    synonyms TEXT[],
                    expected_unit TEXT,
                    search_text TEXT,
                    metaphone_key TEXT,
                    embedding vector(1536),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            # Add metaphone_key column if it doesn't exist (for existing tables)
            try:
                cur.execute("""
                    ALTER TABLE kb.vitals_registry 
                    ADD COLUMN IF NOT EXISTS metaphone_key TEXT;
                """)
            except Exception:
                pass
            
            # Helpful indexes (best-effort)
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_kb_vitals_registry_search_text_trgm
                ON kb.vitals_registry
                USING gin (search_text gin_trgm_ops);
                """
            )
            # Phonetic index on metaphone_key (B-tree for exact metaphone matches)
            try:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kb_vitals_registry_metaphone_key
                    ON kb.vitals_registry
                    USING btree (metaphone_key);
                    """
                )
            except Exception:
                pass
            try:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_kb_vitals_registry_embedding_hnsw
                    ON kb.vitals_registry
                    USING hnsw (embedding vector_cosine_ops);
                    """
                )
            except Exception:
                # HNSW may fail if extension/opclass unavailable; keep table usable.
                pass
        if logger:
            logger.info("✅ kb.vitals_registry table ready")
        return True
    except Exception as e:
        if logger:
            logger.warning(f"⚠️  Could not create kb.vitals_registry: {e}")
        return False


def seed_vitals_registry(
    conn,
    *,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    embed: bool = False,
) -> int:
    """
    Upsert the vitals registry seed rows.
    If embed=True, also stores embeddings (client parameter is optional since embed_text creates its own OpenAI client).
    """
    if not ensure_vitals_registry_table(conn, logger=logger):
        return 0

    embed_fn = None
    to_vec = None
    if embed:
        try:
            from kb_ner_embeddings import embed_text, to_pgvector_literal  # type: ignore
            embed_fn = embed_text
            to_vec = to_pgvector_literal
        except Exception:
            embed_fn = None
            to_vec = None

    upserted = 0
    with conn.cursor() as cur:
        for row in VITALS_REGISTRY_SEED:
            metric_name = (row.get("metric_name") or "").strip()
            if not metric_name:
                continue
            synonyms = row.get("synonyms") or []
            if not isinstance(synonyms, list):
                synonyms = []
            search_text = " ".join([metric_name] + [str(s) for s in synonyms if s])
            
            # Compute metaphone key for phonetic indexing
            metaphone_key = None
            try:
                cur.execute("SELECT metaphone(lower(%s), 10)", (search_text,))
                metaphone_key = cur.fetchone()[0]
            except Exception:
                metaphone_key = None

            embedding_literal = None
            if embed_fn and to_vec:
                try:
                    emb = embed_fn(f"{metric_name}. {row.get('definition') or ''}", client=client, logger=logger)
                    if emb:
                        embedding_literal = to_vec(emb)
                except Exception:
                    embedding_literal = None

            if embedding_literal:
                cur.execute(
                    """
                    INSERT INTO kb.vitals_registry (metric_name, category, definition, synonyms, expected_unit, search_text, metaphone_key, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (metric_name) DO UPDATE SET
                        category = EXCLUDED.category,
                        definition = EXCLUDED.definition,
                        synonyms = EXCLUDED.synonyms,
                        expected_unit = EXCLUDED.expected_unit,
                        search_text = EXCLUDED.search_text,
                        metaphone_key = EXCLUDED.metaphone_key,
                        embedding = EXCLUDED.embedding,
                        updated_at = now();
                    """,
                    (
                        metric_name,
                        row.get("category"),
                        row.get("definition"),
                        synonyms,
                        row.get("expected_unit"),
                        search_text,
                        metaphone_key,
                        embedding_literal,
                    ),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO kb.vitals_registry (metric_name, category, definition, synonyms, expected_unit, search_text, metaphone_key)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (metric_name) DO UPDATE SET
                        category = EXCLUDED.category,
                        definition = EXCLUDED.definition,
                        synonyms = EXCLUDED.synonyms,
                        expected_unit = EXCLUDED.expected_unit,
                        search_text = EXCLUDED.search_text,
                        metaphone_key = EXCLUDED.metaphone_key,
                        updated_at = now();
                    """,
                    (
                        metric_name,
                        row.get("category"),
                        row.get("definition"),
                        synonyms,
                        row.get("expected_unit"),
                        search_text,
                        metaphone_key,
                    ),
                )
            upserted += 1

    if logger:
        logger.info(f"✅ Seeded kb.vitals_registry: upserted={upserted} rows (embed={bool(embed_fn and to_vec)})")
    return upserted


def search_vitals_registry_topk(
    conn,
    query_text: str,
    top_k: int = 5,
    client: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
    embedding: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    Search vitals registry using trigram + phonetic (metaphone + levenshtein) + vector similarity.
    
    This function enables fuzzy phonetic matching on vitals registry entries, similar to
    the multi-modal search used in kb_ner_local_search.py and kb_ner_global_search.py.
    
    Args:
        conn: Database connection
        query_text: Search text (e.g., "temp", "heart rate", "crt")
        top_k: Number of results to return
        client: Optional OpenAI client (for embedding generation if embedding not provided)
        logger: Optional logger
        embedding: Optional pre-computed embedding (1536-dim from text-embedding-3-small)
        
    Returns:
        List of vitals registry matches with scores:
        [
            {
                "vital_id": int,
                "metric_name": str,
                "category": str,
                "definition": str,
                "expected_unit": str,
                "trigram_score": float,
                "phonetic_score": float,
                "vector_score": float,
                "match_score": float,  # max(trigram, phonetic, vector)
            },
            ...
        ]
    """
    if not query_text or not query_text.strip():
        return []
    
    query_text = query_text.strip()
    
    # For very short queries (≤3 chars), use a lower threshold and check exact matches in synonyms
    is_short_query = len(query_text) <= 3
    trigram_threshold = 0.15 if is_short_query else 0.30
    
    # Generate embedding if not provided
    vec_literal = None
    use_vector = False
    if embedding:
        try:
            from kb_ner_embeddings import to_pgvector_literal
            vec_literal = to_pgvector_literal(embedding)
            use_vector = True
        except Exception:
            vec_literal = None
            use_vector = False
    elif client:
        try:
            from kb_ner_embeddings import embed_text, to_pgvector_literal
            emb = embed_text(query_text, client=client, logger=logger)
            if emb:
                vec_literal = to_pgvector_literal(emb)
                use_vector = True
        except Exception:
            vec_literal = None
            use_vector = False
    
    try:
        ensure_fuzzystrmatch(conn, logger=logger)
        ensure_pg_trgm(conn, logger=logger)
    except Exception:
        pass
    
    try:
        with conn.cursor() as cur:
            if use_vector and vec_literal:
                # SQL with vector matching (trigram + phonetic + vector)
                sql = """
                    WITH q AS (
                        SELECT
                            lower(%s) AS q_full,
                            split_part(lower(%s), ' ', 1) AS q_first,
                            metaphone(lower(%s), 10) AS q_mfull,
                            metaphone(split_part(lower(%s), ' ', 1), 10) AS q_mfirst
                    ),
                    scored AS (
                        SELECT 
                            vital_id,
                            metric_name,
                            category,
                            definition,
                            expected_unit,
                            GREATEST(
                                similarity(lower(search_text), q.q_full),
                                similarity(lower(metric_name), q.q_full),
                                -- Boost for exact matches
                                CASE WHEN lower(metric_name) = q.q_full THEN 1.0 ELSE 0.0 END,
                                CASE WHEN q.q_full = ANY(SELECT lower(unnest(synonyms)) FROM kb.vitals_registry vr2 WHERE vr2.vital_id = kb.vitals_registry.vital_id) THEN 1.0 ELSE 0.0 END
                            ) AS trigram_score,
                            GREATEST(
                                -- Phonetic similarity (metaphone edit-distance), scaled to max 0.8
                                CASE 
                                    WHEN metaphone_key IS NOT NULL AND q.q_mfull IS NOT NULL
                                    THEN GREATEST(
                                        0.0,
                                        0.8 * (
                                            1.0 - (
                                                levenshtein(metaphone_key, q.q_mfull)::float
                                                / GREATEST(length(metaphone_key), length(q.q_mfull), 1)
                                            )
                                        )
                                    )
                                    ELSE 0.0
                                END,
                                CASE 
                                    WHEN metaphone_key IS NOT NULL AND q.q_mfirst IS NOT NULL
                                    THEN GREATEST(
                                        0.0,
                                        0.8 * (
                                            1.0 - (
                                                levenshtein(metaphone_key, q.q_mfirst)::float
                                                / GREATEST(length(metaphone_key), length(q.q_mfirst), 1)
                                            )
                                        )
                                    )
                                    ELSE 0.0
                                END,
                                -- Also try metaphone on metric_name directly (fallback if metaphone_key is NULL)
                                CASE 
                                    WHEN metaphone(lower(metric_name), 10) IS NOT NULL AND q.q_mfull IS NOT NULL
                                    THEN GREATEST(
                                        0.0,
                                        0.8 * (
                                            1.0 - (
                                                levenshtein(metaphone(lower(metric_name), 10), q.q_mfull)::float
                                                / GREATEST(length(metaphone(lower(metric_name), 10)), length(q.q_mfull), 1)
                                            )
                                        )
                                    )
                                    ELSE 0.0
                                END
                            ) AS phonetic_score,
                            CASE 
                                WHEN embedding IS NOT NULL 
                                THEN (1.0 - LEAST(embedding <=> %s::vector, 1.0))
                                ELSE 0.0
                            END AS vector_score
                        FROM kb.vitals_registry
                        CROSS JOIN q
                        WHERE (
                            similarity(lower(search_text), q.q_full) >= %s
                            OR similarity(lower(metric_name), q.q_full) >= %s
                            OR lower(metric_name) = q.q_full
                            OR q.q_full = ANY(SELECT lower(unnest(synonyms)) FROM kb.vitals_registry vr2 WHERE vr2.vital_id = kb.vitals_registry.vital_id)
                            OR (metaphone_key IS NOT NULL AND metaphone_key = q.q_mfull)
                            OR (metaphone_key IS NOT NULL AND metaphone_key = q.q_mfirst)
                            OR (embedding IS NOT NULL AND embedding <=> %s::vector < 0.5)
                        )
                    )
                    SELECT 
                        vital_id,
                        metric_name,
                        category,
                        definition,
                        expected_unit,
                        trigram_score,
                        phonetic_score,
                        vector_score,
                        GREATEST(trigram_score, phonetic_score, vector_score) AS match_score
                    FROM scored
                    ORDER BY match_score DESC
                    LIMIT %s;
                """
                params = (
                    query_text, query_text, query_text, query_text,  # q CTE (4)
                    vec_literal,  # vector_score
                    trigram_threshold, trigram_threshold,  # WHERE trigram thresholds (2)
                    vec_literal,  # WHERE vector filter
                    top_k,  # LIMIT
                )
            else:
                # SQL without vector matching (trigram + phonetic only)
                sql = """
                    WITH q AS (
                        SELECT
                            lower(%s) AS q_full,
                            split_part(lower(%s), ' ', 1) AS q_first,
                            metaphone(lower(%s), 10) AS q_mfull,
                            metaphone(split_part(lower(%s), ' ', 1), 10) AS q_mfirst
                    ),
                    scored AS (
                        SELECT 
                            vital_id,
                            metric_name,
                            category,
                            definition,
                            expected_unit,
                            GREATEST(
                                similarity(lower(search_text), q.q_full),
                                similarity(lower(metric_name), q.q_full),
                                -- Boost for exact matches
                                CASE WHEN lower(metric_name) = q.q_full THEN 1.0 ELSE 0.0 END,
                                CASE WHEN q.q_full = ANY(SELECT lower(unnest(synonyms)) FROM kb.vitals_registry vr2 WHERE vr2.vital_id = kb.vitals_registry.vital_id) THEN 1.0 ELSE 0.0 END
                            ) AS trigram_score,
                            GREATEST(
                                -- Phonetic similarity (metaphone edit-distance), scaled to max 0.8
                                CASE 
                                    WHEN metaphone_key IS NOT NULL AND q.q_mfull IS NOT NULL
                                    THEN GREATEST(
                                        0.0,
                                        0.8 * (
                                            1.0 - (
                                                levenshtein(metaphone_key, q.q_mfull)::float
                                                / GREATEST(length(metaphone_key), length(q.q_mfull), 1)
                                            )
                                        )
                                    )
                                    ELSE 0.0
                                END,
                                CASE 
                                    WHEN metaphone_key IS NOT NULL AND q.q_mfirst IS NOT NULL
                                    THEN GREATEST(
                                        0.0,
                                        0.8 * (
                                            1.0 - (
                                                levenshtein(metaphone_key, q.q_mfirst)::float
                                                / GREATEST(length(metaphone_key), length(q.q_mfirst), 1)
                                            )
                                        )
                                    )
                                    ELSE 0.0
                                END,
                                -- Also try metaphone on metric_name directly (fallback if metaphone_key is NULL)
                                CASE 
                                    WHEN metaphone(lower(metric_name), 10) IS NOT NULL AND q.q_mfull IS NOT NULL
                                    THEN GREATEST(
                                        0.0,
                                        0.8 * (
                                            1.0 - (
                                                levenshtein(metaphone(lower(metric_name), 10), q.q_mfull)::float
                                                / GREATEST(length(metaphone(lower(metric_name), 10)), length(q.q_mfull), 1)
                                            )
                                        )
                                    )
                                    ELSE 0.0
                                END
                            ) AS phonetic_score,
                            0.0 AS vector_score
                        FROM kb.vitals_registry
                        CROSS JOIN q
                        WHERE (
                            similarity(lower(search_text), q.q_full) >= %s
                            OR similarity(lower(metric_name), q.q_full) >= %s
                            OR lower(metric_name) = q.q_full
                            OR q.q_full = ANY(SELECT lower(unnest(synonyms)) FROM kb.vitals_registry vr2 WHERE vr2.vital_id = kb.vitals_registry.vital_id)
                            OR (metaphone_key IS NOT NULL AND metaphone_key = q.q_mfull)
                            OR (metaphone_key IS NOT NULL AND metaphone_key = q.q_mfirst)
                        )
                    )
                    SELECT 
                        vital_id,
                        metric_name,
                        category,
                        definition,
                        expected_unit,
                        trigram_score,
                        phonetic_score,
                        vector_score,
                        GREATEST(trigram_score, phonetic_score, vector_score) AS match_score
                    FROM scored
                    ORDER BY match_score DESC
                    LIMIT %s;
                """
                params = (
                    query_text, query_text, query_text, query_text,  # q CTE (4)
                    trigram_threshold, trigram_threshold,  # WHERE trigram thresholds (2)
                    top_k,  # LIMIT
                )
            
            cur.execute(sql, params)
            rows = cur.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "vital_id": row[0],
                    "metric_name": row[1],
                    "category": row[2],
                    "definition": row[3],
                    "expected_unit": row[4],
                    "trigram_score": float(row[5]) if row[5] is not None else 0.0,
                    "phonetic_score": float(row[6]) if row[6] is not None else 0.0,
                    "vector_score": float(row[7]) if row[7] is not None else 0.0,
                    "match_score": float(row[8]) if row[8] is not None else 0.0,
                })
            
            return results
            
    except Exception as e:
        if logger:
            logger.warning(f"⚠️  Vitals registry search failed: {e}")
        return []


def ensure_learned_aliases_table(conn, logger: Optional[logging.Logger] = None) -> bool:
    """
    Best-effort create kb.learned_aliases for self-healing aliasing.
    Production-ready schema with vector support and validation flags.
    """
    try:
        with conn.cursor() as cur:
            # Create table with production-ready schema
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kb.learned_aliases (
                    id SERIAL PRIMARY KEY,
                    alias_text TEXT NOT NULL,
                    kb_concept_id BIGINT NOT NULL,
                    kb_kind TEXT,
                    confidence_score DOUBLE PRECISION,
                    source TEXT,
                    embedding vector,
                    is_validated BOOLEAN DEFAULT FALSE,
                    frequency_count BIGINT DEFAULT 1,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    last_used_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    CONSTRAINT unique_alias_mapping UNIQUE (alias_text, kb_concept_id),
                    CONSTRAINT fk_learned_aliases_concept 
                        FOREIGN KEY (kb_concept_id) 
                        REFERENCES kb.concepts(concept_id) 
                        ON DELETE CASCADE
                );
            """)
            
            # Create performance indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_aliases_text_exact ON kb.learned_aliases (alias_text);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_aliases_embedding ON kb.learned_aliases USING hnsw (embedding vector_cosine_ops);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_aliases_trigram ON kb.learned_aliases USING gin (alias_text gin_trgm_ops);")
            
        conn.commit()
        return True
    except Exception as e:
        if logger:
            logger.debug(f"Could not ensure kb.learned_aliases table: {e}")
    return False


def insert_ai_draft_entity(
    conn,
    visit_id: str,
    span_text: str,
    guessed_kb_concept_id: Optional[int],
    confidence_score: Optional[float] = None,
    clinic_id: Optional[int] = None,
    stock_id: Optional[int] = None,
    service_id: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Mailbox Pattern (Deferred Learning):
        Store AI guesses temporarily for PMS to validate once an invoice is finalized.

    Writes into `soap.ai_draft_entities` (visit-scoped mailbox).
    This intentionally does NOT write to `kb.learned_aliases`.
    
    Args:
        visit_id: UUID of the visit/appointment (required for PMS harvester lookup)
        span_text: Original text from transcript (e.g., "Cortex caps")
        guessed_kb_concept_id: KB concept ID the AI matched (e.g., 78800 for "Coatex")
        confidence_score: AI's confidence in the match (0.0-1.0)
        clinic_id: Clinic ID for clinic-scoped duplicate checking (prevents cross-clinic contamination)
        stock_id: Stock ID from inventory (if known at draft time, else NULL - harvester will populate from invoice)
        service_id: Service ID from service_master (if known at draft time, else NULL - harvester will populate from invoice)
    """
    if not visit_id or not span_text:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO soap.ai_draft_entities
                    (visit_id, span_text, guessed_kb_concept_id, confidence_score, clinic_id, stock_id, service_id)
                VALUES 
                    (%s::uuid, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
                """,
                (visit_id, span_text, guessed_kb_concept_id, confidence_score, clinic_id, stock_id, service_id),
            )
        conn.commit()
        return True
    except Exception as e:
        if logger:
            logger.debug(f"Failed to insert draft entity: {e}")
        return False


def write_manifest_to_mailbox(
    conn,
    visit_id: str,
    manifest: List[Dict[str, Any]],
    clinic_id: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    GAP 1 FIX: Write entity manifest to mailbox for deferred learning.
    
    Writes all entities from manifest that have a kb_concept_id to soap.ai_draft_entities.
    This ensures the PMS harvester can validate AI guesses when the invoice is finalized.
    
    Args:
        conn: Database connection
        visit_id: UUID of the visit/appointment
        manifest: List of entity dicts from analyze_full_soap or run_step_2_3_normalization
        clinic_id: Clinic ID for clinic-scoped duplicate checking
        logger: Optional logger
        
    Returns:
        Number of entities written to mailbox
    """
    if not visit_id or not manifest:
        return 0
    
    written_count = 0
    try:
        with conn.cursor() as cur:
            for entity in manifest:
                # Only save items that found a KB link
                kb_concept_id = entity.get("kb_concept_id") or entity.get("concept_id")
                if not kb_concept_id:
                    continue
                
                span_text = entity.get("span_text") or entity.get("normalized_name") or entity.get("text")
                if not span_text:
                    continue
                
                # Extract confidence score from various possible fields
                confidence_score = (
                    entity.get("confidence_score") or 
                    entity.get("hybrid_score") or 
                    entity.get("similarity_score") or
                    (1.0 - entity.get("distance", 0.0)) if entity.get("distance") is not None else None
                )
                
                # Write to mailbox
                cur.execute(
                    """
                    INSERT INTO soap.ai_draft_entities
                (visit_id, span_text, guessed_kb_concept_id, confidence_score, clinic_id, stock_id, service_id)
                        VALUES 
                (%s::uuid, %s, %s, %s, %s, NULL, NULL)
                    ON CONFLICT DO NOTHING;
                    """,
                    (visit_id, span_text, kb_concept_id, confidence_score, clinic_id),
                )
                written_count += 1
        
        conn.commit()
        if logger:
            logger.info(f"📬 Wrote {written_count} entities to mailbox (visit_id={visit_id})")
        return written_count
    except Exception as e:
        if logger:
            logger.warning(f"Failed to write manifest to mailbox: {e}")
        return 0
