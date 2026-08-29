-- Local soap schema aligned with Master Doc (requires pgvector).
-- Run CREATE EXTENSION vector first (scripts/enable_pgvector.py).

CREATE SCHEMA IF NOT EXISTS soap;
CREATE SCHEMA IF NOT EXISTS kb;

CREATE TABLE IF NOT EXISTS soap.inventory (
  stock_id BIGINT PRIMARY KEY,
  category varchar(255),
  sub_category varchar(255),
  hsn_code bigint,
  item_name varchar(500) NOT NULL,
  trade_name varchar(500),
  brand varchar(255),
  major_active_ingredients text,
  nature varchar(100),
  bar_code_number varchar(100),
  batch_number varchar(100),
  manufacturing_date date,
  expiry_date date,
  uqc_code varchar(50),
  purchase_uom varchar(50),
  sales_uom varchar(50),
  conversion_factor integer DEFAULT 1,
  quantity integer DEFAULT 0,
  subunits integer DEFAULT 0,
  administered_uom varchar(50),
  dosage numeric(10),
  dosage_type varchar(100),
  unit_selling_price numeric(10),
  cgst numeric(5),
  sgst numeric(5),
  item_total numeric(10) DEFAULT 0,
  landed_cost numeric(10),
  billable varchar(1),
  brief_description text,
  location_id integer,
  auto_consumable varchar(3),
  snomed_code varchar(50),
  venom_code varchar(50),
  vector_embedding vector(1536),
  created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  internal_description text,
  internal_description_vector text,
  kb_concept_id integer,
  loinc_code varchar(50),
  domain_key text,
  vector_embedding_vetbert vector(768)
);

CREATE TABLE IF NOT EXISTS soap.service_master (
  service_id BIGINT PRIMARY KEY,
  type varchar(100),
  hsn_sac_code bigint,
  category varchar(255),
  sub_category varchar(255),
  procedure_name varchar(500) NOT NULL,
  tax_applicability varchar(10),
  selling_price numeric(10) NOT NULL,
  tax_percent numeric(5) DEFAULT 0,
  remarks text,
  cgst numeric(5) DEFAULT 0,
  sgst numeric(5) DEFAULT 0,
  total_price numeric(10),
  snomed_code varchar(50),
  cpt_code varchar(50),
  icd10_code varchar(50),
  duration_minutes integer,
  requires_anesthesia boolean DEFAULT false,
  requires_equipment text,
  requires_specialist boolean DEFAULT false,
  billable boolean DEFAULT true,
  requires_consumables boolean DEFAULT false,
  vector_embedding vector(1536),
  created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  is_active boolean DEFAULT true,
  internal_description text,
  internal_description_vector text,
  kb_concept_id integer,
  venom_code varchar(50),
  loinc_code varchar(50),
  domain_key text,
  vector_embedding_vetbert vector(768)
);

CREATE TABLE IF NOT EXISTS kb.vitals_registry (
    vital_id BIGSERIAL PRIMARY KEY,
    metric_name TEXT NOT NULL UNIQUE,
    category TEXT,
    definition TEXT,
    synonyms TEXT[],
    expected_unit TEXT,
    search_text TEXT,
    metaphone_key TEXT,
    embedding text,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_soap_inventory_item_name_trgm
ON soap.inventory USING gin (item_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_soap_inventory_trade_name_trgm
ON soap.inventory USING gin (trade_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_soap_inventory_location_id
ON soap.inventory (location_id);

CREATE INDEX IF NOT EXISTS idx_soap_service_master_procedure_name_trgm
ON soap.service_master USING gin (procedure_name gin_trgm_ops);
