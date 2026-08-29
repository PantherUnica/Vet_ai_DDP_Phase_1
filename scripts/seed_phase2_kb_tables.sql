-- Phase 2 instruction tables (Master Doc / SOAP_notes_billing_phase2_kb_atoms.py)
-- Seeded for local Clinic DB so Phase 2 can load assertion_types + attributes_schema.

CREATE SCHEMA IF NOT EXISTS kb;

CREATE TABLE IF NOT EXISTS kb.assertion_types (
  assertion_id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  description TEXT,
  billing_impact TEXT
);

CREATE TABLE IF NOT EXISTS kb.attributes_schema (
  id SERIAL PRIMARY KEY,
  source_kind TEXT NOT NULL,
  relationship TEXT NOT NULL,
  target_attribute TEXT NOT NULL,
  use_case TEXT,
  is_required BOOLEAN DEFAULT false,
  UNIQUE (source_kind, relationship, target_attribute)
);

INSERT INTO kb.assertion_types (assertion_id, label, description, billing_impact) VALUES
  ('CONF', 'Confirmed', 'Entity is affirmed / performed / prescribed in the note', 'billable_when_applicable'),
  ('NEG', 'Negated', 'Entity was declined, denied, or ruled out', 'usually_not_billable'),
  ('SUSP', 'Suspected', 'Entity is suspected / possible, not confirmed', 'review'),
  ('HIST', 'Historical', 'Past history, not current visit action', 'usually_not_billable'),
  ('HYPO', 'Hypothetical', 'Conditional / if-needed language', 'usually_not_billable'),
  ('RECUR', 'Recurring', 'Chronic or recurring issue', 'context')
ON CONFLICT (assertion_id) DO NOTHING;

INSERT INTO kb.attributes_schema (source_kind, relationship, target_attribute, use_case, is_required) VALUES
  ('Medicine', 'dose', 'dose', 'e.g. 5mg', true),
  ('Medicine', 'route', 'route', 'e.g. PO, IV', false),
  ('Medicine', 'frequency', 'frequency', 'e.g. once daily', false),
  ('Vaccine', 'dose', 'dose', 'dose number or amount', false),
  ('Vaccine', 'route', 'route', 'e.g. SQ', false),
  ('Diagnostic', 'specimen_type', 'specimen_type', 'e.g. Blood', false),
  ('Diagnostic', 'priority', 'priority', 'e.g. Routine / Urgent', false),
  ('VitalSign', 'metric_name', 'metric_name', 'e.g. Temperature', true),
  ('VitalSign', 'value', 'value', 'numeric reading', true),
  ('VitalSign', 'unit', 'unit', 'e.g. Fahrenheit', false),
  ('VitalSign', 'qualitative_flag', 'qualitative_flag', 'High / Low / Normal', false),
  ('Procedure', 'status', 'status', 'Performed / Planned', false),
  ('Reason', 'chief_complaint', 'chief_complaint', 'presenting complaint', false),
  ('Reason', 'urgency', 'urgency', 'Routine / Urgent', false),
  ('Product', 'dose', 'dose', 'product dose if stated', false),
  ('Product', 'route', 'route', 'route if stated', false)
ON CONFLICT (source_kind, relationship, target_attribute) DO NOTHING;
