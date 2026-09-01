-- Minimal clinic catalog for local dev when the full SQL_dump CSVs are unavailable.
-- CLINIC_ID=1 maps to location_id=8 in kb_ner_local_search.py.
-- Safe to re-run: uses fixed primary keys and ON CONFLICT DO NOTHING.

INSERT INTO soap.inventory (
  stock_id, category, item_name, trade_name, dosage_type, sales_uom,
  location_id, billable, domain_key
) VALUES
  (900001, 'Medication', 'Meloxicam Oral Suspension', 'Melonex', 'Liquid', 'ml', 8, 'Y', 'general'),
  (900002, 'Medication', 'Cefpodoxime Dry Syrup', 'CefPET Dry Syrup', 'Liquid', 'ml', 8, 'Y', 'general'),
  (900003, 'Flea & Tick Treatment', 'Fluralaner Chewable', 'Bravecto', 'Tablet', 'tab', 8, 'Y', 'dermatology'),
  (900004, 'Vaccines', 'DHPP Vaccine', 'Nobivac DHPPi', 'Injection', 'dose', 8, 'Y', 'preventive'),
  (900005, 'Diet', 'Renal Support Diet', 'Hill''s k/d', 'Diet', 'kg', 8, 'Y', 'nephrology'),
  (900006, 'Medication', 'Amoxicillin Clavulanate', 'Clavamox', 'Tablet', 'tab', 8, 'Y', 'general')
ON CONFLICT (stock_id) DO NOTHING;

INSERT INTO soap.service_master (
  service_id, category, procedure_name, selling_price, billable, is_active, domain_key
) VALUES
  (800001, 'Consultation', 'General Consultation', 500, true, true, 'general'),
  (800002, 'Consultation', 'Orthopaedic Consultation', 750, true, true, 'orthopaedic'),
  (800003, 'Radiology', 'Digital X-Ray', 1200, true, true, 'radiology'),
  (800004, 'Lab', 'Complete Blood Count', 900, true, true, 'lab'),
  (800005, 'Surgery', 'Soft Tissue Surgery', 5000, true, true, 'surgery'),
  (800006, 'Rehabilitation & Physiotherapy', 'Physiotherapy Session', 800, true, true, 'rehabilitation'),
  (800007, 'Hygiene & Grooming', 'Bath and Brush', 400, true, true, 'grooming')
ON CONFLICT (service_id) DO NOTHING;
