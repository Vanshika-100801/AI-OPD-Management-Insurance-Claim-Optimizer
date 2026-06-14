"""
=============================================================
AI OPD Management & Insurance Claim Optimizer
Phase 1: ETL Pipeline & Data Validation
IBM Project
=============================================================
Run AFTER phase1_generate_data.py
Usage: python3 phase1_etl_validate.py
=============================================================
"""

import sqlite3
import csv
import os
import json
from datetime import date

DB_PATH  = "opd_database.db"
RPT_PATH = "phase1_etl_report.json"

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur  = conn.cursor()

print("="*60)
print("  PHASE 1 — ETL PIPELINE & DATA VALIDATION")
print("="*60)

report = {
    "run_date"    : date.today().isoformat(),
    "database"    : DB_PATH,
    "validations" : [],
    "kpis"        : {},
    "anomalies"   : []
}

def check(label, query, expect_zero=True):
    cur.execute(query)
    rows = cur.fetchall()
    count = len(rows)
    status = "PASS" if (count == 0) == expect_zero else "FAIL"
    icon   = "✓" if status == "PASS" else "✗"
    print(f"  {icon} [{status}] {label}  ({count} issues)")
    report["validations"].append({"check": label, "status": status, "issues": count})
    return rows

print("\n── 1. NULL / COMPLETENESS CHECKS ──────────────────────")
check("Patients with null name",
      "SELECT patient_id FROM patients WHERE patient_name IS NULL OR patient_name=''")
check("Appointments with null date",
      "SELECT appointment_id FROM appointments WHERE appointment_date IS NULL")
check("Billing with null total",
      "SELECT bill_id FROM billing WHERE total_amount IS NULL OR total_amount <= 0")
check("Claims with null claim_amount",
      "SELECT claim_id FROM insurance_claims WHERE claim_amount IS NULL OR claim_amount <= 0")

print("\n── 2. REFERENTIAL INTEGRITY CHECKS ────────────────────")
check("Appointments → invalid patient_id",
      "SELECT appointment_id FROM appointments WHERE patient_id NOT IN (SELECT patient_id FROM patients)")
check("Billing → invalid appointment_id",
      "SELECT bill_id FROM billing WHERE appointment_id NOT IN (SELECT appointment_id FROM appointments)")
check("Claims → invalid bill_id",
      "SELECT claim_id FROM insurance_claims WHERE bill_id NOT IN (SELECT bill_id FROM billing)")
check("Claims → invalid insurer_id",
      "SELECT claim_id FROM insurance_claims WHERE insurer_id NOT IN (SELECT insurer_id FROM insurers)")

print("\n── 3. BUSINESS LOGIC CHECKS ────────────────────────────")
check("Billing where patient_payable > total_amount",
      "SELECT bill_id FROM billing WHERE patient_payable > total_amount")
check("Approved claims with zero approved_amount",
      "SELECT claim_id FROM insurance_claims WHERE claim_status='Approved' AND (approved_amount IS NULL OR approved_amount=0)")
check("Negative processing days",
      "SELECT claim_id FROM insurance_claims WHERE processing_days IS NOT NULL AND processing_days < 0")
check("Completed appointments with null wait_time",
      "SELECT appointment_id FROM appointments WHERE status='Completed' AND wait_time_mins IS NULL")
check("Future appointment dates",
      f"SELECT appointment_id FROM appointments WHERE appointment_date > '2024-12-31'")

print("\n── 4. DATA DISTRIBUTION CHECKS ─────────────────────────")

# Gender distribution
cur.execute("SELECT gender, COUNT(*) as cnt FROM patients GROUP BY gender")
genders = {row["gender"]: row["cnt"] for row in cur.fetchall()}
print(f"  Gender distribution: {genders}")
report["kpis"]["gender_distribution"] = genders

# Age distribution
cur.execute("""
    SELECT
        CASE
            WHEN age < 18  THEN 'Child (0-17)'
            WHEN age < 40  THEN 'Young Adult (18-39)'
            WHEN age < 60  THEN 'Middle Age (40-59)'
            ELSE                'Senior (60+)'
        END as age_group,
        COUNT(*) as cnt
    FROM patients GROUP BY age_group ORDER BY age_group
""")
age_groups = {row["age_group"]: row["cnt"] for row in cur.fetchall()}
print(f"  Age groups: {age_groups}")
report["kpis"]["age_groups"] = age_groups

print("\n── 5. KEY PERFORMANCE INDICATORS ───────────────────────")

# OPD KPIs
cur.execute("""
    SELECT
        COUNT(*) as total_appts,
        SUM(CASE WHEN status='Completed'   THEN 1 ELSE 0 END) as completed,
        SUM(CASE WHEN status='No-Show'     THEN 1 ELSE 0 END) as no_shows,
        SUM(CASE WHEN status='Cancelled'   THEN 1 ELSE 0 END) as cancelled,
        ROUND(AVG(CASE WHEN status='Completed' THEN wait_time_mins END), 1) as avg_wait_mins,
        ROUND(AVG(CASE WHEN status='Completed' THEN consultation_mins END), 1) as avg_consult_mins,
        ROUND(100.0 * SUM(CASE WHEN status='No-Show' THEN 1 ELSE 0 END) / COUNT(*), 2) as no_show_pct
    FROM appointments
""")
opd = dict(cur.fetchone())
print(f"  Total Appointments   : {opd['total_appts']}")
print(f"  Completed            : {opd['completed']}  ({100*opd['completed']//opd['total_appts']}%)")
print(f"  No-Shows             : {opd['no_shows']}   ({opd['no_show_pct']}%)")
print(f"  Avg Wait Time        : {opd['avg_wait_mins']} mins")
print(f"  Avg Consultation     : {opd['avg_consult_mins']} mins")
report["kpis"]["opd"] = opd

# Billing KPIs
cur.execute("""
    SELECT
        COUNT(*)                              as total_bills,
        ROUND(SUM(total_amount), 2)           as total_revenue,
        ROUND(AVG(total_amount), 2)           as avg_bill,
        ROUND(MAX(total_amount), 2)           as max_bill,
        ROUND(SUM(insured_amount), 2)         as total_insured,
        ROUND(100.0*SUM(insured_amount)/SUM(total_amount), 1) as insured_pct,
        SUM(CASE WHEN payment_status='Pending' THEN 1 ELSE 0 END) as pending_payments
    FROM billing
""")
billing = dict(cur.fetchone())
print(f"\n  Total Revenue        : ₹{billing['total_revenue']:,.2f}")
print(f"  Avg Bill Amount      : ₹{billing['avg_bill']:,.2f}")
print(f"  Insured Amount %     : {billing['insured_pct']}%")
print(f"  Pending Payments     : {billing['pending_payments']}")
report["kpis"]["billing"] = billing

# Claims KPIs
cur.execute("""
    SELECT
        COUNT(*)                               as total_claims,
        SUM(CASE WHEN claim_status='Approved'     THEN 1 ELSE 0 END) as approved,
        SUM(CASE WHEN claim_status='Rejected'     THEN 1 ELSE 0 END) as rejected,
        SUM(CASE WHEN claim_status='Pending'      THEN 1 ELSE 0 END) as pending,
        SUM(CASE WHEN claim_status='Under Review' THEN 1 ELSE 0 END) as under_review,
        ROUND(100.0 * SUM(CASE WHEN claim_status='Rejected' THEN 1 ELSE 0 END) / COUNT(*), 2) as rejection_rate,
        ROUND(AVG(CASE WHEN claim_status='Approved' THEN processing_days END), 1) as avg_approval_days,
        ROUND(SUM(claim_amount), 2) as total_claimed,
        ROUND(SUM(approved_amount), 2) as total_approved
    FROM insurance_claims
""")
claims = dict(cur.fetchone())
print(f"\n  Total Claims         : {claims['total_claims']}")
print(f"  Approved             : {claims['approved']} ({100*claims['approved']//claims['total_claims']}%)")
print(f"  Rejected             : {claims['rejected']}  Rejection Rate: {claims['rejection_rate']}%")
print(f"  Avg Approval Days    : {claims['avg_approval_days']} days")
print(f"  Total Claimed        : ₹{claims['total_claimed']:,.2f}")
report["kpis"]["claims"] = claims

print("\n── 6. TOP REJECTION REASONS ────────────────────────────")
cur.execute("""
    SELECT rejection_code, rejection_reason, COUNT(*) as cnt
    FROM insurance_claims
    WHERE claim_status='Rejected'
    GROUP BY rejection_code
    ORDER BY cnt DESC
    LIMIT 5
""")
rej_top = []
for row in cur.fetchall():
    print(f"  {row['rejection_code']}  {row['rejection_reason']:<40}  {row['cnt']} cases")
    rej_top.append(dict(row))
report["kpis"]["top_rejection_reasons"] = rej_top

print("\n── 7. BUSIEST DEPARTMENTS ──────────────────────────────")
cur.execute("""
    SELECT dep.dept_name, COUNT(*) as visits,
           ROUND(AVG(a.wait_time_mins),1) as avg_wait
    FROM appointments a
    JOIN departments dep ON a.department_id = dep.department_id
    WHERE a.status='Completed'
    GROUP BY dep.dept_name
    ORDER BY visits DESC
""")
for row in cur.fetchall():
    print(f"  {row['dept_name']:<22} {row['visits']:>4} visits   avg wait: {row['avg_wait']} mins")

print("\n── 8. INSURER PERFORMANCE ──────────────────────────────")
cur.execute("""
    SELECT ins.insurer_name, ins.insurer_type,
           COUNT(*) as claims,
           ROUND(100.0*SUM(CASE WHEN ic.claim_status='Rejected' THEN 1 ELSE 0 END)/COUNT(*),1) as rej_rate,
           ROUND(AVG(CASE WHEN ic.claim_status='Approved' THEN ic.processing_days END),1) as avg_days
    FROM insurance_claims ic
    JOIN insurers ins ON ic.insurer_id = ins.insurer_id
    GROUP BY ins.insurer_id
    ORDER BY rej_rate DESC
""")
for row in cur.fetchall():
    print(f"  {row['insurer_name']:<22} {row['insurer_type']:<12} claims={row['claims']:>3}  rej={row['rej_rate']:>5}%  avg_days={row['avg_days']}")

# ── Save JSON report
with open(RPT_PATH, "w") as f:
    json.dump(report, f, indent=2, default=str)

conn.close()
print(f"\n✓ ETL report saved: {RPT_PATH}")
print("="*60)
print("  Phase 1 COMPLETE — Ready for Phase 2 (EDA)")
print("="*60)
