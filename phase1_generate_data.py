"""
=============================================================
AI OPD Management & Insurance Claim Optimizer
Phase 1: Synthetic Data Generator
IBM Project | Python 3.x | No external dependencies
=============================================================
Run: python3 phase1_generate_data.py
Output: opd_database.db  +  CSV exports in /csv_exports/
=============================================================
"""

import sqlite3
import random
import os
import csv
from datetime import date, timedelta, datetime

# ─── CONFIG ──────────────────────────────────────────────────
SEED            = 42
NUM_PATIENTS    = 500
NUM_APPOINTMENTS= 2000
START_DATE      = date(2023, 1, 1)
END_DATE        = date(2024, 12, 31)
DB_PATH         = "opd_database.db"
CSV_DIR         = "csv_exports"
random.seed(SEED)
os.makedirs(CSV_DIR, exist_ok=True)

# ─── HELPER FUNCTIONS ────────────────────────────────────────
def rand_date(start=START_DATE, end=END_DATE):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def rand_time(hour_min=8, hour_max=18):
    h = random.randint(hour_min, hour_max - 1)
    m = random.choice([0, 15, 30, 45])
    return f"{h:02d}:{m:02d}"

def add_mins(time_str, mins):
    h, m = map(int, time_str.split(":"))
    total = h * 60 + m + mins
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"

def phone():
    return f"9{random.randint(100000000, 999999999)}"

def policy_num():
    return f"POL{random.randint(100000, 999999)}"

# ─── MASTER DATA ─────────────────────────────────────────────
DEPARTMENTS = [
    ("General Medicine",   "GM",  1, 20),
    ("Cardiology",         "CAR", 2, 15),
    ("Orthopedics",        "ORT", 3, 12),
    ("Gynecology",         "GYN", 2, 10),
    ("Pediatrics",         "PED", 1, 18),
    ("Dermatology",        "DER", 4,  8),
    ("ENT",                "ENT", 4,  6),
    ("Ophthalmology",      "OPH", 3,  6),
    ("Neurology",          "NEU", 5, 10),
    ("Endocrinology",      "END", 5,  8),
]

DOCTOR_POOL = [
    # (name, specialization, dept_index_0based, qualification, experience)
    ("Dr. Rajesh Sharma",    "General Physician",   0, "MBBS, MD",       12),
    ("Dr. Priya Nair",       "General Physician",   0, "MBBS",            6),
    ("Dr. Amit Kulkarni",    "Cardiologist",        1, "MBBS, DM Cardio",18),
    ("Dr. Sunita Mehta",     "Cardiologist",        1, "MBBS, MD, DM",   14),
    ("Dr. Vikram Singh",     "Orthopedic Surgeon",  2, "MBBS, MS Ortho", 10),
    ("Dr. Rekha Pillai",     "Orthopedic Surgeon",  2, "MBBS, DNB",       8),
    ("Dr. Anita Desai",      "Gynecologist",        3, "MBBS, MS OBG",   15),
    ("Dr. Kavitha Rajan",    "Gynecologist",        3, "MBBS, DGO",       9),
    ("Dr. Suresh Babu",      "Pediatrician",        4, "MBBS, DCH, MD",  11),
    ("Dr. Meena Krishnan",   "Pediatrician",        4, "MBBS, MD Peds",   7),
    ("Dr. Arjun Verma",      "Dermatologist",       5, "MBBS, DVD",       5),
    ("Dr. Pooja Agarwal",    "ENT Specialist",      6, "MBBS, MS ENT",   13),
    ("Dr. Ravi Menon",       "Ophthalmologist",     7, "MBBS, MS Ophth", 16),
    ("Dr. Nisha Tiwari",     "Neurologist",         8, "MBBS, DM Neuro", 20),
    ("Dr. Gopal Rao",        "Endocrinologist",     9, "MBBS, DM Endo",  11),
]

INSURERS = [
    ("CGHS",              "Government", 5,  8.2),
    ("ESIC",              "Government", 7, 11.5),
    ("Star Health",       "Private",    10, 18.3),
    ("HDFC ERGO",         "Private",    12, 15.7),
    ("New India Assurance","Government",8, 13.1),
    ("Bajaj Allianz",     "Private",    14, 20.4),
    ("Aditya Birla Health","Corporate", 9, 10.8),
    ("ICICI Lombard",     "Private",    11, 16.9),
    ("United India",      "Government", 6,  9.3),
    ("Religare Health",   "Corporate",  8, 12.6),
]

FIRST_NAMES = [
    "Aarav","Vivaan","Aditya","Vihaan","Arjun","Rohan","Sai","Karan","Raj","Dev",
    "Priya","Ananya","Diya","Riya","Shreya","Kavya","Nisha","Pooja","Sunita","Meena",
    "Amit","Suresh","Vikram","Deepak","Manoj","Rahul","Sanjay","Ajay","Vinay","Nikhil",
    "Rekha","Kavitha","Anita","Geeta","Radha","Lata","Usha","Veena","Sheela","Mala",
    "Ravi","Gopal","Ramesh","Sunil","Anil","Ashok","Mohan","Ganesh","Kartik","Pranav",
]

LAST_NAMES = [
    "Sharma","Verma","Gupta","Singh","Kumar","Patel","Nair","Menon","Pillai","Rao",
    "Iyer","Krishnan","Desai","Mehta","Shah","Joshi","Mishra","Pandey","Tiwari","Dubey",
    "Reddy","Naidu","Gowda","Hegde","Kamat","Shetty","Bhat","Kulkarni","Jain","Agarwal",
]

CITIES = [
    "Mumbai","Delhi","Bangalore","Hyderabad","Chennai","Kolkata","Pune","Ahmedabad",
    "Jaipur","Lucknow","Surat","Kanpur","Nagpur","Indore","Bhopal","Patna","Vadodara",
]

BLOOD_GROUPS = ["A+","A-","B+","B-","AB+","AB-","O+","O-"]

DIAGNOSIS_CODES = [
    ("J06.9",  "Acute upper respiratory infection"),
    ("K21.0",  "Gastro-oesophageal reflux with oesophagitis"),
    ("I10",    "Essential hypertension"),
    ("E11.9",  "Type 2 diabetes mellitus"),
    ("M54.5",  "Low back pain"),
    ("J45.9",  "Asthma, unspecified"),
    ("K29.7",  "Gastritis, unspecified"),
    ("N39.0",  "Urinary tract infection"),
    ("A09",    "Diarrhoea and gastroenteritis"),
    ("R05",    "Cough"),
    ("R51",    "Headache"),
    ("K80.20", "Calculus of gallbladder"),
    ("I25.10", "Coronary artery disease"),
    ("G43.909","Migraine, unspecified"),
    ("L30.9",  "Dermatitis, unspecified"),
    ("H10.9",  "Conjunctivitis, unspecified"),
    ("J03.90", "Acute tonsillitis"),
    ("E78.5",  "Hyperlipidaemia"),
    ("F41.1",  "Generalized anxiety disorder"),
    ("M06.9",  "Rheumatoid arthritis, unspecified"),
]

REJECTION_REASONS = [
    ("R001", "Policy lapsed / not active"),
    ("R002", "Treatment not covered under policy"),
    ("R003", "Pre-existing condition exclusion"),
    ("R004", "Incomplete documentation submitted"),
    ("R005", "Duplicate claim"),
    ("R006", "Exceeds policy limit"),
    ("R007", "Waiting period not completed"),
    ("R008", "Non-empanelled hospital/doctor"),
    ("R009", "Wrong ICD code submitted"),
    ("R010", "Claim submitted after deadline"),
]

SLOT_TIMES = [f"{h:02d}:{m:02d}" for h in range(8, 18) for m in [0, 30]]

# ─── DATABASE SETUP ──────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
cur  = conn.cursor()

# Execute schema
with open("phase1_schema.sql", "r") as f:
    cur.executescript(f.read())
conn.commit()
print("✓ Schema created")

# ─── INSERT DEPARTMENTS ──────────────────────────────────────
cur.executemany(
    "INSERT INTO departments (dept_name, dept_code, floor_no, total_beds) VALUES (?,?,?,?)",
    DEPARTMENTS
)
conn.commit()
print("✓ Departments inserted:", len(DEPARTMENTS))

# ─── INSERT INSURERS ─────────────────────────────────────────
cur.executemany(
    "INSERT INTO insurers (insurer_name, insurer_type, avg_approval_days, rejection_rate_pct) VALUES (?,?,?,?)",
    INSURERS
)
conn.commit()
print("✓ Insurers inserted:", len(INSURERS))

# ─── INSERT DOCTORS ──────────────────────────────────────────
for name, spec, dept_idx, qual, exp in DOCTOR_POOL:
    cur.execute(
        "INSERT INTO doctors (doctor_name, specialization, department_id, qualification, experience_yrs) VALUES (?,?,?,?,?)",
        (name, spec, dept_idx + 1, qual, exp)
    )
conn.commit()
print("✓ Doctors inserted:", len(DOCTOR_POOL))

# ─── INSERT PATIENTS ─────────────────────────────────────────
patient_ids = []
for _ in range(NUM_PATIENTS):
    name    = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    age     = random.randint(2, 85)
    gender  = random.choices(["Male","Female","Other"], weights=[48,50,2])[0]
    city    = random.choice(CITIES)
    ph      = phone()
    bg      = random.choice(BLOOD_GROUPS)
    ins_id  = random.randint(1, len(INSURERS))
    pol     = policy_num()
    reg     = rand_date(date(2020,1,1), START_DATE)
    cur.execute(
        "INSERT INTO patients (patient_name,age,gender,city,phone,blood_group,insurer_id,policy_number,registration_date) VALUES (?,?,?,?,?,?,?,?,?)",
        (name, age, gender, city, ph, bg, ins_id, pol, reg.isoformat())
    )
    patient_ids.append(cur.lastrowid)
conn.commit()
print("✓ Patients inserted:", NUM_PATIENTS)

# ─── INSERT APPOINTMENTS ─────────────────────────────────────
appointment_ids = []
used_slots = set()

for _ in range(NUM_APPOINTMENTS):
    pat_id  = random.choice(patient_ids)
    doc_id  = random.randint(1, len(DOCTOR_POOL))
    dept_id = DOCTOR_POOL[doc_id - 1][2] + 1
    appt_dt = rand_date()
    slot    = random.choice(SLOT_TIMES)

    # Ensure unique slot per doctor per day
    key = (doc_id, appt_dt, slot)
    attempts = 0
    while key in used_slots and attempts < 10:
        slot = random.choice(SLOT_TIMES)
        key  = (doc_id, appt_dt, slot)
        attempts += 1
    used_slots.add(key)

    # Status (weighted)
    status = random.choices(
        ["Completed","No-Show","Cancelled","Rescheduled"],
        weights=[72, 12, 10, 6]
    )[0]

    wait_mins   = None
    consult_mins = None
    checkin     = None
    consult_s   = None
    consult_e   = None
    diag_code   = None
    diag_desc   = None

    if status == "Completed":
        wait_mins    = random.randint(5, 90)
        consult_mins = random.randint(5, 30)
        checkin      = add_mins(slot, random.randint(-5, 10))
        consult_s    = add_mins(checkin, wait_mins)
        consult_e    = add_mins(consult_s, consult_mins)
        diag         = random.choice(DIAGNOSIS_CODES)
        diag_code    = diag[0]
        diag_desc    = diag[1]

    cur.execute("""
        INSERT INTO appointments
        (patient_id,doctor_id,department_id,appointment_date,slot_time,
         check_in_time,consultation_start,consultation_end,
         wait_time_mins,consultation_mins,diagnosis_code,diagnosis_desc,status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (pat_id, doc_id, dept_id, appt_dt.isoformat(), slot,
          checkin, consult_s, consult_e,
          wait_mins, consult_mins, diag_code, diag_desc, status))
    appointment_ids.append((cur.lastrowid, pat_id, dept_id))

conn.commit()
print("✓ Appointments inserted:", NUM_APPOINTMENTS)

# ─── INSERT BILLING ──────────────────────────────────────────
bill_records = []
for appt_id, pat_id, dept_id in appointment_ids:
    # Only bill completed appointments
    cur.execute("SELECT status, appointment_date FROM appointments WHERE appointment_id=?", (appt_id,))
    row = cur.fetchone()
    if row[0] != "Completed":
        continue

    consult_fee = random.choice([200, 300, 400, 500, 600, 800, 1000, 1200])
    med_charges = round(random.uniform(0, 2000), 2)
    lab_charges = round(random.uniform(0, 3000) * random.choices([0,1],[0.4,0.6])[0], 2)
    proc_charges= round(random.uniform(0, 5000) * random.choices([0,1],[0.6,0.4])[0], 2)
    total       = round(consult_fee + med_charges + lab_charges + proc_charges, 2)

    # Get insurer
    cur.execute("SELECT insurer_id FROM patients WHERE patient_id=?", (pat_id,))
    ins_id = cur.fetchone()[0]

    insured_pct  = random.uniform(0.5, 0.9) if ins_id else 0
    insured_amt  = round(total * insured_pct, 2)
    patient_pay  = round(total - insured_amt, 2)

    pay_mode = random.choices(
        ["Cash","Card","UPI","Insurance","Mixed"],
        weights=[20, 20, 25, 25, 10]
    )[0]
    pay_status = random.choices(
        ["Paid","Pending","Partial"],
        weights=[75, 15, 10]
    )[0]

    bill_date = row[1]
    cur.execute("""
        INSERT INTO billing
        (appointment_id,patient_id,bill_date,consultation_fee,medicine_charges,
         lab_charges,procedure_charges,total_amount,insured_amount,
         patient_payable,payment_mode,payment_status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (appt_id, pat_id, bill_date, consult_fee, med_charges,
          lab_charges, proc_charges, total, insured_amt,
          patient_pay, pay_mode, pay_status))
    bill_records.append((cur.lastrowid, pat_id, ins_id, insured_amt, bill_date))

conn.commit()
print("✓ Billing records inserted:", len(bill_records))

# ─── INSERT INSURANCE CLAIMS ─────────────────────────────────
claims_count = 0
for bill_id, pat_id, ins_id, claim_amt, bill_date in bill_records:
    if claim_amt <= 0:
        continue
    if random.random() > 0.85:   # ~85% bills result in a claim
        continue

    sub_date = date.fromisoformat(bill_date) + timedelta(days=random.randint(1, 7))

    # Get insurer rejection rate
    cur.execute("SELECT rejection_rate_pct, avg_approval_days FROM insurers WHERE insurer_id=?", (ins_id,))
    ins_row = cur.fetchone()
    rejection_rate = ins_row[0] / 100
    avg_days       = ins_row[1]

    status = random.choices(
        ["Approved","Rejected","Pending","Under Review"],
        weights=[
            int((1 - rejection_rate) * 70),
            int(rejection_rate * 100),
            15,
            10
        ]
    )[0]

    approval_date  = None
    settlement_date= None
    proc_days      = None
    rej_reason     = None
    rej_code       = None
    approved_amt   = None

    if status == "Approved":
        proc_days      = random.randint(avg_days - 3, avg_days + 10)
        approval_date  = (sub_date + timedelta(days=proc_days)).isoformat()
        settlement_date= (sub_date + timedelta(days=proc_days + random.randint(2,7))).isoformat()
        approved_amt   = round(claim_amt * random.uniform(0.75, 1.0), 2)
    elif status == "Rejected":
        proc_days = random.randint(3, 21)
        rr = random.choice(REJECTION_REASONS)
        rej_code   = rr[0]
        rej_reason = rr[1]
        approved_amt = 0
    elif status == "Under Review":
        proc_days = random.randint(avg_days, avg_days + 15)

    cur.execute("""
        INSERT INTO insurance_claims
        (bill_id,patient_id,insurer_id,claim_amount,submission_date,
         approval_date,settlement_date,processing_days,claim_status,
         rejection_reason,approved_amount,rejection_code)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (bill_id, pat_id, ins_id, round(claim_amt, 2),
          sub_date.isoformat(), approval_date, settlement_date,
          proc_days, status, rej_reason, approved_amt, rej_code))
    claims_count += 1

conn.commit()
print("✓ Insurance claims inserted:", claims_count)

# ─── EXPORT TO CSV ───────────────────────────────────────────
TABLES = ["departments","doctors","insurers","patients",
          "appointments","billing","insurance_claims"]

for table in TABLES:
    cur.execute(f"SELECT * FROM {table}")
    rows    = cur.fetchall()
    headers = [desc[0] for desc in cur.description]
    path    = os.path.join(CSV_DIR, f"{table}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  ✓ Exported {table}.csv  ({len(rows)} rows)")

# ─── QUICK SUMMARY STATS ─────────────────────────────────────
print("\n" + "="*55)
print("   DATABASE SUMMARY")
print("="*55)
for table in TABLES:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    print(f"  {table:<25} {cur.fetchone()[0]:>6} rows")

cur.execute("SELECT claim_status, COUNT(*), ROUND(AVG(processing_days),1) FROM insurance_claims GROUP BY claim_status")
print("\n  CLAIM STATUS BREAKDOWN:")
for row in cur.fetchall():
    print(f"    {row[0]:<15} count={row[1]:>4}  avg_days={row[2]}")

cur.execute("SELECT ROUND(AVG(wait_time_mins),1) FROM appointments WHERE status='Completed'")
print(f"\n  Avg OPD wait time : {cur.fetchone()[0]} mins")

cur.execute("SELECT status, COUNT(*) FROM appointments GROUP BY status")
print("\n  APPOINTMENT STATUS:")
for row in cur.fetchall():
    print(f"    {row[0]:<15} {row[1]:>4}")

conn.close()
print("\n✓ opd_database.db created successfully")
print("✓ CSV exports saved in /csv_exports/")
print("="*55)
