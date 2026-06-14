-- =============================================================
-- AI OPD Management & Insurance Claim Optimizer
-- Phase 1: Database Schema
-- IBM Project | Created with Python + SQLite
-- =============================================================

-- Drop tables if they exist (for re-runs)
DROP TABLE IF EXISTS insurance_claims;
DROP TABLE IF EXISTS billing;
DROP TABLE IF EXISTS appointments;
DROP TABLE IF EXISTS doctors;
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS insurers;
DROP TABLE IF EXISTS departments;

-- =============================================================
-- TABLE 1: departments
-- =============================================================
CREATE TABLE departments (
    department_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    dept_name       TEXT NOT NULL,
    dept_code       TEXT NOT NULL UNIQUE,
    floor_no        INTEGER,
    total_beds      INTEGER
);

-- =============================================================
-- TABLE 2: doctors
-- =============================================================
CREATE TABLE doctors (
    doctor_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    doctor_name     TEXT NOT NULL,
    specialization  TEXT NOT NULL,
    department_id   INTEGER NOT NULL,
    qualification   TEXT,
    experience_yrs  INTEGER,
    availability    TEXT DEFAULT 'Mon-Sat',
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

-- =============================================================
-- TABLE 3: insurers
-- =============================================================
CREATE TABLE insurers (
    insurer_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    insurer_name    TEXT NOT NULL,
    insurer_type    TEXT CHECK(insurer_type IN ('Government','Private','Corporate')),
    avg_approval_days INTEGER,
    rejection_rate_pct REAL
);

-- =============================================================
-- TABLE 4: patients
-- =============================================================
CREATE TABLE patients (
    patient_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name    TEXT NOT NULL,
    age             INTEGER,
    gender          TEXT CHECK(gender IN ('Male','Female','Other')),
    city            TEXT,
    phone           TEXT,
    blood_group     TEXT,
    insurer_id      INTEGER,
    policy_number   TEXT,
    registration_date DATE,
    FOREIGN KEY (insurer_id) REFERENCES insurers(insurer_id)
);

-- =============================================================
-- TABLE 5: appointments
-- =============================================================
CREATE TABLE appointments (
    appointment_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id          INTEGER NOT NULL,
    doctor_id           INTEGER NOT NULL,
    department_id       INTEGER NOT NULL,
    appointment_date    DATE NOT NULL,
    slot_time           TEXT NOT NULL,
    check_in_time       TEXT,
    consultation_start  TEXT,
    consultation_end    TEXT,
    wait_time_mins      INTEGER,
    consultation_mins   INTEGER,
    diagnosis_code      TEXT,
    diagnosis_desc      TEXT,
    status              TEXT CHECK(status IN ('Completed','No-Show','Cancelled','Rescheduled')),
    FOREIGN KEY (patient_id)    REFERENCES patients(patient_id),
    FOREIGN KEY (doctor_id)     REFERENCES doctors(doctor_id),
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

-- =============================================================
-- TABLE 6: billing
-- =============================================================
CREATE TABLE billing (
    bill_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    appointment_id      INTEGER NOT NULL UNIQUE,
    patient_id          INTEGER NOT NULL,
    bill_date           DATE NOT NULL,
    consultation_fee    REAL,
    medicine_charges    REAL,
    lab_charges         REAL,
    procedure_charges   REAL,
    total_amount        REAL,
    insured_amount      REAL,
    patient_payable     REAL,
    payment_mode        TEXT CHECK(payment_mode IN ('Cash','Card','UPI','Insurance','Mixed')),
    payment_status      TEXT CHECK(payment_status IN ('Paid','Pending','Partial','Waived')),
    FOREIGN KEY (appointment_id) REFERENCES appointments(appointment_id),
    FOREIGN KEY (patient_id)     REFERENCES patients(patient_id)
);

-- =============================================================
-- TABLE 7: insurance_claims
-- =============================================================
CREATE TABLE insurance_claims (
    claim_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    bill_id             INTEGER NOT NULL,
    patient_id          INTEGER NOT NULL,
    insurer_id          INTEGER NOT NULL,
    claim_amount        REAL NOT NULL,
    submission_date     DATE NOT NULL,
    approval_date       DATE,
    settlement_date     DATE,
    processing_days     INTEGER,
    claim_status        TEXT CHECK(claim_status IN ('Approved','Rejected','Pending','Under Review')),
    rejection_reason    TEXT,
    approved_amount     REAL,
    rejection_code      TEXT,
    FOREIGN KEY (bill_id)    REFERENCES billing(bill_id),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (insurer_id) REFERENCES insurers(insurer_id)
);

-- =============================================================
-- INDEXES for query performance
-- =============================================================
CREATE INDEX idx_appointments_date     ON appointments(appointment_date);
CREATE INDEX idx_appointments_doctor   ON appointments(doctor_id);
CREATE INDEX idx_appointments_patient  ON appointments(patient_id);
CREATE INDEX idx_billing_date          ON billing(bill_date);
CREATE INDEX idx_claims_status         ON insurance_claims(claim_status);
CREATE INDEX idx_claims_insurer        ON insurance_claims(insurer_id);
CREATE INDEX idx_claims_submission     ON insurance_claims(submission_date);

-- =============================================================
-- VIEWS for analytics
-- =============================================================

-- View 1: Full appointment details
CREATE VIEW vw_appointment_details AS
SELECT
    a.appointment_id,
    a.appointment_date,
    a.slot_time,
    a.wait_time_mins,
    a.consultation_mins,
    a.status,
    a.diagnosis_code,
    a.diagnosis_desc,
    p.patient_name,
    p.age,
    p.gender,
    p.city,
    d.doctor_name,
    d.specialization,
    dep.dept_name
FROM appointments a
JOIN patients    p   ON a.patient_id    = p.patient_id
JOIN doctors     d   ON a.doctor_id     = d.doctor_id
JOIN departments dep ON a.department_id = dep.department_id;

-- View 2: Claim summary with insurer and patient info
CREATE VIEW vw_claim_summary AS
SELECT
    ic.claim_id,
    ic.claim_status,
    ic.claim_amount,
    ic.approved_amount,
    ic.processing_days,
    ic.rejection_reason,
    ic.rejection_code,
    ic.submission_date,
    ic.approval_date,
    p.patient_name,
    p.age,
    p.gender,
    ins.insurer_name,
    ins.insurer_type,
    b.total_amount,
    b.payment_mode,
    dep.dept_name,
    doc.specialization,
    ap.diagnosis_code,
    ap.diagnosis_desc
FROM insurance_claims ic
JOIN patients     p   ON ic.patient_id  = p.patient_id
JOIN insurers     ins ON ic.insurer_id  = ins.insurer_id
JOIN billing      b   ON ic.bill_id     = b.bill_id
JOIN appointments ap  ON b.appointment_id = ap.appointment_id
JOIN doctors      doc ON ap.doctor_id   = doc.doctor_id
JOIN departments  dep ON ap.department_id = dep.department_id;

-- View 3: Doctor daily workload
CREATE VIEW vw_doctor_workload AS
SELECT
    d.doctor_name,
    d.specialization,
    a.appointment_date,
    COUNT(a.appointment_id)         AS total_appointments,
    SUM(CASE WHEN a.status='Completed' THEN 1 ELSE 0 END) AS completed,
    SUM(CASE WHEN a.status='No-Show'   THEN 1 ELSE 0 END) AS no_shows,
    ROUND(AVG(a.wait_time_mins), 1) AS avg_wait_mins,
    ROUND(AVG(a.consultation_mins),1) AS avg_consult_mins
FROM appointments a
JOIN doctors d ON a.doctor_id = d.doctor_id
GROUP BY d.doctor_id, a.appointment_date;

-- View 4: Department-wise OPD flow
CREATE VIEW vw_dept_opd_flow AS
SELECT
    dep.dept_name,
    strftime('%Y-%m', a.appointment_date) AS month,
    COUNT(*)                              AS total_visits,
    ROUND(AVG(a.wait_time_mins), 1)       AS avg_wait_mins,
    SUM(CASE WHEN a.status='No-Show' THEN 1 ELSE 0 END) AS no_shows,
    ROUND(100.0 * SUM(CASE WHEN a.status='No-Show' THEN 1 ELSE 0 END) / COUNT(*), 1) AS no_show_pct
FROM appointments a
JOIN departments dep ON a.department_id = dep.department_id
GROUP BY dep.dept_name, month;
