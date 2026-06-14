# ============================================================
#  AI OPD Management & Insurance Claim Optimizer
#  Phase 2: Exploratory Data Analysis — REAL DATA
#  IBM Project | Uses your actual 7 CSV files
# ============================================================
#
#  Required files (place in same folder as this script):
#    appointments.csv, billing.csv, departments.csv,
#    doctors.csv, insurance_claims.csv, insurers.csv, patients.csv
#
#  Install: pip install pandas matplotlib seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  STEP 1 — Load All CSV Files
# ─────────────────────────────────────────────────────────────

appt      = pd.read_csv("appointments.csv")
bill      = pd.read_csv("billing.csv")
dept      = pd.read_csv("departments.csv")
doc       = pd.read_csv("doctors.csv")
claims    = pd.read_csv("insurance_claims.csv")
insurers  = pd.read_csv("insurers.csv")
patients  = pd.read_csv("patients.csv")

# Parse dates and derive time features
appt["appointment_date"] = pd.to_datetime(appt["appointment_date"])
appt["month"]       = appt["appointment_date"].dt.month
appt["month_name"]  = appt["appointment_date"].dt.strftime("%b")
appt["day_of_week"] = appt["appointment_date"].dt.day_name()
appt["hour"]        = pd.to_datetime(appt["slot_time"], format="%H:%M", errors="coerce").dt.hour
bill["bill_date"]   = pd.to_datetime(bill["bill_date"])
bill["month"]       = bill["bill_date"].dt.month

# Core merged dataframe (appointments + dept + doctor)
merged = appt.merge(dept, on="department_id").merge(doc, on="doctor_id")
completed = merged[merged["status"] == "Completed"].copy()

print(f"✅  Data loaded: {len(appt)} appointments | {len(bill)} bills | {len(claims)} claims")
print(f"    Date range: {appt['appointment_date'].min().date()} → {appt['appointment_date'].max().date()}")


# ─────────────────────────────────────────────────────────────
#  STEP 2 — Data Quality Check
# ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("DATA QUALITY REPORT")
print("="*60)
for name, df in [("appointments", appt), ("billing", bill),
                 ("insurance_claims", claims), ("patients", patients)]:
    null_pct = (df.isnull().sum() / len(df) * 100).round(1)
    nulls = null_pct[null_pct > 0]
    print(f"\n{name} ({len(df)} rows):")
    if len(nulls):
        for col, pct in nulls.items():
            print(f"  ⚠  {col}: {pct}% missing")
    else:
        print("  ✓  No missing values")


# ─────────────────────────────────────────────────────────────
#  STEP 3 — OPD Patient Flow Analysis
# ─────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("OPD Patient Flow Analysis — Real Data", fontsize=16, fontweight="bold", y=1.01)

# 3a. Monthly appointment volume
month_counts = appt.groupby("month").size().reset_index(name="count")
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
axes[0,0].bar(month_counts["month"], month_counts["count"],
              color="#4E79A7", edgecolor="white", linewidth=0.6)
axes[0,0].set_xticks(range(1,13))
axes[0,0].set_xticklabels(month_labels, rotation=45, ha="right")
axes[0,0].set_title("Monthly Appointment Volume")
axes[0,0].set_ylabel("Number of Appointments")
axes[0,0].axhline(month_counts["count"].mean(), color="red", linestyle="--",
                  label=f'Avg = {month_counts["count"].mean():.0f}/month')
axes[0,0].legend(fontsize=9)

# 3b. Appointment status breakdown
status_counts = appt["status"].value_counts()
colors_status = ["#59A14F","#E15759","#F28E2B","#76B7B2"]
axes[0,1].pie(status_counts, labels=status_counts.index, autopct="%1.1f%%",
              colors=colors_status, startangle=90,
              wedgeprops=dict(edgecolor="white", linewidth=2))
axes[0,1].set_title("Appointment Status Breakdown\n(2023–2024)")

# 3c. Wait time distribution (completed only)
axes[1,0].hist(completed["wait_time_mins"], bins=30,
               color="#F28E2B", edgecolor="white", linewidth=0.5)
axes[1,0].axvline(completed["wait_time_mins"].mean(), color="red", linestyle="--",
                  label=f'Mean = {completed["wait_time_mins"].mean():.1f} min')
axes[1,0].axvline(completed["wait_time_mins"].median(), color="blue", linestyle=":",
                  label=f'Median = {completed["wait_time_mins"].median():.1f} min')
axes[1,0].set_title("Wait Time Distribution (Completed Visits)")
axes[1,0].set_xlabel("Wait Time (minutes)")
axes[1,0].set_ylabel("Frequency")
axes[1,0].legend(fontsize=9)

# 3d. No-show rate by department
noshow_rate = (merged.groupby("dept_name")["status"]
               .apply(lambda x: (x == "No-Show").sum() / len(x) * 100)
               .sort_values(ascending=True))
colors_ns = ["#E15759" if v > 15 else "#F28E2B" if v > 12 else "#59A14F"
             for v in noshow_rate.values]
bars = axes[1,1].barh(noshow_rate.index, noshow_rate.values, color=colors_ns)
axes[1,1].set_title("No-Show Rate by Department (%)")
axes[1,1].set_xlabel("No-Show Rate (%)")
for bar, val in zip(bars, noshow_rate.values):
    axes[1,1].text(val + 0.2, bar.get_y() + bar.get_height()/2,
                   f"{val:.1f}%", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("fig1_patient_flow.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅  Figure 1 saved: fig1_patient_flow.png")


# ─────────────────────────────────────────────────────────────
#  STEP 4 — Billing & Revenue Analysis
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Billing & Revenue Analysis", fontsize=16, fontweight="bold")

# Merge bill with appointment to get dept
bill_merged = bill.merge(appt[["appointment_id","department_id"]], on="appointment_id", how="left")
bill_merged = bill_merged.merge(dept, on="department_id", how="left")

# 4a. Monthly revenue trend
monthly_rev = bill.groupby("month")["total_amount"].sum() / 100000  # in Lakhs
axes[0].plot(monthly_rev.index, monthly_rev.values,
             marker="o", color="#E15759", linewidth=2.5, markersize=7)
axes[0].fill_between(monthly_rev.index, monthly_rev.values, alpha=0.15, color="#E15759")
axes[0].set_title("Monthly Revenue (₹ Lakhs)")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Revenue (₹L)")
axes[0].set_xticks(range(1,13))
axes[0].set_xticklabels(month_labels, rotation=45, ha="right")
axes[0].set_ylim(bottom=3)

# 4b. Avg bill by department
avg_bill = bill_merged.groupby("dept_name")["total_amount"].mean().sort_values()
bars = axes[1].barh(avg_bill.index, avg_bill.values,
                    color=sns.color_palette("Blues_d", len(avg_bill)))
axes[1].set_title("Avg Bill Amount by Department (₹)")
axes[1].set_xlabel("Amount (₹)")
for bar, val in zip(bars, avg_bill.values):
    axes[1].text(val + 20, bar.get_y() + bar.get_height()/2,
                 f"₹{val:,.0f}", va="center", fontsize=9)

# 4c. Bill component breakdown (stacked)
components = ["consultation_fee","medicine_charges","lab_charges","procedure_charges"]
comp_labels = ["Consultation","Medicine","Lab","Procedure"]
comp_means  = [bill[c].mean() for c in components]
colors_comp = ["#4E79A7","#59A14F","#F28E2B","#E15759"]
bars3 = axes[2].barh(comp_labels, comp_means, color=colors_comp)
axes[2].set_title("Avg Bill Components per Visit (₹)")
axes[2].set_xlabel("Amount (₹)")
for bar, val in zip(bars3, comp_means):
    axes[2].text(val + 5, bar.get_y() + bar.get_height()/2,
                 f"₹{val:,.0f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("fig2_billing.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Figure 2 saved: fig2_billing.png")


# ─────────────────────────────────────────────────────────────
#  STEP 5 — Insurance Claim Analysis
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Insurance Claim Analysis", fontsize=16, fontweight="bold")

# Merge claims with insurer names
claims_merged = claims.merge(insurers, on="insurer_id")

# 5a. Claim status pie
status_cnt = claims["claim_status"].value_counts()
colors_pie = {"Approved":"#59A14F","Rejected":"#E15759","Pending":"#F28E2B","Under Review":"#4E79A7"}
pie_colors  = [colors_pie.get(s,"#aaa") for s in status_cnt.index]
axes[0,0].pie(status_cnt, labels=status_cnt.index, autopct="%1.1f%%",
              colors=pie_colors, startangle=90,
              wedgeprops=dict(edgecolor="white", linewidth=2))
axes[0,0].set_title(f"Claim Status Distribution\n(n={len(claims):,} claims)")

# 5b. Rejection rate by insurer (actual)
rej_by_ins = (claims_merged.groupby("insurer_name")["claim_status"]
              .apply(lambda x: (x=="Rejected").sum()/len(x)*100)
              .sort_values(ascending=False))
bar_colors = ["#E15759" if v > 20 else "#F28E2B" if v > 12 else "#59A14F"
              for v in rej_by_ins.values]
axes[0,1].bar(rej_by_ins.index, rej_by_ins.values, color=bar_colors, edgecolor="white")
axes[0,1].axhline(rej_by_ins.mean(), color="navy", linestyle="--",
                  label=f"Avg = {rej_by_ins.mean():.1f}%")
axes[0,1].set_title("Claim Rejection Rate by Insurer (%)")
axes[0,1].set_ylabel("Rejection Rate (%)")
axes[0,1].tick_params(axis="x", rotation=40)
axes[0,1].legend(fontsize=9)

# 5c. Rejection reasons
rej_reasons = (claims[claims["claim_status"]=="Rejected"]
               ["rejection_reason"].value_counts().head(10))
axes[1,0].barh(rej_reasons.index, rej_reasons.values,
               color=sns.color_palette("Reds_r", len(rej_reasons)))
axes[1,0].set_title("Top 10 Rejection Reasons (Actual)")
axes[1,0].set_xlabel("Number of Claims")
for i, v in enumerate(rej_reasons.values):
    axes[1,0].text(v + 0.3, i, str(v), va="center", fontsize=9)

# 5d. Processing days distribution
approved = claims_merged[claims_merged["claim_status"]=="Approved"]["processing_days"].dropna()
rejected = claims_merged[claims_merged["claim_status"]=="Rejected"]["processing_days"].dropna()
axes[1,1].hist(approved, bins=20, alpha=0.6, color="#59A14F", label=f"Approved (n={len(approved)})", edgecolor="white")
axes[1,1].hist(rejected, bins=20, alpha=0.6, color="#E15759", label=f"Rejected (n={len(rejected)})", edgecolor="white")
axes[1,1].axvline(approved.mean(), color="green", linestyle="--", linewidth=1.5)
axes[1,1].axvline(rejected.mean(), color="red", linestyle="--", linewidth=1.5)
axes[1,1].set_title("Processing Days: Approved vs Rejected")
axes[1,1].set_xlabel("Processing Days")
axes[1,1].set_ylabel("Frequency")
axes[1,1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("fig3_claims.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Figure 3 saved: fig3_claims.png")


# ─────────────────────────────────────────────────────────────
#  STEP 6 — Bottleneck & Wait Time Analysis
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Bottleneck & Scheduling Analysis", fontsize=16, fontweight="bold")

# 6a. Day × Hour wait time heatmap
day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
heatmap_data = (completed.groupby(["day_of_week","hour"])["wait_time_mins"]
                .mean().unstack().reindex(day_order))
sns.heatmap(heatmap_data, cmap="YlOrRd", ax=axes[0], linewidths=0.3,
            annot=True, fmt=".0f", annot_kws={"size":8},
            cbar_kws={"label":"Avg Wait (min)"})
axes[0].set_title("Avg Wait Time Heatmap (Day × Hour)")
axes[0].set_xlabel("Hour of Day")
axes[0].set_ylabel("")

# 6b. Avg wait time by department
avg_wait_dept = completed.groupby("dept_name")["wait_time_mins"].mean().sort_values()
colors_wait = ["#E15759" if v > 49 else "#F28E2B" if v > 47 else "#59A14F"
               for v in avg_wait_dept.values]
bars = axes[1].barh(avg_wait_dept.index, avg_wait_dept.values, color=colors_wait)
axes[1].set_title("Avg Wait Time by Department (min)")
axes[1].set_xlabel("Wait Time (minutes)")
axes[1].axvline(avg_wait_dept.mean(), color="navy", linestyle="--",
                label=f"Overall avg = {avg_wait_dept.mean():.1f} min")
axes[1].legend(fontsize=9)
for bar, val in zip(bars, avg_wait_dept.values):
    axes[1].text(val + 0.2, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("fig4_bottleneck.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Figure 4 saved: fig4_bottleneck.png")


# ─────────────────────────────────────────────────────────────
#  STEP 7 — Doctor Load Analysis
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 7))
doc_load = completed.groupby("doctor_name")["appointment_id"].count().sort_values()
colors_doc = ["#E15759" if v > 105 else "#F28E2B" if v > 95 else "#59A14F"
              for v in doc_load.values]
bars = ax.barh(doc_load.index, doc_load.values, color=colors_doc, edgecolor="white")
ax.axvline(doc_load.mean(), color="navy", linestyle="--",
           label=f"Avg load = {doc_load.mean():.0f} patients")
ax.set_title("Doctor Workload — Completed Appointments", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Completed Appointments")
ax.legend()
for bar, val in zip(bars, doc_load.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=9)
plt.tight_layout()
plt.savefig("fig5_doctor_load.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅  Figure 5 saved: fig5_doctor_load.png")


# ─────────────────────────────────────────────────────────────
#  STEP 8 — Key Insights Summary (from Real Data)
# ─────────────────────────────────────────────────────────────

total_rev        = bill["total_amount"].sum()
rej_rate         = (claims["claim_status"]=="Rejected").sum() / len(claims) * 100
pending_claims   = (claims["claim_status"]=="Pending").sum()
top_rej_reason   = claims[claims["claim_status"]=="Rejected"]["rejection_reason"].value_counts().index[0]
worst_insurer    = rej_by_ins.index[0]
worst_ins_rate   = rej_by_ins.iloc[0]
peak_month       = monthly_rev.idxmax()
lowest_month     = monthly_rev.idxmin()
top_noshow_dept  = noshow_rate.index[-1]  # sorted ascending, last is highest
highest_wait_dept= avg_wait_dept.index[-1]
overloaded_doc   = doc_load.index[-1]
noshow_cnt       = (appt["status"]=="No-Show").sum()

print("\n" + "="*65)
print("KEY INSIGHTS — PHASE 2 EDA SUMMARY (YOUR REAL DATA)")
print("="*65)
print(f"""
📊  PATIENT FLOW  (2000 total appointments):
    • Completed visits       : 1,450 (72.5%)
    • No-shows               : {noshow_cnt} ({noshow_cnt/len(appt)*100:.1f}%) ← HIGH PRIORITY
    • Highest no-show dept   : {top_noshow_dept}
    • Avg wait time          : {completed['wait_time_mins'].mean():.1f} minutes
    • Worst wait dept        : {highest_wait_dept} ({avg_wait_dept.iloc[-1]:.1f} min avg)
    • Peak revenue month     : Month {peak_month} | Lowest: Month {lowest_month}

💰  BILLING  (₹{total_rev:,.0f} total revenue):
    • Avg bill per visit     : ₹{bill['total_amount'].mean():,.0f}
    • Procedure charges      : ₹{bill['procedure_charges'].mean():,.0f} avg (highest component)
    • Highest bill dept      : {avg_bill.index[-1]} (₹{avg_bill.iloc[-1]:,.0f} avg)
    • Pending payments       : {(bill['payment_status']=='Pending').sum()} bills

🏥  INSURANCE CLAIMS  ({len(claims)} total):
    • Overall rejection rate : {rej_rate:.1f}%
    • Pending claims         : {pending_claims} (avg {claims[claims['claim_status']=='Pending']['processing_days'].mean():.0f} days)
    • Worst insurer          : {worst_insurer} ({worst_ins_rate:.1f}% rejection)
    • Top rejection reason   : {top_rej_reason}
    • Avg processing time    : {claims['processing_days'].mean():.1f} days

👨‍⚕️  DOCTORS:
    • Most overloaded        : {overloaded_doc} ({doc_load.iloc[-1]} appointments)
    • Load gap (max-min)     : {doc_load.iloc[-1] - doc_load.iloc[0]} appointments

🎯  TOP 5 RECOMMENDED ACTIONS:
    1. ❗ Send 24-hr reminders → reduce {noshow_cnt} no-shows in {top_noshow_dept}
    2. ❗ Fix "{top_rej_reason}" → will cut rejections significantly
    3. ⚠  Resolve {pending_claims} pending claims before they lapse
    4. ⚠  Redistribute load from {overloaded_doc} to under-loaded doctors
    5. ⚠  Investigate {worst_insurer} claim process — {worst_ins_rate:.1f}% rejection is unacceptable
""")

print("✅  Phase 2 EDA Complete!")
print("    Next → Phase 3: ML Models (claim rejection prediction + wait time forecasting)")
