from flask import Flask, request, jsonify, render_template_string, send_file, make_response
import pandas as pd
import joblib
import numpy as np
import io
import os
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

CLUSTER_LABELS = {
    0: "Ambassadeurs (Fidèles & VIP)",
    1: "Acheteurs de Gros Volume",
    2: "Clients Actifs Standards",
    3: "Nouveaux / À Risque",
}

COLS_TO_DROP = [
    'Recency', 'AccountStatus', 'RFMSegment', 'ChurnRiskCategory',
    'CustomerID', 'RegistrationDate', 'LastLoginIP', 'NewsletterSubscribed',
    'MonetaryAvg', 'TotalQuantity', 'TotalTransactions', 'Churn',
]

HIGH_CARDINALITY_THRESHOLD = 0.90


def load_resources():
    clf     = joblib.load("models/best_model_churn.pkl")
    reg     = joblib.load("models/regression_model.pkl")
    kmeans  = joblib.load("models/kmeans_model.pkl")
    sc      = joblib.load("models/scaler.pkl")
    pca_mod = joblib.load("models/pca_model.pkl")
    cols_scaler = sc.feature_names_in_.tolist()
    cols_regres = reg.feature_names_in_.tolist()
    return clf, reg, kmeans, sc, pca_mod, cols_scaler, cols_regres


clf, reg, kmeans, sc, pca_mod, cols_scaler, cols_regres = load_resources()


def drop_high_cardinality(df, threshold=HIGH_CARDINALITY_THRESHOLD):
    cols_to_drop = []
    for col in df.columns:
        if col not in ['Churn', 'MonetaryTotal']:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > threshold:
                cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop)


def preprocess_raw(df_input):
    df = df_input.copy()
    if 'RegistrationDate' in df.columns:
        df['RegistrationDate'] = pd.to_datetime(df['RegistrationDate'], dayfirst=True, errors='coerce')
        df['RegYear']  = df['RegistrationDate'].dt.year.fillna(df['RegistrationDate'].dt.year.median())
        df['RegMonth'] = df['RegistrationDate'].dt.month.fillna(df['RegistrationDate'].dt.month.median())
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], errors='ignore')
    df = drop_high_cardinality(df)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.fillna(df.median(numeric_only=True))
    return df


def run_pipeline(df_raw):
    df_clean = preprocess_raw(df_raw)
    for col in cols_scaler:
        if col not in df_clean.columns:
            df_clean[col] = 0
    df_for_scaler = df_clean[cols_scaler].fillna(0)
    X_scaled = sc.transform(df_for_scaler)
    X_pca    = pca_mod.transform(X_scaled)
    X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(10)])

    result = df_raw.copy()
    result["Churn_Pred"]    = clf.predict(X_pca_df)
    result["Churn_Proba_%"] = (clf.predict_proba(X_pca_df)[:, 1] * 100).round(1)
    result["Cluster_ID"]    = kmeans.predict(X_pca_df)
    result["Segment"]       = result["Cluster_ID"].map(CLUSTER_LABELS)

    for col in cols_regres:
        if col not in df_clean.columns:
            df_clean[col] = 0
    df_for_reg = df_clean[cols_regres].fillna(0)
    preds_raw  = reg.predict(df_for_reg)
    result["Depense_Prevue_DT"] = np.expm1(preds_raw).round(2)
    return result


def build_stats(df):
    total       = len(df)
    churn_count = int(df["Churn_Pred"].sum())
    churn_rate  = round(churn_count / total * 100, 1)
    stable_count = total - churn_count
    avg_spend   = round(float(df["Depense_Prevue_DT"].mean()), 2)
    total_spend = round(float(df["Depense_Prevue_DT"].sum()), 2)

    segments = df["Segment"].value_counts().to_dict()

    # Churn probability buckets for histogram
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
    proba_hist = {}
    for i, label in enumerate(labels):
        count = int(((df["Churn_Proba_%"] >= bins[i]) & (df["Churn_Proba_%"] < bins[i+1])).sum())
        proba_hist[label] = count

    # Spend distribution per segment
    spend_by_segment = {}
    for seg in df["Segment"].unique():
        mask = df["Segment"] == seg
        spend_by_segment[seg] = round(float(df.loc[mask, "Depense_Prevue_DT"].mean()), 2)

    # Churn by segment
    churn_by_segment = {}
    for seg in df["Segment"].unique():
        mask = df["Segment"] == seg
        total_seg = mask.sum()
        churn_seg = int(df.loc[mask, "Churn_Pred"].sum())
        churn_by_segment[seg] = round(churn_seg / total_seg * 100, 1) if total_seg > 0 else 0

    # Risk tiers
    high_risk   = int((df["Churn_Proba_%"] >= 70).sum())
    medium_risk = int(((df["Churn_Proba_%"] >= 30) & (df["Churn_Proba_%"] < 70)).sum())
    low_risk    = int((df["Churn_Proba_%"] < 30).sum())

    # Top churn customers (highest probability)
    top_churn_cols = ["Churn_Proba_%", "Segment", "Depense_Prevue_DT"]
    if "CustomerID" in df.columns:
        top_churn_cols = ["CustomerID"] + top_churn_cols
    top_churn = df.nlargest(10, "Churn_Proba_%")[top_churn_cols].to_dict("records")
    # Convert floats for JSON
    for row in top_churn:
        for k, v in row.items():
            if isinstance(v, (np.integer,)): row[k] = int(v)
            elif isinstance(v, (np.floating,)): row[k] = float(v)

    preview = df[["Churn_Pred", "Churn_Proba_%", "Segment", "Depense_Prevue_DT"]].head(15).to_dict("records")
    for row in preview:
        for k, v in row.items():
            if isinstance(v, (np.integer,)): row[k] = int(v)
            elif isinstance(v, (np.floating,)): row[k] = float(v)

    return {
        "total":             total,
        "churn_count":       churn_count,
        "stable_count":      stable_count,
        "churn_rate":        churn_rate,
        "avg_spend":         avg_spend,
        "total_spend":       total_spend,
        "segments":          segments,
        "proba_hist":        proba_hist,
        "spend_by_segment":  spend_by_segment,
        "churn_by_segment":  churn_by_segment,
        "high_risk":         high_risk,
        "medium_risk":       medium_risk,
        "low_risk":          low_risk,
        "top_churn":         top_churn,
        "preview":           preview,
    }


HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Retail Intelligence — Siwar ADALI</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {
  --bg:        #070b12;
  --surface:   #0c1220;
  --surface2:  #111827;
  --border:    #1e2d42;
  --border2:   #263548;
  --g1:        #00ffa3;
  --g2:        #0066ff;
  --danger:    #ff4757;
  --warn:      #ffa502;
  --info:      #1e90ff;
  --safe:      #2ed573;
  --purple:    #a855f7;
  --text:      #e8f0fe;
  --text2:     #94a3b8;
  --text3:     #4a6080;
  --radius:    16px;
  --font:      'Outfit', sans-serif;
  --mono:      'JetBrains Mono', monospace;
  --shadow:    0 8px 32px rgba(0,0,0,0.4);
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── animated background ── */
body::before {
  content: '';
  position: fixed; inset: 0;
  background:
    radial-gradient(ellipse 80% 50% at 80% -10%, rgba(0,102,255,.12) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at -10% 80%, rgba(0,255,163,.08) 0%, transparent 60%);
  pointer-events: none; z-index: 0;
}
body::after {
  content: '';
  position: fixed; inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.015'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  pointer-events: none; z-index: 0;
}
.kpi-risk-advanced {
  grid-column: span 2; /* makes it bigger */
  padding: 28px;
}

.risk-main {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  margin-bottom: 14px;
}

.risk-total {
  font-size: 2.8rem;
  font-weight: 900;
  background: linear-gradient(90deg, var(--danger), var(--warn), var(--safe));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.risk-desc {
  font-family: var(--mono);
  font-size: .7rem;
  color: var(--text3);
}

/* progress stacked bar */
.risk-progress {
  display: flex;
  height: 8px;
  border-radius: 99px;
  overflow: hidden;
  margin: 14px 0 18px;
  background: var(--border);
}

.risk-progress-high { background: var(--danger); }
.risk-progress-med  { background: var(--warn); }
.risk-progress-low  { background: var(--safe); }

/* details */
.risk-details {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.risk-line {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: var(--mono);
  font-size: .75rem;
}

.risk-label { color: var(--text2); }

.risk-value {
  font-weight: 700;
}

.risk-pct {
  color: var(--text3);
}

main { position: relative; z-index: 1; max-width: 1240px; margin: 0 auto; padding: 40px 24px 100px; }

/* ── HEADER ── */
header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 48px; flex-wrap: wrap; gap: 20px;
}
.brand-wrap { display: flex; align-items: center; gap: 20px; }
.brand-icon {
  width: 52px; height: 52px;
  background: linear-gradient(135deg, var(--g1), var(--g2));
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem;
  box-shadow: 0 0 24px rgba(0,255,163,.3);
  flex-shrink: 0;
}
.brand { display: flex; flex-direction: column; gap: 3px; }
.brand-title {
  font-size: clamp(1.5rem, 3.5vw, 2.2rem);
  font-weight: 800;
  letter-spacing: -.03em;
  color: var(--text);
}
.brand-title span {
  background: linear-gradient(90deg, var(--g1), var(--g2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.brand-sub { font-family: var(--mono); font-size: .68rem; color: var(--text3); letter-spacing: .12em; text-transform: uppercase; }
.header-right { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.badge {
  font-family: var(--mono); font-size: .68rem; font-weight: 600;
  padding: 7px 14px; border-radius: 8px; letter-spacing: .06em;
}
.badge-green { color: var(--g1); border: 1px solid rgba(0,255,163,.3); background: rgba(0,255,163,.07); }
.badge-blue  { color: var(--g2); border: 1px solid rgba(0,102,255,.3); background: rgba(0,102,255,.07); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--g1); animation: pulse 2s infinite; display: inline-block; margin-right: 6px; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.6;transform:scale(1.3)} }

/* ── UPLOAD CARD ── */
.upload-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 40px;
  margin-bottom: 32px;
  position: relative; overflow: hidden;
  box-shadow: var(--shadow);
}
.upload-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--g1), var(--g2), var(--purple));
}
.section-label {
  font-family: var(--mono); font-size: .67rem; letter-spacing: .14em;
  color: var(--text3); text-transform: uppercase; margin-bottom: 20px;
  display: flex; align-items: center; gap: 8px;
}
.section-label::before { content: ''; width: 16px; height: 1px; background: var(--g1); }

.drop-zone {
  border: 2px dashed var(--border2); border-radius: 14px;
  padding: 52px 24px; text-align: center; cursor: pointer;
  transition: all .3s ease; position: relative;
  background: rgba(255,255,255,.01);
}
.drop-zone:hover, .drop-zone.drag-over {
  border-color: var(--g1);
  background: rgba(0,255,163,.04);
  box-shadow: inset 0 0 40px rgba(0,255,163,.05);
}
.drop-zone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%; }
.drop-icon {
  width: 64px; height: 64px; margin: 0 auto 18px;
  background: linear-gradient(135deg, rgba(0,255,163,.12), rgba(0,102,255,.12));
  border-radius: 16px; display: flex; align-items: center; justify-content: center;
  font-size: 1.8rem; border: 1px solid rgba(0,255,163,.15);
}
.drop-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 6px; }
.drop-hint  { font-family: var(--mono); font-size: .72rem; color: var(--text3); }
.file-tag {
  display: inline-flex; align-items: center; gap: 8px;
  margin-top: 16px; padding: 8px 16px; border-radius: 8px;
  background: rgba(0,255,163,.08); border: 1px solid rgba(0,255,163,.2);
  font-family: var(--mono); font-size: .78rem; color: var(--g1);
}

.btn-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 24px; align-items: center; }
.btn {
  display: inline-flex; align-items: center; gap: 10px;
  background: linear-gradient(135deg, var(--g1), var(--g2));
  color: #000; font-family: var(--font); font-size: .88rem; font-weight: 700;
  letter-spacing: .02em; padding: 14px 32px; border: none; border-radius: 10px;
  cursor: pointer; transition: all .25s ease;
  box-shadow: 0 4px 20px rgba(0,255,163,.25);
}
.btn:hover   { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(0,255,163,.35); }
.btn:active  { transform: translateY(0); }
.btn:disabled { opacity: .35; cursor: not-allowed; transform: none; box-shadow: none; }
.btn-ghost {
  display: inline-flex; align-items: center; gap: 8px;
  background: transparent; color: var(--text2);
  border: 1px solid var(--border2); font-family: var(--font);
  font-size: .82rem; font-weight: 500; padding: 13px 22px;
  border-radius: 10px; cursor: pointer; transition: all .2s ease;
}
.btn-ghost:hover { border-color: var(--g1); color: var(--g1); background: rgba(0,255,163,.05); }

.loader { display: none; align-items: center; gap: 14px; margin-top: 20px; }
.spinner {
  width: 20px; height: 20px;
  border: 2px solid var(--border2); border-top-color: var(--g1);
  border-radius: 50%; animation: spin .8s linear infinite; flex-shrink: 0;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loader-text { font-family: var(--mono); font-size: .78rem; color: var(--text3); }
.loader-steps { display: flex; gap: 6px; margin-top: 10px; flex-wrap: wrap; }
.step-pill {
  font-family: var(--mono); font-size: .62rem; padding: 4px 10px; border-radius: 99px;
  background: var(--surface2); border: 1px solid var(--border); color: var(--text3);
  transition: all .3s; letter-spacing: .06em;
}
.step-pill.active { background: rgba(0,255,163,.1); border-color: var(--g1); color: var(--g1); }
.step-pill.done   { background: rgba(0,255,163,.06); border-color: rgba(0,255,163,.3); color: rgba(0,255,163,.6); }

.error-box {
  background: rgba(255,71,87,.07); border: 1px solid rgba(255,71,87,.3);
  border-radius: 12px; padding: 16px 20px;
  font-family: var(--mono); font-size: .8rem; color: #ff8a94;
  display: none; margin-top: 16px;
}

/* ── RESULTS ── */
#results { display: none; }
.results-header {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 28px; flex-wrap: wrap; gap: 12px;
}
.results-title { font-size: 1.1rem; font-weight: 700; }
.results-meta  { font-family: var(--mono); font-size: .72rem; color: var(--text3); }

/* ── KPI GRID ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 16px; margin-bottom: 28px;
}
.kpi {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 24px 22px;
  position: relative; overflow: hidden;
  transition: all .3s ease; cursor: default;
}
.kpi:hover { border-color: var(--border2); transform: translateY(-2px); box-shadow: 0 12px 32px rgba(0,0,0,.3); }
.kpi-label { font-family: var(--mono); font-size: .67rem; color: var(--text3); letter-spacing: .1em; text-transform: uppercase; margin-bottom: 14px; font-weight: 500; }
.kpi-value { font-size: 2.4rem; font-weight: 800; letter-spacing: -.03em; line-height: 1; }
.kpi-sub   { font-family: var(--mono); font-size: .72rem; color: var(--text3); margin-top: 8px; }
.kpi-bar   { position: absolute; bottom: 0; left: 0; right: 0; height: 3px; }
.kpi.c-total  .kpi-value { color: var(--info); }
.kpi.c-churn  .kpi-value { color: var(--danger); }
.kpi.c-safe   .kpi-value { color: var(--safe); }
.kpi.c-spend  .kpi-value { color: var(--g1); }
.kpi.c-total2 .kpi-value { color: var(--warn); }
.kpi.c-high   .kpi-value { color: var(--danger); }
.kpi.c-med    .kpi-value { color: var(--warn); }
.kpi.c-low    .kpi-value { color: var(--safe); }

/* risk tier badge inside kpi */
.kpi-risk-row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
.risk-chip {
  font-family: var(--mono); font-size: .62rem; padding: 4px 10px;
  border-radius: 6px; font-weight: 600; letter-spacing: .04em;
}
.risk-high   { background: rgba(255,71,87,.12);  color: #ff8a94; }
.risk-medium { background: rgba(255,165,2,.12);  color: #ffc342; }
.risk-low    { background: rgba(46,213,115,.12); color: #73e0a3; }

/* ── TABS ── */
.tabs-wrap { margin-bottom: 28px; }
.tabs {
  display: flex; gap: 4px; flex-wrap: wrap;
  background: var(--surface); border: 1px solid var(--border);
  padding: 6px; border-radius: 12px; width: fit-content;
}
.tab {
  font-family: var(--font); font-size: .82rem; font-weight: 600;
  padding: 10px 20px; border-radius: 8px; cursor: pointer;
  color: var(--text3); background: transparent;
  border: none; transition: all .2s ease; letter-spacing: .01em;
  display: flex; align-items: center; gap: 8px;
}
.tab:hover { color: var(--text2); background: var(--surface2); }
.tab.active { background: linear-gradient(135deg, var(--g1), var(--g2)); color: #000; }
.tab-panel { display: none; animation: fadeIn .3s ease; }
.tab-panel.active { display: block; }
@keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }

/* ── CARDS ── */
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 28px; margin-bottom: 20px;
  box-shadow: var(--shadow);
}
.card-title { font-size: .95rem; font-weight: 700; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
.card-title .dot { width: 10px; height: 10px; border-radius: 50%; }

/* ── CHART GRID ── */
.chart-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
.chart-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px; }
@media(max-width:900px) { .chart-grid-2,.chart-grid-3 { grid-template-columns: 1fr; } }
.chart-wrap { position: relative; width: 100%; }
canvas { max-width: 100%; }

/* ── SEGMENT BARS ── */
.seg-row { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; }
.seg-row:last-child { margin-bottom: 0; }
.seg-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.seg-name { font-size: .85rem; font-weight: 600; min-width: 230px; color: var(--text2); }
.seg-bar-wrap { flex: 1; height: 8px; background: var(--border); border-radius: 99px; overflow: hidden; }
.seg-bar { height: 100%; border-radius: 99px; transition: width 1.2s cubic-bezier(.4,0,.2,1); }
.seg-pct  { font-family: var(--mono); font-size: .7rem; color: var(--text3); min-width: 38px; text-align: right; }
.seg-count{ font-family: var(--mono); font-size: .7rem; color: var(--text2); font-weight: 600; min-width: 45px; text-align: right; }

/* ── TABLE ── */
.table-wrap { overflow-x: auto; border-radius: 12px; }
table { width: 100%; border-collapse: collapse; font-size: .83rem; }
thead th {
  font-family: var(--mono); font-size: .67rem; color: var(--text3);
  letter-spacing: .1em; text-transform: uppercase;
  padding: 14px 18px; text-align: left;
  border-bottom: 2px solid var(--border); font-weight: 600;
  white-space: nowrap; background: var(--surface2);
}
thead th:first-child { border-radius: 10px 0 0 0; }
thead th:last-child  { border-radius: 0 10px 0 0; }
tbody td { padding: 13px 18px; border-bottom: 1px solid rgba(255,255,255,.04); white-space: nowrap; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: rgba(0,255,163,.025); }
.pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 4px 12px; border-radius: 6px;
  font-family: var(--mono); font-size: .7rem; font-weight: 700;
}
.pill-danger { background: rgba(255,71,87,.12); color: #ff8a94; }
.pill-safe   { background: rgba(46,213,115,.12); color: #73e0a3; }
.pill-warn   { background: rgba(255,165,2,.12); color: #ffc342; }
.mono { font-family: var(--mono); }
.col-g1   { color: var(--g1); }
.col-danger { color: #ff8a94; }
.col-warn { color: #ffc342; }
.col-muted { color: var(--text3); }
.col-text2 { color: var(--text2); }

/* ── RISK GAUGE ── */
.gauge-row { display: flex; gap: 16px; margin-bottom: 20px; }
.gauge-item {
  flex: 1; background: var(--surface2); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px 16px; text-align: center;
}
.gauge-value { font-size: 2rem; font-weight: 800; letter-spacing: -.02em; }
.gauge-label { font-family: var(--mono); font-size: .65rem; color: var(--text3); text-transform: uppercase; letter-spacing: .1em; margin-top: 6px; }
.gauge-bar-wrap { height: 6px; background: var(--border); border-radius: 99px; margin-top: 12px; overflow: hidden; }
.gauge-bar { height: 100%; border-radius: 99px; transition: width 1.2s ease; }

/* ── FOOTER ── */
footer {
  position: relative; z-index: 1;
  margin-top: 80px; padding-top: 24px;
  border-top: 1px solid var(--border);
  font-family: var(--mono); font-size: .7rem; color: var(--text3);
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;
  letter-spacing: .04em;
}

/* ── PROGRESS BAR ── */
.progress-track { height: 4px; background: var(--border); border-radius: 99px; overflow: hidden; margin-top: 8px; }
.progress-fill  { height: 100%; border-radius: 99px; transition: width .5s ease; background: linear-gradient(90deg, var(--g1), var(--g2)); }

@media(max-width:600px){
  .kpi-grid { grid-template-columns: 1fr 1fr; }
  .gauge-row { flex-direction: column; }
  .seg-name { min-width: 130px; }
  header { flex-direction: column; align-items: flex-start; }
}
</style>
</head>
<body>
<main>
  <!-- HEADER -->
  <header>
    <div class="brand-wrap">
      <div class="brand-icon">📡</div>
      <div class="brand">
        <div class="brand-title">Retail <span>Intelligence</span></div>
        <div class="brand-sub">Système de Prédiction 360° — Siwar ADALI · ENIS GI2</div>
      </div>
    </div>
    <div class="header-right">
      <span class="badge badge-green"><span class="status-dot"></span>Modèles actifs</span>
      <span class="badge badge-blue">v2.0 · Flask ML</span>
    </div>
  </header>

  <!-- UPLOAD -->
  <div class="upload-card">
    <div class="section-label">Analyse de données clients</div>
    <div class="drop-zone" id="dropZone">
      <input type="file" id="csvFile" accept=".csv"/>
      <div class="drop-icon">🗂</div>
      <div class="drop-title">Déposer votre fichier CSV ici</div>
      <div class="drop-hint">ou cliquer pour parcourir — format .csv uniquement</div>
      <div id="fileTag" style="display:none" class="file-tag">
        <span>📎</span><span id="fileName"></span>
      </div>
    </div>

    <div class="btn-row">
      <button class="btn" id="predictBtn" disabled onclick="runPredict()">
        ▶ &nbsp;Lancer la prédiction
      </button>
      <button class="btn-ghost" id="resetBtn" onclick="resetAll()" style="display:none">↩ Réinitialiser</button>
    </div>

    <div class="loader" id="loader">
      <div class="spinner"></div>
      <div>
        <div class="loader-text" id="loaderText">Traitement en cours…</div>
        <div class="loader-steps" id="loaderSteps">
          <span class="step-pill" id="sp0">Encodage</span>
          <span class="step-pill" id="sp1">Scaling</span>
          <span class="step-pill" id="sp2">PCA</span>
          <span class="step-pill" id="sp3">Churn</span>
          <span class="step-pill" id="sp4">Clustering</span>
          <span class="step-pill" id="sp5">Régression</span>
        </div>
      </div>
    </div>
    <div class="error-box" id="errorBox"></div>
  </div>

  <!-- RESULTS -->
  <div id="results">

    <div class="results-header">
      <div>
        <div class="results-title">Tableau de bord — Analyse complète</div>
        <div class="results-meta" id="resMeta"></div>
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <button class="btn-ghost" onclick="downloadCSV()">⬇ Exporter CSV</button>
        <button class="btn-ghost" onclick="downloadJSON()">⬇ Exporter JSON</button>
      </div>
    </div>

    <!-- KPIs -->
    <div class="kpi-grid" id="kpiGrid">
      <div class="kpi c-total">
        <div class="kpi-label">Clients analysés</div>
        <div class="kpi-value" id="kTotal">—</div>
        <div class="kpi-sub">fichier complet</div>
        <div class="kpi-bar" style="background:linear-gradient(90deg,var(--info),var(--g2))"></div>
      </div>
      <div class="kpi c-churn">
        <div class="kpi-label">Churn détecté</div>
        <div class="kpi-value" id="kChurn">—</div>
        <div class="kpi-sub" id="kChurnRate"></div>
        <div class="kpi-bar" style="background:var(--danger)"></div>
      </div>
      <div class="kpi c-safe">
        <div class="kpi-label">Clients stables</div>
        <div class="kpi-value" id="kStable">—</div>
        <div class="kpi-sub" id="kStableRate"></div>
        <div class="kpi-bar" style="background:var(--safe)"></div>
      </div>
      <div class="kpi c-spend">
        <div class="kpi-label">Dépense moyenne</div>
        <div class="kpi-value" id="kAvg">—</div>
        <div class="kpi-sub">DT / client</div>
        <div class="kpi-bar" style="background:var(--g1)"></div>
      </div>
      <div class="kpi c-total2">
        <div class="kpi-label">Potentiel total</div>
        <div class="kpi-value" id="kTotal2">—</div>
        <div class="kpi-sub">DT estimé</div>
        <div class="kpi-bar" style="background:var(--warn)"></div>
      </div>
      <div class="kpi kpi-risk-advanced">
  <div class="kpi-label">Niveaux de risque</div>

  <!-- BIG GLOBAL INDICATOR -->
  <div class="risk-main">
    <div class="risk-total" id="kRiskTotal">—</div>
    <div class="risk-desc">clients analysés</div>
  </div>

  <!-- PROGRESS BAR GLOBAL -->
  <div class="risk-progress">
    <div class="risk-progress-high" id="riskBarHigh"></div>
    <div class="risk-progress-med" id="riskBarMed"></div>
    <div class="risk-progress-low" id="riskBarLow"></div>
  </div>

  <!-- DETAILED ROWS -->
  <div class="risk-details">

    <div class="risk-line">
      <span class="risk-label">🔴 Élevé</span>
      <span class="risk-value" id="kHigh">—</span>
      <span class="risk-pct" id="kHighPct">—%</span>
    </div>

    <div class="risk-line">
      <span class="risk-label">🟠 Moyen</span>
      <span class="risk-value" id="kMed">—</span>
      <span class="risk-pct" id="kMedPct">—%</span>
    </div>

    <div class="risk-line">
      <span class="risk-label">🟢 Faible</span>
      <span class="risk-value" id="kLow">—</span>
      <span class="risk-pct" id="kLowPct">—%</span>
    </div>

  </div>

  <div class="kpi-sub">distribution intelligente du churn</div>
</div>
    </div>

    <!-- TABS -->
    <div class="tabs-wrap">
      <div class="tabs">
        <button class="tab active" onclick="switchTab('overview')">📊 Vue d'ensemble</button>
        <button class="tab" onclick="switchTab('churn')">🎯 Analyse Churn</button>
        <button class="tab" onclick="switchTab('segments')">🔵 Segments</button>
        <button class="tab" onclick="switchTab('table')">📋 Données</button>
      </div>
    </div>

    <!-- TAB: Overview -->
    <div class="tab-panel active" id="tab-overview">
      <div class="chart-grid-2">
        <div class="card">
          <div class="card-title"><span class="dot" style="background:var(--g1)"></span>Répartition Churn vs Stable</div>
          <div class="chart-wrap" style="height:260px"><canvas id="chartDonut"></canvas></div>
        </div>
        <div class="card">
          <div class="card-title"><span class="dot" style="background:var(--g2)"></span>Dépense moyenne par segment</div>
          <div class="chart-wrap" style="height:260px"><canvas id="chartSpendSeg"></canvas></div>
        </div>
      </div>
      <div class="card">
        <div class="card-title"><span class="dot" style="background:var(--warn)"></span>Répartition des segments clients</div>
        <div id="segBars"></div>
      </div>
    </div>

    <!-- TAB: Churn Analysis -->
    <div class="tab-panel" id="tab-churn">
      <div class="chart-grid-2">
        <div class="card">
          <div class="card-title"><span class="dot" style="background:var(--danger)"></span>Distribution probabilité de churn</div>
          <div class="chart-wrap" style="height:280px"><canvas id="chartProbaHist"></canvas></div>
        </div>
        <div class="card">
          <div class="card-title"><span class="dot" style="background:var(--purple)"></span>Taux de churn par segment</div>
          <div class="chart-wrap" style="height:280px"><canvas id="chartChurnSeg"></canvas></div>
        </div>
      </div>

      <!-- Risk gauge -->
      <div class="card">
        <div class="card-title"><span class="dot" style="background:var(--warn)"></span>Répartition des niveaux de risque</div>
        <div class="gauge-row" id="gaugeRow"></div>
      </div>

      <!-- Top at-risk table -->
      <div class="card">
        <div class="card-title"><span class="dot" style="background:var(--danger)"></span>Top 10 clients à risque élevé</div>
        <div class="table-wrap">
          <table>
            <thead><tr>
              <th>#</th>
              <th id="topChurnIdCol" style="display:none">ID Client</th>
              <th>Probabilité Churn</th>
              <th>Segment</th>
              <th>Dépense prévue (DT)</th>
              <th>Action recommandée</th>
            </tr></thead>
            <tbody id="topChurnTable"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- TAB: Segments -->
    <div class="tab-panel" id="tab-segments">
      <div class="chart-grid-2">
        <div class="card">
          <div class="card-title"><span class="dot" style="background:var(--g2)"></span>Distribution des segments</div>
          <div class="chart-wrap" style="height:300px"><canvas id="chartSegPie"></canvas></div>
        </div>
        <div class="card">
          <div class="card-title"><span class="dot" style="background:var(--warn)"></span>Churn vs Stable par segment</div>
          <div class="chart-wrap" style="height:300px"><canvas id="chartSegStacked"></canvas></div>
        </div>
      </div>
    </div>

    <!-- TAB: Data Table -->
    <div class="tab-panel" id="tab-table">
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;flex-wrap:wrap;gap:12px">
          <div class="card-title" style="margin-bottom:0"><span class="dot" style="background:var(--g1)"></span>Aperçu — 15 premiers clients</div>
          <div style="font-family:var(--mono);font-size:.7rem;color:var(--text3)">Téléchargez le CSV pour le dataset complet</div>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr>
              <th>#</th>
              <th>Statut Churn</th>
              <th>Prob. Churn</th>
              <th>Segment</th>
              <th>Dépense prévue (DT)</th>
            </tr></thead>
            <tbody id="previewTable"></tbody>
          </table>
        </div>
      </div>
    </div>

  </div><!-- /results -->
</main>

<footer>
  <span>Retail Intelligence · Siwar ADALI — ENIS GI2 · 2024</span>
  <span id="footerTime"></span>
</footer>

<script>
// ── state ──────────────────────────────────────────────────────
let lastBlob = null, lastStats = null;
const charts = {};

// ── DOM refs ───────────────────────────────────────────────────
const dropZone   = document.getElementById('dropZone');
const csvFile    = document.getElementById('csvFile');
const predictBtn = document.getElementById('predictBtn');
const resetBtn   = document.getElementById('resetBtn');
const loader     = document.getElementById('loader');
const loaderText = document.getElementById('loaderText');
const errorBox   = document.getElementById('errorBox');
const results    = document.getElementById('results');

// ── clock ──────────────────────────────────────────────────────
(function tick(){ document.getElementById('footerTime').textContent = new Date().toLocaleString('fr-FR'); setTimeout(tick,1000); })();

// ── drag & drop ────────────────────────────────────────────────
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.name.endsWith('.csv')) { csvFile.files = e.dataTransfer.files; onFileSelected(f); }
});
csvFile.addEventListener('change', () => { if (csvFile.files[0]) onFileSelected(csvFile.files[0]); });

function onFileSelected(f) {
  document.getElementById('fileName').textContent = f.name + ' (' + (f.size/1024).toFixed(1) + ' KB)';
  document.getElementById('fileTag').style.display = 'inline-flex';
  predictBtn.disabled = false;
  results.style.display = 'none';
  errorBox.style.display = 'none';
  lastBlob = null;
}

// ── loader steps ───────────────────────────────────────────────
const STEPS = ['Encodage','Scaling','PCA','Churn','Clustering','Régression'];
let stepInterval, currentStep = 0;

function startSteps() {
  currentStep = 0;
  STEPS.forEach((_,i) => { document.getElementById('sp'+i).className = 'step-pill'; });
  stepInterval = setInterval(() => {
    if (currentStep > 0) document.getElementById('sp'+(currentStep-1)).className = 'step-pill done';
    if (currentStep < STEPS.length) {
      document.getElementById('sp'+currentStep).className = 'step-pill active';
      loaderText.textContent = STEPS[currentStep] + '…';
      currentStep++;
    }
  }, 600);
}
function stopSteps() {
  clearInterval(stepInterval);
  STEPS.forEach((_,i) => { document.getElementById('sp'+i).className = 'step-pill done'; });
}

// ── predict ────────────────────────────────────────────────────
async function runPredict() {
  if (!csvFile.files[0]) return;
  predictBtn.disabled = true;
  loader.style.display = 'flex';
  results.style.display = 'none';
  errorBox.style.display = 'none';
  lastBlob = null;
  startSteps();

  try {
    const fd = new FormData();
    fd.append('file', csvFile.files[0]);
    const resp = await fetch('/predict', { method: 'POST', body: fd });

    stopSteps(); loader.style.display = 'none';

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: 'Erreur inconnue' }));
      showError(err.error || 'Erreur serveur'); predictBtn.disabled = false; return;
    }

    const statsRaw = resp.headers.get('X-Stats');
    lastBlob = await resp.blob();
    if (statsRaw) { lastStats = JSON.parse(statsRaw); renderResults(lastStats); }
    predictBtn.disabled = false;
    resetBtn.style.display = 'inline-flex';

  } catch(e) {
    stopSteps(); loader.style.display = 'none';
    showError('Impossible de joindre le serveur : ' + e.message);
    predictBtn.disabled = false;
  }
}

// ── render ─────────────────────────────────────────────────────
const SEG_COLORS = {
  "Ambassadeurs (Fidèles & VIP)": "#2ed573",
  "Acheteurs de Gros Volume":     "#1e90ff",
  "Clients Actifs Standards":     "#ffa502",
  "Nouveaux / À Risque":          "#ff4757",
};

function renderResults(s) {
  // meta
  document.getElementById('resMeta').textContent =
    'Analysé le ' + new Date().toLocaleString('fr-FR') + ' · ' + s.total.toLocaleString('fr-FR') + ' clients';

  // KPIs
  document.getElementById('kTotal').textContent  = fmt(s.total);
  document.getElementById('kChurn').textContent  = fmt(s.churn_count);
  document.getElementById('kChurnRate').textContent  = s.churn_rate + '% du fichier';
  document.getElementById('kStable').textContent = fmt(s.stable_count);
  document.getElementById('kStableRate').textContent = (100 - s.churn_rate).toFixed(1) + '% du fichier';
  document.getElementById('kAvg').textContent    = fmtDT(s.avg_spend);
  document.getElementById('kTotal2').textContent = fmtDT(s.total_spend);
  const total = s.total;

document.getElementById('kRiskTotal').textContent = fmt(total);

// values
document.getElementById('kHigh').textContent = fmt(s.high_risk);
document.getElementById('kMed').textContent  = fmt(s.medium_risk);
document.getElementById('kLow').textContent  = fmt(s.low_risk);

// percentages
const highPct = (s.high_risk / total * 100).toFixed(1);
const medPct  = (s.medium_risk / total * 100).toFixed(1);
const lowPct  = (s.low_risk / total * 100).toFixed(1);

document.getElementById('kHighPct').textContent = highPct + '%';
document.getElementById('kMedPct').textContent  = medPct + '%';
document.getElementById('kLowPct').textContent  = lowPct + '%';

// animated stacked bar
document.getElementById('riskBarHigh').style.width = highPct + '%';
document.getElementById('riskBarMed').style.width  = medPct + '%';
document.getElementById('riskBarLow').style.width  = lowPct + '%';

  // destroy old charts
  Object.values(charts).forEach(c => c.destroy());
  Object.keys(charts).forEach(k => delete charts[k]);

  buildDonut(s);
  buildSpendSeg(s);
  buildSegBars(s);
  buildProbaHist(s);
  buildChurnSeg(s);
  buildGauge(s);
  buildTopChurn(s);
  buildSegPie(s);
  buildSegStacked(s);
  buildTable(s);

  results.style.display = 'block';
  results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── CHART BUILDERS ─────────────────────────────────────────────
const CHART_DEFAULTS = {
  color: '#94a3b8',
  font: { family: "'JetBrains Mono', monospace", size: 11 },
};
Chart.defaults.color = CHART_DEFAULTS.color;
Chart.defaults.font  = CHART_DEFAULTS.font;

function buildDonut(s) {
  charts.donut = new Chart(document.getElementById('chartDonut'), {
    type: 'doughnut',
    data: {
      labels: ['Stable', 'Churn'],
      datasets: [{ data: [s.stable_count, s.churn_count],
        backgroundColor: ['rgba(46,213,115,.8)', 'rgba(255,71,87,.8)'],
        borderColor: ['#2ed573', '#ff4757'], borderWidth: 2,
        hoverOffset: 8 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '72%',
      plugins: {
        legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true, pointStyleWidth: 10 } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed.toLocaleString('fr-FR')} (${(ctx.parsed/s.total*100).toFixed(1)}%)` } }
      }
    }
  });
}

function buildSpendSeg(s) {
  const labels = Object.keys(s.spend_by_segment);
  const data   = Object.values(s.spend_by_segment);
  const colors = labels.map(l => SEG_COLORS[l] || '#94a3b8');
  charts.spendSeg = new Chart(document.getElementById('chartSpendSeg'), {
    type: 'bar',
    data: { labels: labels.map(l => l.length > 20 ? l.slice(0,20)+'…' : l),
      datasets: [{ label: 'DT moyen', data, backgroundColor: colors.map(c => c+'44'), borderColor: colors, borderWidth: 2, borderRadius: 8 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,.04)' }, ticks: { callback: v => fmtDT(v) } },
        y: { grid: { display: false } }
      }
    }
  });
}

function buildSegBars(s) {
  const el = document.getElementById('segBars');
  el.innerHTML = '';
  const total = s.total;
  for (const [name, count] of Object.entries(s.segments)) {
    const pct   = (count / total * 100).toFixed(1);
    const color = SEG_COLORS[name] || '#94a3b8';
    el.innerHTML += `
      <div class="seg-row">
        <span class="seg-dot" style="background:${color}"></span>
        <div class="seg-name">${name}</div>
        <div class="seg-bar-wrap"><div class="seg-bar" style="width:0%;background:${color}" data-w="${pct}%"></div></div>
        <div class="seg-pct">${pct}%</div>
        <div class="seg-count">${count.toLocaleString('fr-FR')}</div>
      </div>`;
  }
  requestAnimationFrame(() => {
    document.querySelectorAll('.seg-bar').forEach(b => { b.style.width = b.dataset.w; });
  });
}

function buildProbaHist(s) {
  const labels = Object.keys(s.proba_hist);
  const data   = Object.values(s.proba_hist);
  const colors = labels.map(l => {
    const v = parseInt(l); 
    if(v >= 70) return 'rgba(255,71,87,.8)';
    if(v >= 30) return 'rgba(255,165,2,.8)';
    return 'rgba(46,213,115,.8)';
  });
  charts.probaHist = new Chart(document.getElementById('chartProbaHist'), {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Clients', data, backgroundColor: colors, borderRadius: 6, borderSkipped: false }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { title: t => 'Prob. ' + t[0].label + '%', label: ctx => ` ${ctx.parsed.y.toLocaleString('fr-FR')} clients` } }
      },
      scales: {
        x: { grid: { display: false }, title: { display: true, text: 'Tranche de probabilité (%)' } },
        y: { grid: { color: 'rgba(255,255,255,.04)' }, title: { display: true, text: 'Nb clients' } }
      }
    }
  });
}

function buildChurnSeg(s) {
  const labels = Object.keys(s.churn_by_segment);
  const data   = Object.values(s.churn_by_segment);
  const colors = labels.map(l => SEG_COLORS[l] || '#94a3b8');
  charts.churnSeg = new Chart(document.getElementById('chartChurnSeg'), {
    type: 'bar',
    data: { labels: labels.map(l => l.length>18?l.slice(0,18)+'…':l),
      datasets: [{ label: 'Taux churn (%)', data, backgroundColor: colors.map(c=>c+'55'), borderColor: colors, borderWidth: 2, borderRadius: 8 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: 'rgba(255,255,255,.04)' }, max: 100,
          ticks: { callback: v => v + '%' }
        }
      }
    }
  });
}

function buildGauge(s) {
  const el = document.getElementById('gaugeRow');
  const total = s.total;
  const items = [
    { label: 'Risque Élevé',  value: s.high_risk,   color: '#ff4757', pct: (s.high_risk/total*100).toFixed(1) },
    { label: 'Risque Moyen',  value: s.medium_risk, color: '#ffa502', pct: (s.medium_risk/total*100).toFixed(1) },
    { label: 'Risque Faible', value: s.low_risk,    color: '#2ed573', pct: (s.low_risk/total*100).toFixed(1) },
  ];
  el.innerHTML = items.map(item => `
    <div class="gauge-item">
      <div class="gauge-value" style="color:${item.color}">${item.value.toLocaleString('fr-FR')}</div>
      <div class="gauge-label">${item.label}</div>
      <div class="gauge-bar-wrap"><div class="gauge-bar" style="width:0%;background:${item.color}" data-w="${item.pct}%"></div></div>
      <div style="font-family:var(--mono);font-size:.7rem;color:var(--text3);margin-top:6px">${item.pct}% du total</div>
    </div>`).join('');
  requestAnimationFrame(() => {
    document.querySelectorAll('.gauge-bar').forEach(b => { b.style.width = b.dataset.w; });
  });
}

function buildTopChurn(s) {
  const hasId = s.top_churn.length && 'CustomerID' in s.top_churn[0];
  if (hasId) document.getElementById('topChurnIdCol').style.display = '';
  const tbody = document.getElementById('topChurnTable');
  tbody.innerHTML = '';
  const actions = ['Offre de rétention', 'Contact prioritaire', 'Réduction personnalisée', 'Programme fidélité', 'Appel de suivi'];
  s.top_churn.forEach((row, i) => {
    const prob = row['Churn_Proba_%'];
    let pillClass = prob >= 70 ? 'pill-danger' : prob >= 30 ? 'pill-warn' : 'pill-safe';
    const action = actions[i % actions.length];
    tbody.innerHTML += `
      <tr>
        <td class="mono col-muted">${i+1}</td>
        ${hasId ? `<td class="mono col-text2">${row.CustomerID ?? '—'}</td>` : ''}
        <td><span class="pill ${pillClass}">${prob}%</span></td>
        <td style="color:${SEG_COLORS[row.Segment]||'#94a3b8'};font-weight:600">${row.Segment}</td>
        <td class="mono col-g1">${fmtDT(row.Depense_Prevue_DT)} DT</td>
        <td style="font-size:.8rem;color:var(--text3)">${action}</td>
      </tr>`;
  });
}

function buildSegPie(s) {
  const labels = Object.keys(s.segments);
  const data   = Object.values(s.segments);
  const colors = labels.map(l => SEG_COLORS[l] || '#94a3b8');
  charts.segPie = new Chart(document.getElementById('chartSegPie'), {
    type: 'pie',
    data: { labels, datasets: [{ data, backgroundColor: colors.map(c=>c+'cc'), borderColor: colors, borderWidth: 2 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { padding: 16, usePointStyle: true } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.label}: ${ctx.parsed.toLocaleString('fr-FR')} (${(ctx.parsed/s.total*100).toFixed(1)}%)` } }
      }
    }
  });
}

function buildSegStacked(s) {
  const labels = Object.keys(s.segments);
  const churnData  = labels.map(l => {
    const total = s.segments[l];
    return Math.round(total * (s.churn_by_segment[l]||0) / 100);
  });
  const stableData = labels.map((l,i) => s.segments[l] - churnData[i]);
  const colors     = labels.map(l => SEG_COLORS[l] || '#94a3b8');
  charts.segStacked = new Chart(document.getElementById('chartSegStacked'), {
    type: 'bar',
    data: {
      labels: labels.map(l => l.length>18?l.slice(0,18)+'…':l),
      datasets: [
        { label: 'Stable',  data: stableData, backgroundColor: 'rgba(46,213,115,.6)', borderColor: '#2ed573', borderWidth: 1, borderRadius: {topLeft:0,topRight:0,bottomLeft:6,bottomRight:6} },
        { label: 'Churn',   data: churnData,  backgroundColor: 'rgba(255,71,87,.7)',  borderColor: '#ff4757', borderWidth: 1, borderRadius: {topLeft:6,topRight:6,bottomLeft:0,bottomRight:0} },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { stacked: true, grid: { display: false } },
        y: { stacked: true, grid: { color: 'rgba(255,255,255,.04)' } }
      },
      plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, padding: 16 } } }
    }
  });
}

function buildTable(s) {
  const tbody = document.getElementById('previewTable');
  tbody.innerHTML = '';
  s.preview.forEach((row, i) => {
    const isChurn = row.Churn_Pred === 1;
    const prob    = row['Churn_Proba_%'];
    const pill    = isChurn ? '<span class="pill pill-danger">⚠ Churn</span>' : '<span class="pill pill-safe">✓ Stable</span>';
    let probClass = prob >= 70 ? 'col-danger' : prob >= 30 ? 'col-warn' : 'col-g1';
    tbody.innerHTML += `
      <tr>
        <td class="mono col-muted">${i+1}</td>
        <td>${pill}</td>
        <td class="mono ${probClass}">${prob}%</td>
        <td style="color:${SEG_COLORS[row.Segment]||'#94a3b8'};font-weight:600;font-size:.82rem">${row.Segment}</td>
        <td class="mono col-g1">${fmtDT(row.Depense_Prevue_DT)} DT</td>
      </tr>`;
  });
}

// ── TABS ───────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    const panels = ['overview','churn','segments','table'];
    t.classList.toggle('active', panels[i] === name);
  });
  document.querySelectorAll('.tab-panel').forEach(p => {
    p.classList.toggle('active', p.id === 'tab-' + name);
  });
}

// ── HELPERS ────────────────────────────────────────────────────
function fmt(v) {
  if(v >= 1000000) return (v/1000000).toFixed(1) + 'M';
  if(v >= 1000)    return (v/1000).toFixed(1) + 'K';
  return v.toLocaleString('fr-FR');
}
function fmtDT(v) {
  if(v >= 1000000) return (v/1000000).toFixed(1) + 'M';
  if(v >= 1000)    return (v/1000).toFixed(1) + 'K';
  return parseFloat(v).toLocaleString('fr-FR', {minimumFractionDigits:2,maximumFractionDigits:2});
}
function showError(msg) {
  errorBox.textContent = '⚠  ' + msg;
  errorBox.style.display = 'block';
}
function downloadCSV() {
  if (!lastBlob) return;
  const url = URL.createObjectURL(lastBlob);
  const a = document.createElement('a'); a.href = url;
  a.download = 'predictions_' + Date.now() + '.csv'; a.click();
  URL.revokeObjectURL(url);
}
function downloadJSON() {
  if (!lastStats) return;
  const blob = new Blob([JSON.stringify(lastStats, null, 2)], {type: 'application/json'});
  const url  = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url;
  a.download = 'analytics_' + Date.now() + '.json'; a.click();
  URL.revokeObjectURL(url);
}
function resetAll() {
  csvFile.value = '';
  document.getElementById('fileName').textContent = '';
  document.getElementById('fileTag').style.display = 'none';
  predictBtn.disabled = true;
  resetBtn.style.display = 'none';
  results.style.display = 'none';
  errorBox.style.display = 'none';
  lastBlob = null; lastStats = null;
}
</script>
</body>
</html>"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "Aucun fichier reçu."}), 400
        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Format invalide — fichier .csv requis."}), 400
        df_raw = pd.read_csv(file)
        if df_raw.empty:
            return jsonify({"error": "Le fichier CSV est vide."}), 400

        df_result = run_pipeline(df_raw)
        stats     = build_stats(df_result)

        output = io.BytesIO()
        df_result.to_csv(output, index=False)
        output.seek(0)

        response = make_response(send_file(
            output, mimetype="text/csv", as_attachment=True,
            download_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        ))
        response.headers["X-Stats"] = json.dumps(stats)
        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "ok",
        "models":    ["classifier", "regressor", "kmeans", "scaler", "pca"],
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/stats", methods=["GET"])
def stats_schema():
    return jsonify({
        "description": "POST /predict with a CSV file to get predictions.",
        "fields": ["Churn_Pred", "Churn_Proba_%", "Cluster_ID", "Segment", "Depense_Prevue_DT"],
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)