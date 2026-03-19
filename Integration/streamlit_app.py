import os, pickle, re, json
from datetime import datetime
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
W_P1, W_P2, W_P4, W_P5 = 0.40, 0.30, 0.20, 0.10
LINK_RANK_WEIGHTS = {1: 0.50, 2: 0.30, 3: 0.20}
DEFAULT_VISUAL_SCORE = 0.5
FINAL_THRESHOLDS = {'CRITICAL': 0.70, 'HIGH RISK': 0.50, 'REVIEW': 0.30, 'LOW RISK': 0.00}
P3_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are an AI analyst for the INTERPOL Risk Intelligence System (CS610 project).
You help compliance officers understand screening results and explore the fugitive database.

PIPELINE OVERVIEW (P3 is a biometric confirmation check):
- Pillar 3 (confirmation): Biometric prediction via ensemble voting (LR + RF + XGBoost). Compares client vs matched fugitive on 7 features: name_similarity, age_difference, same_gender, height_difference, weight_difference, same_hair_colour, same_eye_colour. Outputs confidence (0-1). If >= 0.5 (MATCH), the identity is biometrically confirmed → immediate CRITICAL escalation. If < 0.5 (MISMATCH), biometrics are inconclusive → proceed with full multi-pillar scoring.
- Pillar 1 (40%): TF-IDF char n-gram cosine similarity for identity resolution
- Pillar 2 (30%): Crime severity from categorisation export (terrorism=1.0, homicide=0.9, sexual crime=0.8, armed formation/assault/narcotics=0.7, financial crime=0.5)
- Pillar 4 (20%): Hidden linkage from GraphSAGE link predictions (top-3 linked fugitives, rank weights normalised to sum to 1.0)
- Pillar 5 (10%): Visual similarity via CLIP (placeholder 0.5 until delivered)
- If P3 MATCH: Final Risk = 1.0 (CRITICAL). If P3 mismatch/N/A: Final Risk = P1×0.40 + P2×0.30 + P4×0.20 + P5×0.10

RISK TIERS:
- CRITICAL (≥0.70): Freeze + escalate to compliance officer
- HIGH RISK (≥0.50): Senior analyst review within 24hr
- REVIEW (≥0.30): Junior analyst flag, monitor 30 days
- LOW RISK (<0.30): Pass, log to audit trail

BIOMETRIC DATA COLLECTION:
When a user requests to screen a client but only provides a name, you MUST ask for the following biometric details before screening can proceed:
1. Gender (M/F)
2. Date of birth (or approximate age)
3. Height (in metres, e.g. 1.75)
4. Weight (in kg, e.g. 70)
5. Hair colour (e.g. Black, Brown, Blond, Red, Grey, White)
6. Eye colour (e.g. Brown, Blue, Green, Hazel, Grey)

Present this as a friendly form-like request. Explain that biometric data enables the P3 biometric confirmation check — if biometrics match, the case is immediately escalated as CRITICAL; if they don't match, the system proceeds with full multi-pillar evaluation.

If the user provides ALL biometric fields along with the name in one message, proceed directly with screening.

When you receive the biometric data, respond with EXACTLY this format on its own line so the system can parse it:
BIOMETRICS_COLLECTED: gender=<M or F>, dob=<YYYY-MM-DD or age_XX>, height=<metres>, weight=<kg>, hair=<colour>, eye=<colour>

RESPONSE FORMAT RULES — you MUST follow these:
- Use markdown formatting: **bold** for emphasis, headers (##, ###) for sections, bullet points for lists.
- For screening results, structure your response with these sections:
  ### Match Summary
  ### Pillar Breakdown
  ### Risk Assessment
  ### Recommended Action
- Use tables when comparing data (e.g. pillar scores, fugitive comparisons).
- Keep responses concise and professional — 150-300 words for screening, shorter for DB queries.
- Use the data provided — do not fabricate scores or fugitive names."""

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    # ── Directory config (mirrors notebook Cell 4) ────────────────────────────
    repo_root  = os.path.abspath(os.path.join(os.getcwd(), '..'))

    def _res(*candidates):
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[-1]

    p1_base    = os.path.join(repo_root, 'Pillar 1 (Identity Resolution)')
    p2_base    = os.path.join(repo_root, 'Pillar 2 (Crime Severity)')
    p3_base    = os.path.join(repo_root, 'Pillar 3 (Biometric Prediction)')
    p4_base    = os.path.join(repo_root, 'Pillar 4 (GCN and Graphsage)')
    p4_out_dir = os.path.join(p4_base, 'outputs')

    fugitive_csv = os.path.join(p4_base, 'crime_analysis_results_aft_transformer_ner.csv')

    p1_vectorizer_pkl = _res(
        os.path.join(p1_base, 'outputs', 'pillar1_name_vectorizer.pkl'),
        os.path.join(p1_base, 'output',  'pillar1_name_vectorizer.pkl'),
        os.path.join(repo_root, 'outputs', 'pillar1_name_vectorizer.pkl'),
    )
    p1_embeddings_pkl = _res(
        os.path.join(p1_base, 'outputs', 'pillar1_name_embeddings.pkl'),
        os.path.join(p1_base, 'output',  'pillar1_name_embeddings.pkl'),
        os.path.join(repo_root, 'outputs', 'pillar1_name_embeddings.pkl'),
    )
    p2_crime_csv = _res(
        os.path.join(p2_base, 'crime_categorisation_export.csv'),
        os.path.join(p2_base, 'outputs', 'crime_categorisation_export.csv'),
    )
    p4_links_csv = os.path.join(p4_out_dir, 'top_new_links_all.csv')

    # P2: Crime categorisation
    df_crime = pd.read_csv(p2_crime_csv)
    severity_map = df_crime.set_index('id')[['final_crime_label', 'severity_score', 'risk_tier']].to_dict('index')
    crime_severity = df_crime.groupby('final_crime_label')['severity_score'].first().to_dict()
    crime_severity['Unknown'] = 0.1

    # Main fugitive CSV
    df_fug = pd.read_csv(fugitive_csv)
    df_unique = df_fug.drop_duplicates(subset='id', keep='first').reset_index(drop=True)
    df_unique['final_crime_label'] = df_unique['id'].map(
        lambda x: severity_map.get(x, {}).get('final_crime_label', 'Unknown'))
    df_unique['severity_score'] = df_unique['id'].map(
        lambda x: severity_map.get(x, {}).get('severity_score', 0.1))

    # P1: TF-IDF from pickle or rebuild
    if os.path.exists(p1_vectorizer_pkl) and os.path.exists(p1_embeddings_pkl):
        with open(p1_vectorizer_pkl, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(p1_embeddings_pkl, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                                     max_features=5000, sublinear_tf=True)
        embeddings = normalize(vectorizer.fit_transform(df_unique['name'].str.upper().astype(str)))

    # P4: Links CSV + hidden linkage scores
    df_links = pd.read_csv(p4_links_csv) if os.path.exists(p4_links_csv) else pd.DataFrame()
    p4_scores_csv = os.path.join(p4_out_dir, 'hidden_linkage_scores.csv')
    if os.path.exists(p4_scores_csv):
        linkage_score_map = pd.read_csv(p4_scores_csv).set_index('id')['linkage_score'].to_dict()
    else:
        linkage_score_map = {}

    # P3: Biometric model + client database
    p3_model_pkl = _res(
        os.path.join(repo_root, 'outputs', 'pillar3_ensemble_voting.pkl'),
        os.path.join(p3_base, 'output', 'ensemble_voting.pkl'),
        os.path.join(p3_base, 'outputs', 'ensemble_voting.pkl'),
    )
    p3_model = None
    if os.path.exists(p3_model_pkl):
        with open(p3_model_pkl, 'rb') as f:
            p3_model = pickle.load(f)

    client_csv = os.path.join(repo_root, 'synthetic_data_generation', 'synthetic_client_data.csv')
    client_bio_map = {}
    if os.path.exists(client_csv):
        df_cl = pd.read_csv(client_csv)
        df_cl['full_name_upper'] = df_cl['full_name'].str.upper().str.strip()
        df_cl = df_cl.drop_duplicates(subset='full_name_upper', keep='first')
        client_bio_map = df_cl.set_index('full_name_upper').to_dict('index')

    return {
        'df_unique': df_unique, 'vectorizer': vectorizer, 'embeddings': embeddings,
        'df_links': df_links, 'linkage_score_map': linkage_score_map,
        'severity_map': severity_map, 'crime_severity': crime_severity,
        'df_crime': df_crime,
        'p3_model': p3_model, 'client_bio_map': client_bio_map,
    }

# ─────────────────────────────────────────────────────────────────────────────
# PILLAR 3 HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _parse_dob_st(val):
    if pd.isna(val) or str(val).strip() in ('', 'nan', 'NaN'):
        return None
    try:
        return pd.to_datetime(str(val).strip())
    except Exception:
        return None

def compute_p3_features_st(client, fugitive):
    c_name = str(client.get('full_name', ''))
    f_name = str(fugitive.get('name', ''))
    name_sim = SequenceMatcher(None, c_name.upper(), f_name.upper()).ratio()
    today = datetime.today()
    c_dob, f_dob = _parse_dob_st(client.get('date_of_birth')), _parse_dob_st(fugitive.get('birth_date'))
    age_diff = abs(relativedelta(today, c_dob).years - relativedelta(today, f_dob).years) if c_dob and f_dob else 0
    c_g = str(client.get('gender', '')).strip().upper()
    f_g = str(fugitive.get('GENDER', '')).strip().upper()
    same_gender = 1 if (c_g and f_g and c_g == f_g) else 0
    try: c_h = float(client.get('height_m', 0) or 0)
    except: c_h = 0.0
    try: f_h = float(fugitive.get('height', 0) or 0)
    except: f_h = 0.0
    height_diff = abs(c_h - f_h)
    try: c_w = float(client.get('weight_kg', 0) or 0)
    except: c_w = 0.0
    try: f_w = float(fugitive.get('weight', 0) or 0)
    except: f_w = 0.0
    weight_diff = abs(c_w - f_w)
    c_hair = str(client.get('hair_color', '')).strip().upper()
    f_hair = str(fugitive.get('hairColor', '')).strip().upper()
    same_hair = 1 if (c_hair and f_hair and c_hair not in ('', 'NAN') and f_hair not in ('', 'NAN', 'OTHD') and c_hair == f_hair) else 0
    c_eye = str(client.get('eye_color', '')).strip().upper()
    f_eye = str(fugitive.get('eyeColor', '')).strip().upper()
    same_eye = 1 if (c_eye and f_eye and c_eye not in ('', 'NAN') and f_eye not in ('', 'NAN', 'OTHD') and c_eye == f_eye) else 0
    return [name_sim, age_diff, same_gender, height_diff, weight_diff, same_hair, same_eye]

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def screen(client_name, data, client_bio=None):
    df = data['df_unique']
    # Candidate retrieval via TF-IDF
    q = normalize(data['vectorizer'].transform([client_name.upper()]))
    sims = cosine_similarity(q, data['embeddings'])[0]
    top_idx = int(np.argmax(sims))
    p1_score = float(sims[top_idx])
    fug = df.iloc[top_idx]
    fid = fug['id']

    top3_idx = np.argsort(sims)[::-1][:3]
    top3 = [(df.iloc[i]['name'], round(float(sims[i]), 4)) for i in top3_idx]

    # P3: Biometric validation (runs FIRST)
    p3_conf, p3_match_flag, p3_note = None, None, ''
    p3_model = data.get('p3_model')
    client_bio_map = data.get('client_bio_map', {})
    client_key = client_name.upper().strip()
    bio = client_bio if client_bio else client_bio_map.get(client_key)
    if p3_model and bio:
        if 'full_name' not in bio:
            bio['full_name'] = client_name
        feats = compute_p3_features_st(bio, fug)
        feats = [0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in feats]
        p3_proba = p3_model.predict_proba([feats])[0]
        p3_conf = float(p3_proba[1])
        p3_match_flag = int(p3_conf >= P3_THRESHOLD)
    else:
        p3_note = 'No biometric data provided' if p3_model else 'P3 model not loaded'

    p3_str = f'{p3_conf:.4f}' if p3_conf is not None else f'N/A ({p3_note})'
    p3_verdict = 'MATCH' if p3_match_flag == 1 else ('NO MATCH' if p3_match_flag == 0 else 'N/A')
    p3_confirmed = (p3_match_flag == 1) if p3_match_flag is not None else False

    if p3_confirmed:
        # Biometrics confirm identity → strongest evidence → immediate CRITICAL
        if fid in data['severity_map']:
            e = data['severity_map'][fid]
            p2_label = e['final_crime_label']
        else:
            p2_label = fug.get('final_crime_label', 'Unknown')
        ctx = f"""SCREENING RESULT FOR: "{client_name}"
Matched Fugitive: {fug['name']} [{fid}]
Crime Type: {p2_label}

P3 BIOMETRIC CONFIRMATION:
  Confidence: {p3_str}  ({p3_verdict})
  Biometric match confirmed — this is the strongest evidence.

FINAL RISK SCORE: 1.0000
RISK TIER: CRITICAL
ACTION: Biometric match confirmed — freeze + escalate immediately

TOP-3 IDENTITY MATCHES:
""" + "\\n".join(f"  #{i+1} {n}: {s:.4f}" for i, (n, s) in enumerate(top3))
        return ctx, 'CRITICAL', 1.0

    # P3 mismatch or N/A → need all pillars to evaluate
    if fid in data['severity_map']:
        e = data['severity_map'][fid]
        p2_label, p2_score = e['final_crime_label'], e['severity_score']
    else:
        ct = fug.get('final_crime_label', 'Unknown')
        p2_label = ct
        p2_score = data['crime_severity'].get(ct, 0.1)

    links = data['df_links']
    link_rows = links[links['anchor_id'] == fid].sort_values('rank') if not links.empty else pd.DataFrame()
    avail_ranks = [int(r['rank']) for _, r in link_rows.iterrows()]
    wsum = sum(LINK_RANK_WEIGHTS.get(r, 0.0) for r in avail_ranks)
    p4_total, p4_detail = 0.0, []
    for _, row in link_rows.iterrows():
        rank = int(row['rank'])
        ls = float(row['predicted_link_score'])
        cid = row['candidate_id']
        if cid in data['severity_map']:
            cl, cs = data['severity_map'][cid]['final_crime_label'], data['severity_map'][cid]['severity_score']
        else:
            cl, cs = 'Unknown', 0.1
        rw = LINK_RANK_WEIGHTS.get(rank, 0.0)
        nw = rw / wsum if wsum > 0 else 0.0
        contrib = nw * ls * cs
        p4_total += contrib
        p4_detail.append(f"  Rank {rank} ({nw:.0%}): {row['candidate_name']} — link={ls:.4f}, crime={cl} (sev={cs}), contrib={contrib:.4f}")
    if link_rows.empty and fid in data.get('linkage_score_map', {}):
        p4_total = data['linkage_score_map'][fid]
        p4_detail.append(f"  (score from hidden_linkage_scores.csv — no per-link detail)")

    p5_score = DEFAULT_VISUAL_SCORE
    final = p1_score * W_P1 + p2_score * W_P2 + p4_total * W_P4 + p5_score * W_P5

    if final >= 0.70: tier, action = 'CRITICAL', 'Freeze + escalate to compliance officer'
    elif final >= 0.50: tier, action = 'HIGH RISK', 'Senior analyst review within 24hr'
    elif final >= 0.30: tier, action = 'REVIEW', 'Junior analyst flag, monitor 30 days'
    else: tier, action = 'LOW RISK', 'Pass, log to audit trail'

    ctx = f"""SCREENING RESULT FOR: "{client_name}"
Matched Fugitive: {fug['name']} [{fid}]
Crime Type: {p2_label}

PILLAR SCORES:
  P3 Biometric Conf.:    {p3_str}  ({p3_verdict})  <- full evaluation needed
  P1 Identity (TF-IDF): {p1_score:.4f} × {W_P1} = {p1_score*W_P1:.4f}
  P2 Crime Severity:     {p2_score:.4f} × {W_P2} = {p2_score*W_P2:.4f}
  P4 Hidden Linkage:     {p4_total:.4f} × {W_P4} = {p4_total*W_P4:.4f}
  P5 Visual Similarity:  {p5_score:.4f} × {W_P5} = {p5_score*W_P5:.4f}

FINAL RISK SCORE: {final:.4f}
RISK TIER: {tier}
ACTION: {action}

TOP-3 IDENTITY MATCHES:
""" + "\n".join(f"  #{i+1} {n}: {s:.4f}" for i, (n, s) in enumerate(top3))

    if p4_detail:
        ctx += "\n\nP4 LINKAGE BREAKDOWN:\n" + "\n".join(p4_detail)
    else:
        ctx += "\n\nP4 LINKAGE: No pre-computed links for this fugitive"

    return ctx, tier, final


def db_context(query, data):
    df = data['df_unique']
    parts = []

    parts.append(f"DATABASE SUMMARY: {len(df):,} unique fugitives")
    crime_dist = df['final_crime_label'].value_counts()
    parts.append("CRIME DISTRIBUTION:\n" + "\n".join(f"  {k}: {v:,}" for k, v in crime_dist.items()))

    if 'GENDER' in df.columns:
        gender_dist = df['GENDER'].value_counts()
        parts.append("GENDER: " + ", ".join(f"{k}={v:,}" for k, v in gender_dist.items()))

    if 'age_today' in df.columns:
        ages = df['age_today'].dropna()
        parts.append(f"AGE: mean={ages.mean():.1f}, min={ages.min():.0f}, max={ages.max():.0f}")

    if 'label' in df.columns:
        top_countries = df['label'].value_counts().head(10)
        parts.append("TOP 10 COUNTRIES:\n" + "\n".join(f"  {k}: {v:,}" for k, v in top_countries.items()))

    name_matches = df[df['name'].str.contains(query.upper(), na=False)]
    if not name_matches.empty and len(name_matches) <= 20:
        rows = []
        for _, r in name_matches.head(10).iterrows():
            rows.append(f"  {r['name']} [{r['id']}] — {r.get('final_crime_label', '?')}, "
                        f"age={r.get('age_today', '?')}, country={r.get('label', '?')}")
        parts.append(f"NAME MATCHES FOR '{query}':\n" + "\n".join(rows))

    severity_dist = df.groupby('final_crime_label')['severity_score'].first()
    parts.append("SEVERITY SCORES:\n" + "\n".join(f"  {k}: {v}" for k, v in severity_dist.items()))

    links = data['df_links']
    if not links.empty:
        parts.append(f"P4 LINKAGE DATA: {len(links):,} link predictions across {links['anchor_id'].nunique():,} anchors")
    lsm = data.get('linkage_score_map', {})
    if lsm:
        parts.append(f"P4 HIDDEN LINKAGE SCORES: {len(lsm):,} fugitives with pre-computed mean scores")

    return "\n\n".join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
TIER_STYLES = {
    'CRITICAL':  {'bg': '#fef2f2', 'border': '#fca5a5', 'color': '#991b1b', 'icon': '\U0001f534', 'bar': '#dc2626'},
    'HIGH RISK': {'bg': '#fff7ed', 'border': '#fdba74', 'color': '#9a3412', 'icon': '\U0001f7e0', 'bar': '#ea580c'},
    'REVIEW':    {'bg': '#fefce8', 'border': '#fde047', 'color': '#854d0e', 'icon': '\U0001f7e1', 'bar': '#ca8a04'},
    'LOW RISK':  {'bg': '#f0fdf4', 'border': '#86efac', 'color': '#166534', 'icon': '\u2705', 'bar': '#16a34a'},
}

def parse_biometrics(text):
    """Extract BIOMETRICS_COLLECTED line from LLM response and build a client_bio dict."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('BIOMETRICS_COLLECTED:'):
            parts = line.split(':', 1)[1].strip()
            bio = {}
            for pair in parts.split(','):
                pair = pair.strip()
                if '=' not in pair:
                    continue
                k, v = pair.split('=', 1)
                k, v = k.strip().lower(), v.strip()
                if k == 'gender':
                    bio['gender'] = v.upper()
                elif k == 'dob':
                    if v.startswith('age_'):
                        try:
                            age = int(v.replace('age_', ''))
                            from datetime import datetime, timedelta
                            bio['date_of_birth'] = (datetime.today() - timedelta(days=age*365)).strftime('%Y-%m-%d')
                        except: pass
                    else:
                        bio['date_of_birth'] = v
                elif k == 'height':
                    try: bio['height_m'] = float(v)
                    except: pass
                elif k == 'weight':
                    try: bio['weight_kg'] = float(v)
                    except: pass
                elif k == 'hair':
                    bio['hair_color'] = v
                elif k == 'eye':
                    bio['eye_color'] = v
            return bio if bio else None
    return None


def render_score_card(client_name, tier, score, rag_context):
    s = TIER_STYLES.get(tier, TIER_STYLES['REVIEW'])
    pct = min(score * 100, 100)
    lines = rag_context.split('\n')
    matched = next((l for l in lines if l.startswith('Matched Fugitive:')), '')
    crime = next((l for l in lines if l.startswith('Crime Type:')), '')
    p1_line = next((l for l in lines if 'P1 Identity' in l), '')
    p2_line = next((l for l in lines if 'P2 Crime' in l), '')
    p4_line = next((l for l in lines if 'P4 Hidden' in l), '')
    p5_line = next((l for l in lines if 'P5 Visual' in l), '')
    def extract_scores(line):
        parts = line.strip().split()
        raw = parts[-3] if len(parts) >= 3 else '\u2014'
        weighted = parts[-1] if len(parts) >= 1 else '\u2014'
        return raw, weighted
    p1r, p1w = extract_scores(p1_line)
    p2r, p2w = extract_scores(p2_line)
    p4r, p4w = extract_scores(p4_line)
    p5r, p5w = extract_scores(p5_line)
    return f"""<div style="background:{s['bg']}; border:1px solid {s['border']}; border-radius:12px;
                padding:1.2rem 1.5rem; margin-bottom:1rem; font-family:'Inter',sans-serif;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem;">
    <div>
      <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em; color:{s['color']}; font-weight:700;">{s['icon']} {tier}</div>
      <div style="font-size:1.3rem; font-weight:700; color:#1a1a2e;">Screening: {client_name}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:2rem; font-weight:800; color:{s['color']}; line-height:1;">{score:.2f}</div>
      <div style="font-size:0.65rem; color:#64748b; text-transform:uppercase;">Risk Score</div>
    </div>
  </div>
  <div style="background:#e2e8f0; border-radius:6px; height:8px; margin-bottom:1rem; overflow:hidden;">
    <div style="background:{s['bar']}; height:100%; width:{pct}%; border-radius:6px;"></div>
  </div>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.4rem 1.5rem; font-size:0.82rem; color:#334155;">
    <div><span style="color:#64748b;">Matched:</span> <b>{matched.replace('Matched Fugitive: ','')}</b></div>
    <div><span style="color:#64748b;">Crime:</span> <b>{crime.replace('Crime Type: ','')}</b></div>
    <div><span style="color:#64748b;">P1 Identity:</span> <code>{p1r}</code> &rarr; <b>{p1w}</b></div>
    <div><span style="color:#64748b;">P2 Severity:</span> <code>{p2r}</code> &rarr; <b>{p2w}</b></div>
    <div><span style="color:#64748b;">P4 Linkage:</span> <code>{p4r}</code> &rarr; <b>{p4w}</b></div>
    <div><span style="color:#64748b;">P5 Visual:</span> <code>{p5r}</code> &rarr; <b>{p5w}</b></div>
  </div>
</div>"""

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="INTERPOL Risk Intelligence", page_icon="\U0001f534", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #f8f9fc 0%, #eef1f8 100%);
    }
    .block-container { max-width: 860px; padding-top: 2rem; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1629 0%, #1a2342 100%);
    }
    [data-testid="stSidebar"] * {
        color: #c8cfe0 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stDivider {
        border-color: rgba(255,255,255,0.1);
    }

    .header-bar {
        background: linear-gradient(135deg, #0f1629 0%, #1e2a4a 100%);
        border-radius: 12px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .header-bar h1 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.7rem;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.02em;
    }
    .header-bar p {
        color: #8b95b0;
        font-size: 0.85rem;
        margin: 0;
    }
    .header-bar .accent {
        color: #e74c3c;
        font-weight: 600;
    }

    .pillar-row {
        display: flex; gap: 0.6rem; margin-top: 0.8rem; flex-wrap: wrap;
    }
    .pillar-tag {
        font-size: 0.7rem;
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.02em;
    }
    .pillar-tag.p1 { background: #1e3a5f; color: #7eb8f7; }
    .pillar-tag.p2 { background: #3a1f1f; color: #f7a07e; }
    .pillar-tag.p4 { background: #1f3a2a; color: #7ef7a0; }
    .pillar-tag.p5 { background: #3a2a1f; color: #f7d77e; }

    .sidebar-example {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin: 0.3rem 0;
        font-family: 'Inter', monospace;
        font-size: 0.8rem;
        color: #a0b0cc !important;
        cursor: default;
        transition: background 0.2s;
    }
    .sidebar-example:hover { background: rgba(255,255,255,0.1); }
    .sidebar-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7a96 !important;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
    }

    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 2px solid #d0d7e3 !important;
        background: #ffffff !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #4a6cf7 !important;
        box-shadow: 0 0 0 3px rgba(74,108,247,0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 0.2rem 0;">
        <div style="font-size:2rem;">🔴</div>
        <div style="font-size:1.2rem; font-weight:700; color:#fff !important; letter-spacing:0.04em;">
            INTERPOL IRIS
        </div>
        <div style="font-size:0.75rem; color:#6b7a96 !important; margin-top:0.15rem;">
            Integrated Risk Intelligence System
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model = "gpt-5-mini"
    st.markdown(f'<div style="font-size:0.8rem; color:#8b95b0 !important; margin-top:0.3rem;">Model: <b style="color:#fff !important;">{model}</b></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div class="sidebar-label">Example queries</div>', unsafe_allow_html=True)
    for ex in [
        "Screen S. Nikitenko",
        "Screen Abdul Sambolotov",
        "How many terrorism fugitives?",
        "Show me fugitives from Russia",
        "What crime types are in the database?",
    ]:
        st.markdown(f'<div class="sidebar-example">{ex}</div>', unsafe_allow_html=True)

st.markdown("""
<div class="header-bar">
    <h1>INTERPOL Risk Intelligence <span class="accent">Chatbot</span></h1>
    <p>CS610 Integrated Screening Pipeline &mdash; ask a question or screen a client name</p>
    <div class="pillar-row">
        <span class="pillar-tag p1">P1 &middot; TF-IDF &middot; 40%</span>
        <span class="pillar-tag p2">P2 &middot; Crime &middot; 30%</span>
        <span class="pillar-tag p4">P4 &middot; GraphSAGE &middot; 20%</span>
        <span class="pillar-tag p5">P5 &middot; CLIP &middot; 10%</span>
    </div>
</div>
""", unsafe_allow_html=True)

data = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_screen" not in st.session_state:
    st.session_state.pending_screen = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Screen a client or ask about the fugitive database..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    rag_context = None
    user_msg = None
    run_screening = False
    client_bio = None

    if st.session_state.pending_screen:
        pending_name = st.session_state.pending_screen
        if api_key:
            oai = OpenAI(api_key=api_key)
            parse_msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"The user previously asked to screen \"{pending_name}\" and was asked for biometric data. They replied with:\\n\\n\"{prompt}\"\\n\\nExtract the biometric fields and respond with the BIOMETRICS_COLLECTED line. If they did not provide enough info, ask again politely."},
            ]
            parse_resp = oai.chat.completions.create(model=model, messages=parse_msgs)
            llm_text = parse_resp.choices[0].message.content
            client_bio = parse_biometrics(llm_text)

        if client_bio:
            client_bio['full_name'] = pending_name
            rag_context, tier, score = screen(pending_name, data, client_bio=client_bio)
            bio_summary = ", ".join(f"{k}={v}" for k, v in client_bio.items() if k != 'full_name')
            user_msg = f"I screened client \"{pending_name}\" with biometrics: {bio_summary}.\\n\\nPipeline results:\\n\\n{rag_context}\\n\\nExplain this screening result clearly. What does each pillar score mean? Is this person a risk?"
            run_screening = True
            st.session_state.pending_screen = None
        else:
            user_msg = f"The user tried to provide biometric data for screening \"{pending_name}\" but the data was incomplete or unclear. Their message: \"{prompt}\". Ask them again for the missing fields: gender (M/F), date of birth or age, height (m), weight (kg), hair colour, eye colour."

    else:
        screen_match = re.match(r'(?i)screen\s+(.+)', prompt.strip())
        if screen_match:
            client_name = screen_match.group(1).strip().strip('"\'')
            st.session_state.pending_screen = client_name
            user_msg = f"The user wants to screen a client named \"{client_name}\" but has only provided the name. Ask them for the biometric data needed for P3 validation: gender, date of birth (or age), height (m), weight (kg), hair colour, and eye colour. Explain briefly why biometric data is needed (to avoid discrimination based solely on name matching and to power the P3 biometric gate)."
        else:
            rag_context = db_context(prompt, data)
            user_msg = f"The user asked: \"{prompt}\"\\n\\nHere is the current database context:\\n\\n{rag_context}\\n\\nAnswer their question using the data above."

    if not api_key:
        with st.chat_message("assistant"):
            st.warning("Enter your OpenAI API key in the sidebar for AI-powered explanations.")
            if rag_context:
                st.code(rag_context, language=None)
            fallback = f"**Raw pipeline output** (enter API key for AI explanation):\\n```\\n{(rag_context or 'Please provide biometric data.')[:2000]}\\n```"
            st.session_state.messages.append({"role": "assistant", "content": fallback})
    else:
        oai_client = OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        with st.chat_message("assistant"):
            stream = oai_client.chat.completions.create(model=model, messages=messages, stream=True)
            response = st.write_stream(stream)

        if run_screening and rag_context:
            render_score_card(st.session_state.pending_screen or client_name if 'client_name' in dir() else 'Client', tier, score, rag_context)

        bio_from_response = parse_biometrics(response)
        if bio_from_response and st.session_state.pending_screen:
            pending_name = st.session_state.pending_screen
            bio_from_response['full_name'] = pending_name
            rag_context, tier, score = screen(pending_name, data, client_bio=bio_from_response)
            bio_summary = ", ".join(f"{k}={v}" for k, v in bio_from_response.items() if k != 'full_name')
            follow_msg = f"I screened client \"{pending_name}\" with biometrics: {bio_summary}.\\n\\nPipeline results:\\n\\n{rag_context}\\n\\nExplain this screening result clearly."
            st.session_state.pending_screen = None
            msgs2 = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": follow_msg}]
            with st.chat_message("assistant"):
                stream2 = oai_client.chat.completions.create(model=model, messages=msgs2, stream=True)
                response2 = st.write_stream(stream2)
                render_score_card(pending_name, tier, score, rag_context)
            st.session_state.messages.append({"role": "assistant", "content": response2})

        st.session_state.messages.append({"role": "assistant", "content": response})
