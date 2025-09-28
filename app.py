# index.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import sqlite3
from fpdf import FPDF
from spellchecker import SpellChecker
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pycountry
from rapidfuzz import fuzz
import plotly.express as px
import io
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="AI Smart Data Cleaner + Dashboard")

st.title("🧹 AI Smart Data Cleaning & Dashboard")
st.write("Smart cleaning, fuzzy deduplication, visual dashboard, multi-format IO (CSV/Excel/JSON/SQLite/PDF).")

# ---------------------------
# Helpers
# ---------------------------
def read_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    if name.endswith(".json"):
        return pd.read_json(uploaded_file)
    # fallback: try csv
    return pd.read_csv(uploaded_file)

def clean_generic_text(x):
    if pd.isna(x):
        return 'None'
    s = str(x).strip()
    s = re.sub(r'<[^>]+>', '', s)  # remove HTML
    s = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if s == '' or s.lower() == 'nan':
        return 'None'
    return s.title()

def standardize_phone(s):
    if pd.isna(s):
        return 'None'
    s2 = re.sub(r'[^\d]', '', str(s))
    if s2 == '' or s2.lower() == 'nan':
        return 'None'
    return s2

def standardize_email(s):
    if pd.isna(s):
        return 'None'
    s2 = str(s).strip().lower()
    if s2 == '' or s2.lower() == 'nan':
        return 'None'
    return s2

def is_valid_email_regex(s):
    if s == 'None' or pd.isna(s): 
        return False
    return bool(re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", str(s)))

def is_valid_phone_simple(s):
    if s == 'None' or pd.isna(s): return False
    s = str(s)
    return s.isdigit() and 8 <= len(s) <= 15

def build_spellchecker_with_custom(words):
    sp = SpellChecker()
    # add common domain words or dataset-specific tokens to avoid mis-corrections
    for w in words:
        try:
            if isinstance(w, str) and w.strip() and w.lower() != 'none':
                sp.word_frequency.add(w.lower())
        except Exception:
            pass
    return sp

# ---------------------------
# UI: file upload and initial load
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV / Excel / JSON (or drag & drop)", type=["csv","xlsx","xls","json"])
if not uploaded_file:
    st.info("Upload a data file to start (CSV/Excel/JSON).")
    st.stop()

df_original = read_file(uploaded_file).copy()
# keep original copy for before/after comparisons
if 'original_df' not in st.session_state:
    st.session_state['original_df'] = df_original.copy()
if 'cleaned_df' not in st.session_state:
    st.session_state['cleaned_df'] = df_original.copy()

st.subheader("Original Data (sample)")
st.dataframe(st.session_state['original_df'].head(10))

# ---------------------------
# Pre-clean (standardization)
# ---------------------------
st.subheader("Pre-clean & standardize")
if st.button("Run Pre-clean"):
    df = st.session_state['cleaned_df']
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # fill trivial NaNs
    df = df.replace({np.nan: None})
    # apply generic cleaning to object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(clean_generic_text)
    # phones, emails, names standardization
    if 'phone' in df.columns:
        df['phone'] = df['phone'].apply(standardize_phone)
    if 'email' in df.columns:
        df['email'] = df['email'].apply(standardize_email)
    if 'name' in df.columns:
        df['name'] = df['name'].apply(lambda x: str(x).strip().title() if x!='None' else 'None')
    # numeric coercion
    for col in ['age','id']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    st.session_state['cleaned_df'] = df
    st.success("Pre-clean finished.")

# preview
st.subheader("Data after Pre-clean (sample)")
st.dataframe(st.session_state['cleaned_df'].head(10))

# ---------------------------
# Dashboard: before/after missing values
# ---------------------------
st.subheader("Dashboard — Missing Values & Distributions")
orig = st.session_state['original_df'].replace({np.nan:'None'}).copy()
cleaned = st.session_state['cleaned_df'].replace({np.nan:'None'}).copy()

# compute missing counts
orig_missing = orig.isnull().sum().to_dict() if not orig.empty else {}
clean_missing = cleaned.replace('None', np.nan).isnull().sum().to_dict()

# show bar chart compare for top columns
cols_for_missing = list(set(list(orig.columns)[:6] + list(cleaned.columns)[:6]))
missing_df = pd.DataFrame({
    'column': cols_for_missing,
    'before': [orig.replace('None', np.nan)[c].isnull().sum() if c in orig.columns else 0 for c in cols_for_missing],
    'after': [cleaned.replace('None', np.nan)[c].isnull().sum() if c in cleaned.columns else 0 for c in cols_for_missing]
})
fig_missing = px.bar(missing_df.melt(id_vars='column', value_vars=['before','after'], var_name='stage', value_name='missing'), 
                     x='column', y='missing', color='stage', barmode='group', title='Missing values: before vs after (sample columns)')
st.plotly_chart(fig_missing, use_container_width=True)

# Age distribution before/after if age exists
if 'age' in orig.columns or 'age' in cleaned.columns:
    tmp_before = orig.copy()
    tmp_before['age'] = pd.to_numeric(tmp_before['age'], errors='coerce')
    tmp_before = tmp_before.dropna(subset=['age'])
    tmp_after = cleaned.copy()
    tmp_after['age'] = pd.to_numeric(tmp_after['age'], errors='coerce')
    tmp_after = tmp_after.dropna(subset=['age'])
    fig_age = px.histogram(pd.concat([
        tmp_before.assign(stage='before'),
        tmp_after.assign(stage='after')
    ]), x='age', color='stage', barmode='overlay', nbins=20, title='Age distribution before vs after')
    st.plotly_chart(fig_age, use_container_width=True)

# Pie charts for emails/phones validity
def validity_series(df, col, validator):
    if col not in df.columns:
        return None
    s = df[col].fillna('None').replace('None', np.nan)
    valid = s.apply(lambda x: validator(x)).sum()
    invalid = len(s) - valid
    return pd.DataFrame({'status':['valid','invalid'],'count':[valid, invalid]})

col1, col2 = st.columns(2)
with col1:
    email_pie_df = validity_series(st.session_state['cleaned_df'], 'email', is_valid_email_regex)
    if email_pie_df is not None:
        fig_e = px.pie(email_pie_df, names='status', values='count', title='Emails: valid vs invalid')
        st.plotly_chart(fig_e, use_container_width=True)
with col2:
    phone_pie_df = validity_series(st.session_state['cleaned_df'], 'phone', is_valid_phone_simple)
    if phone_pie_df is not None:
        fig_p = px.pie(phone_pie_df, names='status', values='count', title='Phones: valid vs invalid')
        st.plotly_chart(fig_p, use_container_width=True)

# ---------------------------
# Smart deduplication using blocking + rapidfuzz (improved)
# ---------------------------
st.subheader("Smart Deduplication (RapidFuzz-based)")

dedup_cols = []
if 'name' in st.session_state['cleaned_df'].columns: dedup_cols.append('name')
if 'email' in st.session_state['cleaned_df'].columns: dedup_cols.append('email')
if 'phone' in st.session_state['cleaned_df'].columns: dedup_cols.append('phone')
if 'age' in st.session_state['cleaned_df'].columns: dedup_cols.append('age')

st.write("Dedup columns detected:", dedup_cols)

name_threshold = st.slider("Name fuzzy threshold (0-100)", 80, 100, 90, 1)
if st.button("Run RapidFuzz Deduplication"):
    df = st.session_state['cleaned_df'].copy()
    n = len(df)
    to_drop = set()

    # Strong exact matches via email / phone
    if 'email' in df.columns:
        # standardize
        groups = df.groupby('email').groups
        for k, idxs in groups.items():
            if k!='None' and len(idxs)>1:
                # keep first, drop others
                to_drop.update(idxs[1:])
    if 'phone' in df.columns:
        groups = df.groupby('phone').groups
        for k, idxs in groups.items():
            if k!='None' and len(idxs)>1:
                to_drop.update(idxs[1:])

    # Fuzzy name matching within blocks: block by first 3 chars of name to reduce comparisons
    if 'name' in df.columns:
        df['block'] = df['name'].str[:3]
        for blk, blk_df in df.groupby('block'):
            idxs = blk_df.index.tolist()
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    if idxs[j] in to_drop:
                        continue
                    a = str(df.at[idxs[i], 'name'])
                    b = str(df.at[idxs[j], 'name'])
                    score = fuzz.token_sort_ratio(a, b)  # 0..100
                    # require age proximity if present
                    ages_close = True
                    if 'age' in df.columns:
                        ai = df.at[idxs[i], 'age']
                        aj = df.at[idxs[j], 'age']
                        if pd.notna(ai) and pd.notna(aj):
                            ages_close = abs(float(ai) - float(aj)) <= 3
                    if score >= name_threshold and ages_close:
                        to_drop.add(idxs[j])

    df_deduped = df.drop(index=list(to_drop)).reset_index(drop=True)
    # reindex id if exists
    if 'id' in df_deduped.columns:
        df_deduped['id'] = range(1, len(df_deduped)+1)
    st.session_state['cleaned_df'] = df_deduped.drop(columns=['block'], errors='ignore')
    st.success(f"Deduplication finished. Dropped {len(to_drop)} records.")
    st.dataframe(st.session_state['cleaned_df'].head(10))

# ---------------------------
# Spell correction enhancements (Cities + Notes)
# ---------------------------
st.subheader("Spell Correction (Cities & Notes)")

if st.button("Run Spell Correction (Cities & Notes)"):
    df = st.session_state['cleaned_df'].copy()
    # Build custom dictionary from unique city tokens and other tokens to avoid over-correction
    custom_words = set()
    if 'city' in df.columns:
        custom_words.update([str(x).lower() for x in df['city'].unique() if pd.notna(x)])
    if 'notes' in df.columns:
        # take most common tokens
        tokens = ' '.join(df['notes'].dropna().astype(str).tolist()).split()
        for t in tokens:
            if len(t) > 2:
                custom_words.add(t.lower())

    sp = build_spellchecker_with_custom(custom_words)

    # Try to enrich with country names to reduce false positives (pycountry)
    country_names = {c.name.lower() for c in pycountry.countries}
    for name in country_names:
        sp.word_frequency.add(name)

    if 'city' in df.columns:
        def fix_city(x):
            if x == 'None': return 'None'
            s = str(x)
            # only alphabetic words allowed
            if not re.match(r'^[A-Za-z\s\-]+$', s):
                return 'None'
            suggestion = sp.correction(s)
            if suggestion is None:
                return s.title()
            return suggestion.title()
        df['city'] = df['city'].apply(fix_city)

    # For notes, apply correction token-wise but conservatively (only correct tokens not in custom words)
    if 'notes' in df.columns:
        def fix_notes(s):
            if s == 'None': return 'None'
            words = str(s).split()
            corrected = []
            for w in words:
                if w.lower() in custom_words or w.lower() in country_names:
                    corrected.append(w)
                else:
                    corrected.append(sp.correction(w) or w)
            return ' '.join(corrected)
        df['notes'] = df['notes'].apply(fix_notes)

    st.session_state['cleaned_df'] = df
    st.success("Spell correction applied (conservative approach).")
    st.dataframe(st.session_state['cleaned_df'].head(10))

# ---------------------------
# Missing value imputation (scaled KNN) & simple fills
# ---------------------------
st.subheader("Imputation")
impute_col1, impute_col2 = st.columns(2)
with impute_col1:
    if st.button("Impute numeric missing with Scaled KNN (K=5)"):
        try:
            df = st.session_state['cleaned_df'].copy()
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            cols_to_impute = [c for c in numeric_cols if c != 'id']
            if len(cols_to_impute) == 0:
                st.info("No numeric columns to impute.")
            else:
                X = df[cols_to_impute].apply(lambda x: pd.to_numeric(x, errors='coerce'))
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X.fillna(0))  # fill zeros for scaling; KNN handles NaNs
                imputer = KNNImputer(n_neighbors=5, weights='distance')
                imputed = imputer.fit_transform(Xs)
                inv = scaler.inverse_transform(imputed)
                df[cols_to_impute] = inv
                st.session_state['cleaned_df'] = df
                st.success("Numeric columns imputed with Scaled KNN.")
        except Exception as e:
            st.error(f"Error: {e}")

with impute_col2:
    if st.button("Fill numeric missing with mean"):
        df = st.session_state['cleaned_df'].copy()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cols_to_fill = [c for c in numeric_cols if c != 'id']
        if len(cols_to_fill) == 0:
            st.info("No numeric columns to fill.")
        else:
            df[cols_to_fill] = df[cols_to_fill].fillna(df[cols_to_fill].mean())
            st.session_state['cleaned_df'] = df
            st.success("Filled numeric missing with mean.")

# ---------------------------
# Specific validators (email, phone)
# ---------------------------
st.subheader("Validators (Email / Phone)")

val_col1, val_col2 = st.columns(2)
with val_col1:
    if st.button("Validate Emails (mark invalid => 'None')"):
        df = st.session_state['cleaned_df'].copy()
        if 'email' in df.columns:
            df['email_valid'] = df['email'].apply(lambda x: is_valid_email_regex(x))
            # set invalid to 'None'
            df.loc[~df['email_valid'], 'email'] = 'None'
            df = df.drop(columns=['email_valid'])
            st.session_state['cleaned_df'] = df
            st.success("Emails validated and invalid set to 'None'.")
        else:
            st.info("No 'email' column found.")

with val_col2:
    if st.button("Validate Phones (mark invalid => 'None')"):
        df = st.session_state['cleaned_df'].copy()
        if 'phone' in df.columns:
            df['phone_valid'] = df['phone'].apply(lambda x: is_valid_phone_simple(x))
            df.loc[~df['phone_valid'], 'phone'] = 'None'
            df = df.drop(columns=['phone_valid'])
            st.session_state['cleaned_df'] = df
            st.success("Phones validated and invalid set to 'None'.")
        else:
            st.info("No 'phone' column found.")

# ---------------------------
# Export: CSV / JSON / Excel / SQLite / PDF report
# ---------------------------
st.subheader("Export / Save")

export_col1, export_col2, export_col3 = st.columns(3)
with export_col1:
    csv = st.session_state['cleaned_df'].to_csv(index=False).encode('utf-8')
    st.download_button("Download clean CSV", csv, "clean_data.csv", "text/csv")

with export_col2:
    excel_buf = io.BytesIO()
    st.session_state['cleaned_df'].to_excel(excel_buf, index=False, engine='openpyxl')
    st.download_button("Download clean Excel", excel_buf.getvalue(), "clean_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with export_col3:
    json_str = st.session_state['cleaned_df'].to_json(orient='records', force_ascii=False)
    st.download_button("Download clean JSON", json_str, "clean_data.json", "application/json")

# Save to SQLite
if st.button("Save cleaned data to SQLite (local file: cleaned_data.db)"):
    try:
        conn = sqlite3.connect("cleaned_data.db")
        st.session_state['cleaned_df'].to_sql("cleaned", conn, if_exists="replace", index=False)
        conn.close()
        st.success("Data saved to cleaned_data.db (table: cleaned).")
    except Exception as e:
        st.error(f"SQLite save failed: {e}")

# PDF Report generator (text summary + optional images if kaleido installed)
if st.button("Generate PDF Report (summary)"):
    try:
        # compute summary
        before = st.session_state['original_df']
        after = st.session_state['cleaned_df']
        total_before = len(before)
        total_after = len(after)
        duplicates_removed = total_before - total_after
        missing_before = before.replace('None', np.nan).isnull().sum().sum()
        missing_after = after.replace('None', np.nan).isnull().sum().sum()
        invalid_emails = (after['email'].apply(lambda x: not is_valid_email_regex(x)) & after['email'].notnull()).sum() if 'email' in after.columns else 0
        invalid_phones = (after['phone'].apply(lambda x: not is_valid_phone_simple(x)) & after['phone'].notnull()).sum() if 'phone' in after.columns else 0

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, "AI Smart Data Cleaning - Report", ln=True)
        pdf.ln(4)
        pdf.multi_cell(0, 6, f"Total records (before): {total_before}")
        pdf.multi_cell(0, 6, f"Total records (after): {total_after}")
        pdf.multi_cell(0, 6, f"Duplicates removed: {duplicates_removed}")
        pdf.multi_cell(0, 6, f"Missing values (before): {missing_before}")
        pdf.multi_cell(0, 6, f"Missing values (after): {missing_after}")
        pdf.multi_cell(0, 6, f"Invalid emails (after cleaning): {invalid_emails}")
        pdf.multi_cell(0, 6, f"Invalid phones (after cleaning): {invalid_phones}")
        pdf.ln(6)
        pdf.multi_cell(0, 6, "Notes:")
        pdf.multi_cell(0, 6, "- Deduplication used RapidFuzz name scoring + exact email/phone grouping.\n- Spell correction used local dictionary + pycountry enrichment.\n- Numeric imputation done with scaled KNN where requested.")
        # Try to attach images from plotly (requires kaleido)
        try:
            # save missing_df plot as image
            img_path = os.path.join(tempfile.gettempdir(), "missing_plot.png")
            fig_missing.write_image(img_path, scale=2)
            pdf.image(img_path, w=180)
        except Exception:
            # if cannot write image, skip silently
            pass

        out_path = "cleaning_report.pdf"
        pdf.output(out_path)
        with open(out_path, "rb") as f:
            btn = st.download_button("Download PDF Report", f.read(), file_name="cleaning_report.pdf", mime="application/pdf")
        st.success("PDF report generated.")
    except Exception as e:
        st.error(f"PDF report generation failed: {e}")

st.markdown("---")
st.info("Notes: For better PDF charts, install 'kaleido' (pip install kaleido). For global phone validation and E.164 formatting use the 'phonenumbers' library.")
