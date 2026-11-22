"""
Bank Term-Deposit Prediction - Streamlit app
--------------------------------------------
Predicts whether a customer will subscribe to a term deposit ("deposit" = yes/no).

Files expected in the same folder as this app.py:
    model.pkl   - the model saved by train.py (a Pipeline: Prep -> Model that accepts RAW columns)
    bank.csv    - the original dataset (dropdown values + dataset overview)
    mlflow.db   - (optional) MLflow tracking DB, only used by the "Model info" tab

The app reads which input columns the model needs from the model itself, so it adapts
automatically if you retrain with different columns (e.g. DROP_COLUMNS in train.py).
An older model.pkl that expects 48 pre-processed features is still supported (legacy mode).

Run locally:   streamlit run app.py
"""

import sqlite3
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "bank.csv"
MLFLOW_DB = BASE_DIR / "mlflow.db"

FEATURE_COLS = [
    "age", "job", "marital", "education", "default", "balance", "housing", "loan",
    "contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome",
]
EDU_OPTIONS = ["primary", "secondary", "tertiary", "unknown"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
# poutcome is not asked in the UI. If the model still needs it, this value is sent.
DEFAULT_POUTCOME = "unknown"

# Readable names for the MLflow comparison table
MODEL_ALIASES = {"KNeighbors": "KNN", "SVC": "SVM"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return cloudpickle.load(f)


@st.cache_resource(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def model_label(model) -> str:
    name = type(final_estimator(model)).__name__.replace("Classifier", "")
    return MODEL_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Legacy support: old model.pkl = MinMaxScaler -> estimator on 48 pre-processed features.
# (Only used when the model does not know its input column names.)
#   0-39 one-hot | 40 education ordinal | 41 age, 42 day (Standard) |
#   43 balance, 44 duration, 46 campaign (Robust) | 45 (pdays+1)/3 | 47 previous
# ---------------------------------------------------------------------------
LEGACY_CAT_COLS = ["job", "marital", "default", "housing", "loan", "contact", "month", "poutcome"]


@st.cache_resource(show_spinner="Preparing legacy preprocessing...")
def build_legacy_preprocessor():
    df = load_data()
    return {
        "ohe": OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(df[LEGACY_CAT_COLS]),
        "std": StandardScaler().fit(df[["age", "day"]]),
        "rob": RobustScaler().fit(df[["balance", "duration", "campaign"]]),
    }


def legacy_preprocess(raw: pd.DataFrame) -> np.ndarray:
    p = build_legacy_preprocessor()
    X_cat = p["ohe"].transform(raw[LEGACY_CAT_COLS])
    edu = raw["education"].map({k: i for i, k in enumerate(EDU_OPTIONS)}).fillna(3).to_numpy()[:, None]
    age_day = p["std"].transform(raw[["age", "day"]])
    bdc = p["rob"].transform(raw[["balance", "duration", "campaign"]])
    pdays = ((raw["pdays"].clip(lower=-1) + 1) / 3).to_numpy()[:, None]
    previous = raw["previous"].to_numpy(dtype=float)[:, None]
    return np.hstack([X_cat, edu, age_day[:, [0]], age_day[:, [1]], bdc[:, [0]], bdc[:, [1]],
                      pdays, bdc[:, [2]], previous])


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def get_input_cols(model):
    """Raw columns the model needs (None for a legacy model)."""
    cols = getattr(model, "feature_names_in_", None)
    return list(cols) if cols is not None else None


def predict(raw: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    cols = get_input_cols(model)
    raw = raw.copy()
    if "poutcome" not in raw.columns:
        raw["poutcome"] = DEFAULT_POUTCOME

    X = legacy_preprocess(raw[FEATURE_COLS]) if cols is None else raw[cols]
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1].round(4)
    else:                       # e.g. an SVM trained without probability=True
        proba = np.full(len(raw), np.nan)

    out = raw.copy()
    out["probability_yes"] = proba
    out["prediction"] = np.where(pred == 1, "yes", "no")
    return out


@st.cache_data(show_spinner=False)
def load_mlflow_comparison() -> pd.DataFrame | None:
    """Final metrics of each model from mlflow.db (read with sqlite - mlflow is not needed)."""
    if not MLFLOW_DB.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{MLFLOW_DB}?mode=ro", uri=True)
        q = """
            SELECT COALESCE(t.value, e.name) AS Model, m.key AS metric, m.value
            FROM metrics m
            JOIN runs r ON r.run_uuid = m.run_uuid
            JOIN experiments e ON e.experiment_id = r.experiment_id
            LEFT JOIN tags t ON t.run_uuid = r.run_uuid AND t.key = 'model_name'
            WHERE m.key IN ('cv_accuracy', 'train_accuracy', 'test_accuracy', 'test_f1', 'test_roc_auc')
              AND r.run_uuid NOT IN (SELECT run_uuid FROM tags WHERE key = 'mlflow.parentRunId')
        """
        df = pd.read_sql_query(q, con)
        con.close()
        if df.empty:
            return None
        table = df.pivot_table(index="Model", columns="metric", values="value").reset_index()
        table = table.rename(columns={
            "cv_accuracy": "CV accuracy", "train_accuracy": "Train accuracy",
            "test_accuracy": "Test accuracy", "test_f1": "Test F1", "test_roc_auc": "Test ROC-AUC",
        })
        sort_col = "Test accuracy" if "Test accuracy" in table.columns else table.columns[1]
        return table.sort_values(sort_col, ascending=False).reset_index(drop=True)
    except Exception:
        return None


# ---- UI ----
st.set_page_config(page_title="Bank Term Deposit Predictor", page_icon="🏦", layout="wide")

st.title("🏦 Bank Term Deposit Predictor")
st.caption("Predict whether a customer will subscribe to a term deposit after a marketing call.")

missing = [p.name for p in (MODEL_PATH, DATA_PATH) if not p.exists()]
if missing:
    st.error(f"Missing required file(s): {', '.join(missing)}. Place them next to app.py.")
    st.stop()

df_ref = load_data()
model = load_model()
MODEL_LABEL = model_label(model)
MODEL_COLS = get_input_cols(model)                     # None -> legacy model
INPUT_COLS = MODEL_COLS if MODEL_COLS is not None else FEATURE_COLS
UI_COLS = [c for c in INPUT_COLS if c != "poutcome"]   # poutcome is not asked

tab_single, tab_batch, tab_info = st.tabs(["🔮 Single prediction", "📂 Batch prediction", "ℹ️ Model info"])


def render_input(col: str):
    """One input widget per model column."""
    if col == "age":
        return st.number_input("Age", 18, 100, 40)
    if col == "job":
        return st.selectbox("Job", sorted(df_ref["job"].unique()))
    if col == "marital":
        return st.selectbox("Marital status", sorted(df_ref["marital"].unique()))
    if col == "education":
        return st.selectbox("Education", EDU_OPTIONS, index=1)
    if col == "balance":
        return st.number_input("Average yearly balance (€)", -10000, 120000, 1500, step=100)
    if col == "default":
        return st.selectbox("Credit in default?", ["no", "yes"])
    if col == "housing":
        return st.selectbox("Housing loan?", ["no", "yes"])
    if col == "loan":
        return st.selectbox("Personal loan?", ["no", "yes"])
    if col == "contact":
        return st.selectbox("Contact type", sorted(df_ref["contact"].unique()))
    if col == "month":
        return st.selectbox("Last contact month", MONTHS, index=4)
    if col == "day":
        return st.number_input("Last contact day of month", 1, 31, 15)
    if col == "duration":
        return st.number_input("Last call duration (seconds)", 0, 5000, 250, step=10)
    if col == "campaign":
        return st.number_input("Contacts in this campaign", 1, 70, 2)
    if col == "pdays":
        return st.number_input(
            "Days since previous campaign contact (-1 = never)", -1, 900, -1,
            help="Use -1 if the customer was not previously contacted.",
        )
    if col == "previous":
        return st.number_input("Contacts before this campaign", 0, 60, 0)
    return None


# ------------------------------- Single prediction -------------------------------
with tab_single:
    st.subheader("Customer details")
    if "duration" in INPUT_COLS:
        st.caption("Call duration is only known after a call ends, so use this to score customers "
                   "after they have been contacted.")

    top_layout = [
        ("Personal", ["age", "job", "marital", "education"]),
        ("Financial", ["balance", "default", "housing", "loan"]),
        ("Campaign", ["contact", "month", "day", "duration"]),
    ]
    bottom_fields = ["campaign", "pdays", "previous"]

    values = {}
    with st.form("predict_form"):
        for column, (title, fields) in zip(st.columns(3), top_layout):
            with column:
                st.markdown(f"**{title}**")
                for f in fields:
                    if f in UI_COLS:
                        values[f] = render_input(f)
        for column, f in zip(st.columns(3), bottom_fields):
            with column:
                if f in UI_COLS:
                    values[f] = render_input(f)
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([values])
        result = predict(row).iloc[0]
        prob = result["probability_yes"]

        st.divider()
        left, right = st.columns([1, 2])
        with left:
            if result["prediction"] == "yes":
                st.success("✅ Likely to subscribe")
            else:
                st.error("❌ Unlikely to subscribe")
            if pd.notna(prob):
                st.metric("Probability of subscribing", f"{float(prob):.1%}")
        with right:
            if pd.notna(prob):
                st.progress(float(prob))
                st.caption(f"Predicted by the {MODEL_LABEL} model: 'yes' is predicted when the "
                           f"probability is 50% or higher.")
            else:
                st.caption(f"The {MODEL_LABEL} model gives a yes/no decision but no probability.")

# ------------------------------- Batch prediction -------------------------------
with tab_batch:
    st.subheader("Predict for many customers")
    required = [c for c in INPUT_COLS if c != "poutcome"]
    st.write("Upload a CSV with these columns (extra columns such as `deposit` are ignored):")
    st.code(", ".join(required))
    st.download_button(
        "Download sample CSV",
        df_ref[required].head(10).to_csv(index=False).encode(),
        file_name="sample_customers.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            absent = [c for c in required if c not in data.columns]
            if absent:
                st.error(f"Missing columns: {', '.join(absent)}")
            else:
                results = predict(data[required])
                st.success(f"Predicted {len(results):,} customers - "
                           f"{(results['prediction'] == 'yes').mean():.1%} likely to subscribe.")
                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "Download predictions",
                    results.to_csv(index=False).encode(),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process the file: {e}")

# ------------------------------- Model info -------------------------------
with tab_info:
    st.subheader("About the model")
    st.write(f"**Deployed model:** {MODEL_LABEL}")
    show_info = st.toggle("Show model details", value=False)   # only computed when switched on
    if show_info:
        if MODEL_COLS is not None:
            st.write("**Input columns used:** " + ", ".join(MODEL_COLS))
        else:
            st.caption("Legacy model: expects 48 pre-processed features; preprocessing is rebuilt from bank.csv.")
        st.write("**Pipeline:**")
        st.code(str(model), language="text")

        comp = load_mlflow_comparison()
        if comp is not None:
            st.write("**Experiment comparison (from MLflow)**")
            comp = comp.copy()
            comp.insert(1, "Deployed", np.where(comp["Model"] == MODEL_LABEL, "✅", ""))
            for col in ("CV accuracy", "Train accuracy", "Test accuracy"):
                if col in comp.columns:
                    comp[col] = (comp[col] * 100).round(2).astype(str) + "%"
            for col in ("Test F1", "Test ROC-AUC"):
                if col in comp.columns:
                    comp[col] = comp[col].round(4)
            st.dataframe(comp, use_container_width=True, hide_index=True)

    st.write("**Dataset overview**")
    a, b, c = st.columns(3)
    a.metric("Customers", f"{len(df_ref):,}")
    b.metric("Subscribed", f"{(df_ref['deposit'] == 'yes').mean():.1%}")
    c.metric("Input features", len(UI_COLS))