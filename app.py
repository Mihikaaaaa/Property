import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Predict Property Price", page_icon="🏠", layout="centered")

AMENITIES = {"lift": "Lift", "parking": "Parking", "park-facing": "Park-facing", "gym": "Gym",
             "security": "24x7 Security", "power backup": "Power Backup", "club": "Club House"}


# ---------- Data + model ----------
@st.cache_data
def load_clean():
    d = pd.read_csv("transactions.csv").drop_duplicates()
    d["metro_km"] = d["metro_km"].fillna(d["metro_km"].median())
    d["amenities"] = d["amenities"].fillna("")
    d["pps"] = d["price"] / d["area_sqft"]
    # remove price-per-sqft outliers within each locality (IQR rule)
    g = d.groupby("locality")["pps"]
    q1, q3 = g.transform(lambda s: s.quantile(.25)), g.transform(lambda s: s.quantile(.75))
    return d[(d.pps >= q1 - 1.5 * (q3 - q1)) & (d.pps <= q3 + 1.5 * (q3 - q1))].reset_index(drop=True)


def feats(d, localities, types):
    X = pd.DataFrame({"area": d["area_sqft"], "age": d["age_years"], "metro": d["metro_km"],
                      "year": d["txn_year"]})
    for k in AMENITIES:
        X[k] = d["amenities"].apply(lambda s: int(k in [t.strip() for t in s.split(",")]))
    X["locality"] = pd.Categorical(d["locality"], categories=localities).codes
    X["ptype"] = pd.Categorical(d["property_type"], categories=types).codes
    return X


@st.cache_resource
def train():
    d = load_clean()
    loc, typ = sorted(d.locality.unique()), sorted(d.property_type.unique())
    X, y = feats(d, loc, typ), np.log(d["price"])
    cat = [X.columns.get_loc("locality"), X.columns.get_loc("ptype")]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=.06, categorical_features=cat,
                                      random_state=42).fit(Xtr, ytr)
    p = m.predict(Xte)
    stats = {"r2": r2_score(np.exp(yte), np.exp(p)), "mae": mean_absolute_error(np.exp(yte), np.exp(p)),
             "n": len(d), "year": int(d.txn_year.max())}
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=.06, categorical_features=cat,
                                      random_state=42).fit(X, y)
    return m, loc, typ, stats


def inr(x):
    s = str(int(x)); head, tail, parts = s[:-3], s[-3:], []
    while len(head) > 2:
        parts.insert(0, head[-2:]); head = head[:-2]
    if head: parts.insert(0, head)
    return ",".join(parts + [tail])


model, LOCS, TYPES, S = train()

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {max-width: 760px; padding-top: 2.2rem;}
.title-bar {background:#1F3864; color:#fff; text-align:center; font-weight:700; font-size:1.1rem;
            padding:16px; border-radius:4px; margin-bottom:26px;}
.lbl {font-weight:700; font-size:.9rem; color:#1a1a1a; margin:0;}
div[data-testid="stHorizontalBlock"] {margin-bottom:.35rem;}
div.stButton > button {background:#2E7D32; color:#fff; width:100%; font-weight:700; border:none;
            border-radius:4px; padding:.7rem 0; margin-top:10px;}
div.stButton > button:hover {background:#256628; color:#fff; border:none;}
.result {background:#EAF1FB; border:1.5px solid #1F3864; text-align:center; padding:16px;
         margin-top:22px; border-radius:4px;}
.result .s {font-size:.85rem; color:#333;}
.result .v {font-size:1.9rem; font-weight:800; color:#1F3864;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-bar">Predict Property Price</div>', unsafe_allow_html=True)


def row(label):
    c1, c2 = st.columns([1, 1.7], vertical_alignment="center")
    c1.markdown(f'<p class="lbl">{label}</p>', unsafe_allow_html=True)
    return c2


locality = row("Locality").selectbox("Locality", LOCS, index=LOCS.index("Rohini, Delhi"), label_visibility="collapsed")
ptype = row("Property Type").selectbox("Type", TYPES, index=TYPES.index("Flat / Apartment"), label_visibility="collapsed")
area = row("Built-up Area (sq.ft.)").number_input("Area", 300, 5000, 1250, 50, label_visibility="collapsed")
age = row("Property Age (years)").number_input("Age", 0, 50, 5, 1, label_visibility="collapsed")
metro = row("Metro Distance (km)").number_input("Metro", 0.0, 25.0, 1.2, 0.1, label_visibility="collapsed")
amen = row("Amenities").multiselect("Amenities", list(AMENITIES.values()),
                                    default=["Lift", "Parking", "Park-facing"], label_visibility="collapsed")

if st.button("Predict Price"):
    inv = {v: k for k, v in AMENITIES.items()}
    row_df = pd.DataFrame([{"locality": locality, "property_type": ptype, "area_sqft": area,
                            "age_years": age, "metro_km": metro, "txn_year": S["year"],
                            "amenities": ", ".join(inv[a] for a in amen)}])
    p = float(np.exp(model.predict(feats(row_df, LOCS, TYPES))[0]))
    st.session_state["price"] = round(p / 1000) * 1000

if "price" in st.session_state:
    st.markdown(f'<div class="result"><div class="s">Predicted Price</div>'
                f'<div class="v">₹ {inr(st.session_state["price"])}</div></div>', unsafe_allow_html=True)

st.caption(f"Model: Gradient Boosting trained on {S['n']:,} cleaned transactions (2018–{S['year']}) · "
           f"R² = {S['r2']:.2f} · MAE ≈ ₹{inr(S['mae'])} · prices reflect {S['year']} market levels.")
