import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge

st.set_page_config(page_title="Predict Property Price", page_icon="🏠", layout="centered")

# ---------- Reference data (Delhi NCR) ----------
LOCALITY_PPSF = {  # base price per sq.ft (INR)
    "Rohini, Delhi": 6440, "Dwarka, Delhi": 7200, "Saket, Delhi": 13500,
    "Vasant Kunj, Delhi": 14000, "Janakpuri, Delhi": 9800, "Pitampura, Delhi": 8200,
    "Mayur Vihar, Delhi": 7600, "Gurugram Sector 56": 10500, "Noida Sector 62": 6900,
    "Greater Noida West": 4600, "Indirapuram, Ghaziabad": 5800, "Faridabad Sector 21": 4900,
}
TYPE_MULT = {"Flat / Apartment": 1.0, "Builder Floor": 1.12, "Independent House": 1.25,
             "Villa": 1.5, "Studio": 0.9}
AMEN_W = {"Lift": .02, "Parking": .03, "Park-facing": .03, "Gym": .02,
          "24x7 Security": .02, "Power Backup": .015, "Club House": .025}
AMENITIES = list(AMEN_W)


def features(df):
    amen_score = sum(w * df[a].astype(int) for a, w in AMEN_W.items())
    X = pd.DataFrame({
        "log_area": np.log(df["area"]),
        "log_age": np.log(1 - 0.01 * df["age"]),
        "log_metro": np.log(np.maximum(0.85, 1.08 - 0.05 * df["metro"])),
        "log_amen": np.log1p(amen_score),
    })
    for l in LOCALITY_PPSF:
        X["L_" + l] = (df["locality"] == l).astype(int)
    for t in TYPE_MULT:
        X["T_" + t] = (df["ptype"] == t).astype(int)
    return X


@st.cache_resource
def train_model(n=12000, seed=42):
    """Trains a Ridge regression (log-price) on synthetic Delhi NCR-style data."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "locality": rng.choice(list(LOCALITY_PPSF), n),
        "ptype": rng.choice(list(TYPE_MULT), n),
        "area": rng.uniform(350, 4000, n),
        "age": rng.integers(0, 35, n),
        "metro": rng.uniform(0.1, 12, n),
    })
    for a in AMENITIES:
        d[a] = rng.random(n) < 0.5
    ppsf = d["locality"].map(LOCALITY_PPSF) * d["ptype"].map(TYPE_MULT)
    ppsf *= (1 - 0.01 * d["age"]) * np.maximum(0.85, 1.08 - 0.05 * d["metro"])
    ppsf *= 1 + sum(w * d[a] for a, w in AMEN_W.items())
    price = d["area"] * ppsf * np.exp(rng.normal(0, 0.003, n))
    model = Ridge(alpha=1e-4).fit(features(d), np.log(price))
    return model


def predict(model, locality, ptype, area, age, metro, amenities):
    row = pd.DataFrame([{"locality": locality, "ptype": ptype, "area": area,
                         "age": age, "metro": metro,
                         **{a: a in amenities for a in AMENITIES}}])
    p = float(np.exp(model.predict(features(row))[0]))
    return round(p / 25000) * 25000


def inr(x):  # Indian digit grouping: 84,25,000
    s = str(int(x))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail])


# ---------- Styling (matches the mock-up) ----------
st.markdown("""
<style>
.block-container {max-width: 720px; padding-top: 2rem;}
.title-bar {background:#1F3864; color:#fff; text-align:center; font-weight:700;
            padding:14px; border-radius:2px; margin-bottom:22px; font-size:1.05rem;}
.lbl {font-weight:700; font-size:.85rem; padding-top:8px;}
div.stButton > button {background:#2E7D32; color:#fff; width:100%; font-weight:700;
            border:none; border-radius:2px; padding:.6rem 0; margin-top:8px;}
div.stButton > button:hover {background:#256628; color:#fff;}
.result {background:#EAF1FB; border:1.5px solid #1F3864; text-align:center;
         padding:14px; margin-top:18px; border-radius:2px;}
.result .s {font-size:.8rem; color:#333;}
.result .v {font-size:1.7rem; font-weight:800; color:#1F3864;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-bar">Predict Property Price</div>', unsafe_allow_html=True)


def row(label):
    c1, c2 = st.columns([1, 1.6])
    c1.markdown(f'<div class="lbl">{label}</div>', unsafe_allow_html=True)
    return c2


locality = row("Locality").selectbox("Locality", list(LOCALITY_PPSF), label_visibility="collapsed")
ptype = row("Property Type").selectbox("Property Type", list(TYPE_MULT), label_visibility="collapsed")
area = row("Built-up Area (sq.ft.)").number_input("Area", 200, 10000, 1250, 50, label_visibility="collapsed")
age = row("Property Age (years)").number_input("Age", 0, 60, 5, 1, label_visibility="collapsed")
metro = row("Metro Distance (km)").number_input("Metro", 0.0, 30.0, 1.2, 0.1, label_visibility="collapsed")
amen = row("Amenities").multiselect("Amenities", AMENITIES,
                                    default=["Lift", "Parking", "Park-facing"],
                                    label_visibility="collapsed")

if st.button("Predict Price"):
    price = predict(train_model(), locality, ptype, area, age, metro, amen)
    st.session_state["price"] = price

if "price" in st.session_state:
    st.markdown(f'<div class="result"><div class="s">Predicted Price</div>'
                f'<div class="v">₹ {inr(st.session_state["price"])}</div></div>',
                unsafe_allow_html=True)
    st.caption("Fig 4.4  Price Prediction Input Form")
