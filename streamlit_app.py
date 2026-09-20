"""Streamlit UI. Run: streamlit run streamlit_app.py"""
import streamlit as st
from app import predict, inr, meta

st.set_page_config(page_title="Predict Property Price", layout="centered")
st.markdown("""<style>
.block-container{max-width:640px;padding-top:2rem}
#MainMenu,footer,header{visibility:hidden}
html,body,[class*="css"]{font-family:Verdana,'DejaVu Sans',sans-serif}
.hd{background:#1f3864;color:#fff;font-weight:bold;text-align:center;padding:16px;font-size:15px;border:1px solid #16294a;margin-bottom:14px}
.lbl{font-size:12px;font-weight:bold;color:#111;padding-top:10px}
div[data-baseweb="input"],div[data-baseweb="select"]>div{background:#f2f2f2!important;border-radius:0!important}
div[data-testid="stButton"]>button{width:100%;background:#2e7d32;color:#fff;font-weight:bold;border:1px solid #1f5d24;border-radius:0;padding:12px}
div[data-testid="stButton"]>button:hover{background:#276a2b;color:#fff;border:1px solid #1f5d24}
.res{margin-top:30px;background:#eaf2fc;border:2px solid #1f3864;text-align:center;padding:14px 0}
.res small{display:block;font-size:12px;color:#333}.res b{font-size:24px;color:#1f3864}
.cap{text-align:center;font-size:13px;margin-top:30px;color:#222}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="hd">Predict Property Price</div>', unsafe_allow_html=True)

def row(label, widget):
    a, b = st.columns([1, 1.6]); a.markdown(f'<div class="lbl">{label}</div>', unsafe_allow_html=True)
    with b: return widget()

locs = meta["localities"]
loc = row("Locality", lambda: st.selectbox("l", locs, index=locs.index("Rohini, Delhi"), label_visibility="collapsed"))
typ = row("Property Type", lambda: st.selectbox("t", meta["types"], label_visibility="collapsed"))
area = row("Built-up Area (sq.ft.)", lambda: st.number_input("a", 100, 10000, 1250, label_visibility="collapsed"))
age = row("Property Age (years)", lambda: st.number_input("g", 0, 100, 5, label_visibility="collapsed"))
metro = row("Metro Distance (km)", lambda: st.number_input("m", 0.0, 50.0, 1.2, 0.1, label_visibility="collapsed"))
am = row("Amenities", lambda: st.text_input("am", "Lift, Parking, Park-facing", label_visibility="collapsed"))

price = None
if st.button("Predict Price"):
    price = inr(predict(dict(locality=loc, property_type=typ, area_sqft=area,
                             age_years=age, metro_km=metro, amenities=am)))
st.markdown(f'<div class="res"><small>Predicted Price</small><b>{price or "—"}</b></div>', unsafe_allow_html=True)
st.markdown('<div class="cap">Fig 4.4 &nbsp;Price Prediction Input Form</div>', unsafe_allow_html=True)
