import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import requests

st.set_page_config(page_title="AgriGenie AI Dashboard", layout="wide", page_icon="🌾")
st.title("🌾 AgriGenie: Multi-Objective Crop Recommendation System")
st.markdown("""
**XAI-Driven Per-Farm Agricultural Optimisation Pipeline**
This decision-support matrix balances competing agronomic vectors: **Maximising Yield**, **Minimising Irrigation Deficits**, and **Maximising Net Farm Profits**.
""")

district_name_mapping = {
    'ahilyanagar': 'ahmednagar', 'ahmedabad': 'ahmadabad',
    'alluri sitharama raju': 'visakhapatnam', 'ananthapuramu': 'anantapur',
    'annamayya': 'chittoor', 'arvalli': 'aravallis', 'bandipora': 'bandipore',
    'dangs': 'the dangs', 'east singhbum': 'east singhbhum',
    'kaimur (bhabua)': 'kaimur', 'khandwa (east nimar)': 'khandwa',
    'khargone (west nimar)': 'khargone', 'kotputli-behror': 'jaipur',
    'mauganj': 'rewa', 'narsimhapur': 'narsinghpur',
    'north 24 parganas': 'north twenty four parganas', 'palnadu': 'guntur',
    'pratapgarh': 'pratapgarh uttar pradesh', 'sarangarh-bilaigarh': 'raigarh',
    'south 24 parganas': 'south twenty four parganas',
    'sri potti sriramulu nellore': 'spsr nellore', 'vijayanagara': 'ballari',
}

SHAP_NARRATIVE_RULES = {
    "N": {"positive": "The local soil Nitrogen level provides an excellent macronutrient baseline for vegetative canopy growth.", "negative": "Nitrogen limitations may require targeted top-dressing adjustments."},
    "P": {"positive": "Ample Phosphorus ensures strong seedling root structural development.", "negative": "Suboptimal soil Phosphorus may delay maturity."},
    "K": {"positive": "Robust Potassium availability enhances disease resistance.", "negative": "Low Potassium levels can impact overall cell wall stability."},
    "temperature": {"positive": "Current ambient seasonal temperatures align with the ideal metabolic window.", "negative": "Severe temperature anomalies could induce thermal stress."},
    "humidity": {"positive": "Prevailing relative humidity matches necessary atmospheric vapor pressure.", "negative": "Imbalanced humidity increases fungal pathogen risks."},
    "ph": {"positive": "The current soil pH guarantees ideal nutrient bioavailability.", "negative": "Imbalanced soil pH limits chemical uptake."},
    "rainfall": {"positive": "Excellent natural rainfall volumes fulfill seasonal requirements.", "negative": "Insufficient rainfall creates a moisture deficit."}
}
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ---------------------------------------------------------------------------
# FIX #1: crop-label -> government-dataset-name mapping.
# Every value here was checked against the ACTUAL unique Crop values in
# India_Agriculture_Crop_Production.csv (not assumed/typed from memory like
# before -- "Pome Granet", "Blackgram", "Moong", "Lentil", "Coconut " with a
# trailing space were all silently wrong and made those crops fail the
# regional-growth check in EVERY state, always).
#
# Crops in NO_REGIONAL_DATA genuinely do not appear anywhere in India's
# APY (Area-Production-Yield) statistics -- that dataset only tracks
# foodgrains/major commercial crops, not fruits, and not coffee. For these
# we cannot verify "is this grown here" from government production data at
# all, so we say that explicitly instead of silently guessing True or False.
# ---------------------------------------------------------------------------
CROP_TO_APY_NAME = {
    "rice": "Rice", "maize": "Maize", "chickpea": "Gram",
    "pigeonpeas": "Arhar/Tur", "mothbeans": "Moth",
    "mungbean": "Moong(Green Gram)", "blackgram": "Urad", "lentil": "Masoor",
    "banana": "Banana", "coconut": "Coconut", "cotton": "Cotton(lint)",
    "jute": "Jute",
    "apple": "Apple", "mango": "Mango", "grapes": "Grapes",
    "orange": "Orange", "papaya": "Papaya", "pomegranate": "Pomegranate",
}

# Horticulture Statistics Division, DAC&FW (2019-20) -- states with Production > 0
WATERMELON_STATES = {
    "UTTAR PRADESH", "ANDHRA PRADESH", "TAMIL NADU", "MADHYA PRADESH", "KARNATAKA",
    "ODISHA", "WEST BENGAL", "TELANGANA", "HARYANA", "MAHARASHTRA",
    "CHHATTISGARH", "BIHAR", "RAJASTHAN", "PUNJAB", "TRIPURA",
}

# Coffee Board of India (2025-26 Final Estimate) + coffee export industry sourcing data
COFFEE_STATES = {
    "KARNATAKA", "KERALA", "TAMIL NADU", "ANDHRA PRADESH", "ODISHA",
    "ASSAM", "MEGHALAYA", "MANIPUR", "MIZORAM", "NAGALAND",
}

# ICAR-Indian Institute of Pulses Research, Kanpur (Agri Journal World, 2023)
KIDNEYBEANS_STATES = {
    "MAHARASHTRA", "KARNATAKA", "TAMIL NADU", "KERALA", "HIMACHAL PRADESH",
    "UTTARAKHAND", "JAMMU AND KASHMIR", "GUJARAT", "WEST BENGAL",
}

# 2023-24 production data (Horticulture Dept-sourced reporting, cross-confirmed
# across DesiKheti/BigHaat/India Inputs)
MUSKMELON_STATES = {
    "UTTAR PRADESH", "ANDHRA PRADESH", "MADHYA PRADESH", "PUNJAB", "HARYANA",
    "CHHATTISGARH", "TAMIL NADU", "MAHARASHTRA", "TELANGANA", "RAJASTHAN",
}

# nothing left with zero data now -- all 22 model crops have some verification source
NO_REGIONAL_DATA = set()
@st.cache_resource
def load_agri_assets():
    model = joblib.load("best_model_RandomForest.joblib")
    scaler = joblib.load("scaler.joblib")
    le = joblib.load("label_encoder.joblib")
    # FIX #2: use the state-aware benchmark table (see build_state_benchmarks.py)
    # instead of one national-average row per crop. Falls back gracefully if
    # you haven't generated it yet, so the app never hard-crashes.
    try:
        bench_df = pd.read_csv("crop_benchmarks_STATE.csv")
        state_aware = True
    except FileNotFoundError:
        bench_df = pd.read_csv("crop_benchmarks_MASTER.csv")
        bench_df["state"] = None
        bench_df["data_confidence"] = "national"
        state_aware = False
    soil_master_df = pd.read_csv("regional_soil_MASTER.csv")
    coords_df = pd.read_csv("district_coords_MASTER.csv")
    coords_df["District"] = coords_df["District"].str.strip().str.lower()
    district_coords = {row["District"]: (row["Latitude"], row["Longitude"]) for _, row in coords_df.iterrows()}
    data_set = pd.read_csv("data_set.csv")
    grown_ok = data_set[(data_set["Area"] > 0) & (data_set["Production"] > 0)]
    grown_in_state = set(zip(grown_ok["State"].str.upper().str.strip(), grown_ok["Crop"].str.strip()))
    training_url = "https://raw.githubusercontent.com/AbhishekKandoi/Crop-Yield-Prediction-based-on-Indian-Agriculture/main/Crop%20Recommendation%20dataset.csv"
    training_df = pd.read_csv(training_url)
    training_bounds = {col: (float(training_df[col].min()), float(training_df[col].max())) for col in FEATURES}
    return model, scaler, le, bench_df, soil_master_df, district_coords, training_bounds, grown_in_state, state_aware

try:
    best_model, scaler, le, bench_df, soil_master_df, district_coords, training_bounds, grown_in_state, state_aware = load_agri_assets()
    shap_explainer = shap.TreeExplainer(best_model)
except Exception as e:
    st.error(f"⚠️ Missing assets! {e}")
    st.stop()

SOIL_PROFILE_DB = {}
for _, row in soil_master_df.iterrows():
    st_name, dt_name = str(row['State']).strip(), str(row['District']).strip()
    SOIL_PROFILE_DB.setdefault(st_name, {})[dt_name] = {"N": float(row['N']) if not pd.isna(row['N']) else 55.0, "P": float(row['P']) if not pd.isna(row['P']) else 30.0, "K": float(row['K']) if not pd.isna(row['K']) else 50.0, "ph": float(row['ph']) if not pd.isna(row['ph']) else 6.5}
SOIL_PROFILE_DB["default"] = {"N": 50.0, "P": 30.0, "K": 50.0, "ph": 6.5}

def get_coords(district):
    key = district_name_mapping.get(district.strip().lower(), district.strip().lower())
    coords = district_coords.get(key)
    if coords is None: raise ValueError(f"No coordinates for {district}")
    return coords

def fetch_live_climate(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current": "temperature_2m,relative_humidity_2m", "daily": "precipitation_sum", "past_days": 30, "forecast_days": 7, "timezone": "auto"}
    r = requests.get(url, params=params, timeout=15); r.raise_for_status(); data = r.json()
    return data["current"]["temperature_2m"], data["current"]["relative_humidity_2m"], round(sum(x for x in data["daily"]["precipitation_sum"] if x is not None), 1)

def fetch_historical_normal(lat, lon):
    url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
    params = {"latitude": lat, "longitude": lon, "community": "AG", "parameters": "T2M,PRECTOTCORR", "format": "JSON", "start": 2010, "end": 2023}
    r = requests.get(url, params=params, timeout=20); r.raise_for_status(); data = r.json()["properties"]["parameter"]
    return data["T2M"]["ANN"], round(data["PRECTOTCORR"]["ANN"] * 365, 1)

def climate_risk_assessment(district):
    try:
        lat, lon = get_coords(district)
        norm_t, norm_ann_r = fetch_historical_normal(lat, lon)
        t, hum, rain_37d = fetch_live_climate(lat, lon)
        norm_37d = norm_ann_r * (37 / 365)
        pct_dep = ((rain_37d - norm_37d) / norm_37d * 100) if norm_37d > 0 else 0.0
        cat = "Excess" if pct_dep > 50 else "Deficient" if pct_dep < -30 else "Normal"
        risk = "high" if pct_dep < -30 else "moderate" if pct_dep < -10 else "low"
        return {"district": district, "temperature": t, "humidity": hum, "rainfall_37d_actual": rain_37d, "rainfall_37d_normal": round(norm_37d, 1), "pct_departure": round(pct_dep, 1), "category": cat, "risk": risk, "source": "live"}
    except Exception as e: return {"district": district, "temperature": 27.0, "humidity": 65.0, "rainfall_37d_actual": 150.0, "rainfall_37d_normal": 150.0, "pct_departure": 0.0, "category": "Unknown", "risk": "low", "source": f"fallback ({e})"}

def clip_to_training_range(feature_name, value):
    lo, hi = training_bounds[feature_name]
    return float(np.clip(value, lo, hi))

def check_regional_growth(state, crop_label):
    if crop_label in NO_REGIONAL_DATA:
        return "unverifiable"
    lookup_state = state.upper().strip()
    special_lists = {
        "watermelon": WATERMELON_STATES,
        "coffee": COFFEE_STATES,
        "kidneybeans": KIDNEYBEANS_STATES,
        "muskmelon": MUSKMELON_STATES,
    }
    if crop_label in special_lists:
        return "grown" if lookup_state in special_lists[crop_label] else "not_grown"
    apy_name = CROP_TO_APY_NAME.get(crop_label)
    if apy_name is None:
        return "unverifiable"
    return "grown" if (lookup_state, apy_name) in grown_in_state else "not_grown"

def multi_objective_recommend(state, district, N=None, P=None, K=None, ph=None, top_n=5, w_confidence=0.5, w_yield=0.05, w_water=0.4, w_profit=0.05):
    state_data = SOIL_PROFILE_DB.get(state, SOIL_PROFILE_DB["default"])
    profile = state_data.get(district, state_data.get("default", SOIL_PROFILE_DB["default"]))
    soil = {k: (v if (v and v > 0) else profile[k]) for k, v in {"N": N, "P": P, "K": K, "ph": ph}.items()}
    clim = climate_risk_assessment(district)
    X_input = np.array([[clip_to_training_range(f, v) for f, v in zip(FEATURES, [soil["N"], soil["P"], soil["K"], clim["temperature"], clim["humidity"], soil["ph"], clim["rainfall_37d_actual"]])]])
    X_scaled = scaler.transform(X_input)
    probs = best_model.predict_proba(X_scaled)[0] * 100
    c_idx = np.argsort(probs)[::-1][:12]
    cands, confs = le.inverse_transform(c_idx), probs[c_idx]

    # Keep the raw #1 model pick around even if it later gets excluded --
    # this is what surfaces the "coffee at 61% in Bihar" situation instead
    # of hiding it, which matters for an XAI paper.
    raw_top_crop, raw_top_conf = cands[0], confs[0]

    mask = confs >= 5.0
    if mask.sum() == 0: mask[:3] = True
    cands, confs = cands[mask], confs[mask]

    shap_v = shap_explainer.shap_values(X_scaled)
    def get_shap(idx): return shap_v[idx][0] if isinstance(shap_v, list) else shap_v[0, :, idx]

    rows = []
    for c, cf in zip(cands, confs):
        b_rows = bench_df[(bench_df["label"] == c) & ((bench_df["state"] == state) | bench_df["state"].isna())]
        # prefer the state-specific row if present, else the national-fallback row for this crop
        b_row = b_rows[b_rows["state"].str.strip().str.lower() == state.strip().lower()]
        b_row = b_row.iloc[0] if not b_row.empty else bench_df[bench_df["label"] == c].iloc[0]
        # crop_yield.csv reports coconut in nuts/hectare, not tonnes/hectare like every
        # other crop in that file -- using it as-is inflates profit by ~1000x. Convert
        # using a standard ~1.2 kg average nut weight (0.0012 t/nut) to get a real t/ha figure.
        if c == "coconut" and str(b_row.get("yield_source", "")).startswith("state"):
            b_row = b_row.copy()
            b_row["avg_yield_t_ha"] = b_row["avg_yield_t_ha"] * 0.0012

        yf = np.clip(1.1 - (abs(clim["temperature"] - b_row["ideal_temp_c"]) / max(b_row["ideal_temp_c"], 1) + abs(clim["rainfall_37d_actual"] - b_row["ideal_rainfall_mm"]) / max(b_row["ideal_rainfall_mm"], 1)) / 2.0, 0.6, 1.1)
        est_y = b_row["avg_yield_t_ha"] * yf
        irr = max(b_row["water_req_mm"] - clim["rainfall_37d_actual"], 0)
        prof = (est_y * 1000 * b_row["price_per_kg_inr"]) - b_row["cost_cultivation_inr_ha"]
        s_row = get_shap(le.transform([c])[0]); d_idx = int(np.argmax(np.abs(s_row))); d_feat = FEATURES[d_idx]
        expl = SHAP_NARRATIVE_RULES[d_feat]["positive" if s_row[d_idx] >= 0 else "negative"]
        growth_status = check_regional_growth(state, c)
        rows.append({
            "crop": c, "classifier_confidence": round(cf, 2), "estimated_yield_t_ha": round(est_y, 2),
            "water_requirement_mm": b_row["water_req_mm"], "irrigation_needed_mm": round(irr, 1),
            "estimated_profit_inr_ha": round(prof, 0), "dominant_driver": f"{d_feat}", "xai_explanation": expl,
            "growth_status": growth_status,
            "data_confidence": b_row.get("data_confidence", "national"),
        })

    res = pd.DataFrame(rows)
    if res.empty: return res, soil, (N is None), clim, {"raw_top_crop": raw_top_crop, "raw_top_conf": raw_top_conf, "suppressed": False}

    total_w = w_confidence + w_yield + w_water + w_profit
    w_c, w_y, w_wa, w_p = w_confidence/total_w, w_yield/total_w, w_water/total_w, w_profit/total_w

    res["s_conf"] = res["classifier_confidence"] / 100.0
    res["s_yield"] = (res["estimated_yield_t_ha"] - bench_df["avg_yield_t_ha"].min()) / (bench_df["avg_yield_t_ha"].max() - bench_df["avg_yield_t_ha"].min())
    res["s_water"] = 1 - (res["irrigation_needed_mm"] / bench_df["water_req_mm"].max())
    res["s_profit"] = np.log1p(res["estimated_profit_inr_ha"].clip(lower=0)) / np.log1p((bench_df["avg_yield_t_ha"] * 1000 * bench_df["price_per_kg_inr"]).max())
    if clim["risk"] != "low": res["s_water"] *= (0.7 if clim["risk"] == "moderate" else 0.4)
    res["multi_objective_score"] = (w_c * res["s_conf"] + w_y * res["s_yield"] + w_wa * res["s_water"] + w_p * res["s_profit"]).round(4)

    # FIX #4: only hard-filter out crops we could ACTUALLY verify as not
    # grown here. Crops we simply have no data for ("unverifiable") stay in
    # the pool but are labeled as such in the UI -- never silently dropped,
    # never silently trusted either.
    before = res.copy()
    verified_pool = res[res["growth_status"] != "not_grown"].copy()
    if verified_pool.empty:
        # every single candidate failed verification -- fall back to the
        # unfiltered pool but make the reason explicit (not "theoretical
        # climate", which was the wrong message before).
        final = before
        no_verified_crops = True
    else:
        final = verified_pool
        no_verified_crops = False

    final = final.sort_values("multi_objective_score", ascending=False).head(top_n).reset_index(drop=True)

    meta = {
        "raw_top_crop": raw_top_crop, "raw_top_conf": round(float(raw_top_conf), 2),
        "suppressed": raw_top_crop not in final["crop"].values,
        "no_verified_crops": no_verified_crops,
    }
    return final, soil, (N is None), clim, meta

# --- Sidebar UI ---
st.sidebar.header("📍 Location")
sel_state = st.sidebar.selectbox("State", sorted([s for s in SOIL_PROFILE_DB.keys() if s != "default"]))
sel_dist = st.sidebar.selectbox("District", sorted(list(SOIL_PROFILE_DB[sel_state].keys())))

st.sidebar.header("⚖️ Priorities")
w_c = st.sidebar.slider("Suitability", 0.0, 1.0, 0.4)
w_y = st.sidebar.slider("Yield", 0.0, 1.0, 0.15)
w_wa = st.sidebar.slider("Water", 0.0, 1.0, 0.3)
w_p = st.sidebar.slider("Profit", 0.0, 1.0, 0.15)

if st.sidebar.button("🌾 Recommend", type="primary", use_container_width=True):
    res, soil, fb, clim, meta = multi_objective_recommend(sel_state, sel_dist, w_confidence=w_c, w_yield=w_y, w_water=w_wa, w_profit=w_p)
    if fb: st.info(f"📋 Fallback soil for {sel_dist}: N={soil['N']}, P={soil['P']}, K={soil['K']}, pH={soil['ph']}")
    st.info(f"🌦️ Live Climate: {clim['temperature']}°C, {clim['rainfall_37d_actual']}mm rain. Status: {clim['category']}.")

    if meta.get("no_verified_crops"):
        st.warning("⚠️ None of the model's candidate crops could be confirmed as commercially grown in this state from government production records. Results below are the raw model output, unverified against real-world cultivation data — treat with caution.")
    if meta.get("suppressed"):
        st.caption(f"ℹ️ Note: the model's single highest statistical match was **{meta['raw_top_crop']}** ({meta['raw_top_conf']}% confidence), but it does not appear in government production records for {sel_state} and was excluded from the ranked list below.")

    st.subheader("🏆 Recommendations")
    display_cols = ["crop", "classifier_confidence", "estimated_yield_t_ha", "irrigation_needed_mm", "estimated_profit_inr_ha", "growth_status", "data_confidence", "xai_explanation"]
    st.dataframe(res[display_cols], use_container_width=True, hide_index=True)
    st.success(f"**Top Pick: {res.iloc[0]['crop'].title()}** — {res.iloc[0]['xai_explanation']}")
else:
    st.info("👈 Set your farm's details in the sidebar, then click **Get Crop Recommendations**.")
