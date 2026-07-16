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

@st.cache_resource
def load_agri_assets():
    model = joblib.load("best_model_RandomForest.joblib")
    scaler = joblib.load("scaler.joblib")
    le = joblib.load("label_encoder.joblib")
    bench_df = pd.read_csv("crop_benchmarks_MASTER.csv")
    soil_master_df = pd.read_csv("regional_soil_MASTER.csv")
    coords_df = pd.read_csv("district_coords_MASTER.csv")
    coords_df["District"] = coords_df["District"].str.strip().str.lower()
    district_coords = {row["District"]: (row["Latitude"], row["Longitude"]) for _, row in coords_df.iterrows()}
    data_set = pd.read_csv("data_set.csv")
    grown_ok = data_set[(data_set["Area"] > 0) & (data_set["Production" ] > 0)]
    grown_in_state = set(zip(grown_ok["State"].str.upper().str.strip(), grown_ok["Crop"].str.strip())) # Corrected 'STATES' to 'State'
    dataset_map_reverse = {"rice": "Rice", "maize": "Maize", "chickpea": "Gram", "kidneybeans": "Rajmash Kholar", "pigeonpeas": "Arhar/Tur", "mothbeans": "Moth", "mungbean": "Moong", "blackgram": "Blackgram", "lentil": "Lentil", "pomegranate": "Pome Granet", "banana": "Banana", "mango": "Mango", "grapes": "Grapes", "watermelon": "Water Melon", "orange": "Orange", "papaya": "Papaya", "coconut": "Coconut ", "cotton": "Cotton(lint)", "jute": "Jute", "coffee": "Coffee"}
    training_url = "https://raw.githubusercontent.com/AbhishekKandoi/Crop-Yield-Prediction-based-on-Indian-Agriculture/main/Crop%20Recommendation%20dataset.csv"
    training_df = pd.read_csv(training_url)
    training_bounds = {col: (float(training_df[col].min()), float(training_df[col].max())) for col in FEATURES}
    return model, scaler, le, bench_df, soil_master_df, district_coords, training_bounds, grown_in_state, dataset_map_reverse

try:
    best_model, scaler, le, bench_df, soil_master_df, district_coords, training_bounds, grown_in_state, dataset_map_reverse = load_agri_assets()
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

def is_realistically_grown(state, crop_label):
    ds_crop_name = dataset_map_reverse.get(crop_label)
    if ds_crop_name is None: return True
    # Production code fix: Normalize 'And' to '&' for Jammu
    lookup_state = state.upper().strip().replace(' AND ', ' & ')
    lookup_state = "ORISSA" if lookup_state == "ODISHA" else lookup_state
    return (lookup_state, ds_crop_name) in grown_in_state

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

    mask = confs >= 5.0 # Lowered from 10 to catch low-confidence staple possibilities
    if mask.sum() == 0: mask[:3] = True
    cands, confs = cands[mask], confs[mask]

    shap_v = shap_explainer.shap_values(X_scaled)
    def get_shap(idx): return shap_v[idx][0] if isinstance(shap_v, list) else shap_v[0, :, idx]

    rows = []
    for c, cf in zip(cands, confs):
        b_row = bench_df[bench_df["label" ] == c].iloc[0]
        yf = np.clip(1.1 - (abs(clim["temperature"] - b_row["ideal_temp_c"]) / max(b_row["ideal_temp_c"], 1) + abs(clim["rainfall_37d_actual"] - b_row["ideal_rainfall_mm"]) / max(b_row["ideal_rainfall_mm"], 1)) / 2.0, 0.6, 1.1)
        est_y = b_row["avg_yield_t_ha"] * yf
        irr = max(b_row["water_req_mm"] - clim["rainfall_37d_actual"], 0)
        prof = (est_y * 1000 * b_row["price_per_kg_inr"]) - b_row["cost_cultivation_inr_ha"]
        s_row = get_shap(le.transform([c])[0]); d_idx = int(np.argmax(np.abs(s_row))); d_feat = FEATURES[d_idx]
        expl = SHAP_NARRATIVE_RULES[d_feat]["positive" if s_row[d_idx] >= 0 else "negative"]
        rows.append({"crop": c, "classifier_confidence": round(cf, 2), "estimated_yield_t_ha": round(est_y, 2), "water_requirement_mm": b_row["water_req_mm"], "irrigation_needed_mm": round(irr, 1), "estimated_profit_inr_ha": round(prof, 0), "dominant_driver": f"{d_feat}", "xai_explanation": expl})

    res = pd.DataFrame(rows)
    if res.empty: return res, soil, (N is None), clim, False

    total_w = w_confidence + w_yield + w_water + w_profit
    w_c, w_y, w_wa, w_p = w_confidence/total_w, w_yield/total_w, w_water/total_w, w_profit/total_w

    res["s_conf"] = res["classifier_confidence"] / 100.0
    res["s_yield"] = (res["estimated_yield_t_ha"] - bench_df["avg_yield_t_ha"].min()) / (bench_df["avg_yield_t_ha"].max() - bench_df["avg_yield_t_ha"].min())
    res["s_water"] = 1 - (res["irrigation_needed_mm"] / bench_df["water_req_mm"].max())
    res["s_profit"] = np.log1p(res["estimated_profit_inr_ha"].clip(lower=0)) / np.log1p((bench_df["avg_yield_t_ha"] * 1000 * bench_df["price_per_kg_inr"]).max())
    if clim["risk"] != "low": res["s_water"] *= (0.7 if clim["risk"] == "moderate" else 0.4)
    res["multi_objective_score"] = (w_c * res["s_conf" ] + w_y * res["s_yield"] + w_wa * res["s_water"] + w_p * res["s_profit"]).round(4)

    res["regionally_grown"] = res["crop"].apply(lambda c: is_realistically_grown(state, c))
    warn_flag = False
    if res["regionally_grown"].any(): res = res[res["regionally_grown"]].copy()
    else: warn_flag = True

    return res.sort_values("multi_objective_score", ascending=False).head(top_n).reset_index(drop=True), soil, (N is None), clim, warn_flag

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
    res, soil, fb, clim, reg_warn = multi_objective_recommend(sel_state, sel_dist, w_confidence=w_c, w_yield=w_y, w_water=w_wa, w_profit=w_p)
    if fb: st.info(f"📋 Fallback soil for {sel_dist}: N={soil['N']}, P={soil['P']}, K={soil['K']}, pH={soil['ph']}")
    st.info(f"🌦️ Live Climate: {clim['temperature']}°C, {clim['rainfall_37d_actual']}mm rain. Status: {clim['category']}.")
    if reg_warn: st.warning("⚠️ Theoretical climate matches shown (no historical records for these in state).")
    st.subheader("🏆 Recommendations")
    st.dataframe(res[["crop", "classifier_confidence", "estimated_yield_t_ha", "irrigation_needed_mm", "estimated_profit_inr_ha", "xai_explanation"]], use_container_width=True, hide_index=True)
    st.success(f"**Top Pick: {res.iloc[0]['crop'].title()}** — {res.iloc[0]['xai_explanation']}")
else:
    st.info("👈 Set your farm's details in the sidebar, then click **Get Crop Recommendations**.")
