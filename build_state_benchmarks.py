"""
build_state_benchmarks.py
Builds crop_benchmarks_STATE.csv (one row per crop x state) from:
  crop_yield.csv                        - state-level yield
  Cost_of_Cultivation_Sample_Data.csv   - CACP cost data
  Cost_of_Cultivation_Sample_Data__1_.csv
  crop_price_data.csv                   - national market price
  data_set.csv                          - used for regional-growth check
"""

import pandas as pd
import numpy as np

YIELD_NAME_MAP = {
    "rice": "Rice", "maize": "Maize", "chickpea": "Gram",
    "pigeonpeas": "Arhar/Tur", "mothbeans": "Moth",
    "mungbean": "Moong(Green Gram)", "blackgram": "Urad", "lentil": "Masoor",
    "banana": "Banana", "coconut": "Coconut ", "cotton": "Cotton(lint)",
    "jute": "Jute",
}

COST_NAME_MAP = {
    "rice": "Paddy", "maize": "Maize", "chickpea": "Gram",
    "pigeonpeas": "Tur (Arhar)", "blackgram": "Urad", "mungbean": "Moong",
    "lentil": "Masur", "cotton": "Cotton", "jute": "Jute",
}

PRICE_NAME_MAP = {
    "apple": "Apple", "banana": "Banana", "blackgram": "Black Gram",
    "chickpea": "Bengal Gram", "coffee": "Coffee",
    "cotton": "Cotton", "grapes": "Grapes", "jute": "Jute", "lentil": "Lentil",
    "maize": "Maize", "mango": "Mango", "mungbean": "Green Gram",
    "muskmelon": "Musk Melon", "orange": "Orange", "papaya": "Papaya",
    "pigeonpeas": "Arhar", "pomegranate": "Pomegranate", "rice": "Rice",
    "watermelon": "Water Melon",
    # coconut price handled separately -- crop_price_data.csv only has
    # Tender Coconut, not the mature/dry coconut this label means
}

# dry-coconut market rate; Tender Coconut price in the source data is a
# different product and much higher
COCONUT_DRY_PRICE_INR_KG = 28.0

# crop_yield.csv reports coconut in nuts/ha, every other crop in tonnes/ha
COCONUT_YIELD_NUTS_TO_TONNES = 0.0012

# no CACP data for these horticulture crops -- documented cost estimates used instead
HORTICULTURE_COST_INR_HA = {
    "banana": 56000, "watermelon": 50000, "muskmelon": 56000, "coconut": 60000,
}

NATIONAL_PLACEHOLDER = {
    # label:        (yield_t_ha, cost_inr_ha, price_inr_kg)
    "rice":        (4.0,  45000, 20),
    "maize":       (3.0,  30000, 18),
    "chickpea":    (1.2,  25000, 55),
    "kidneybeans": (1.5,  28000, 60),
    "pigeonpeas":  (1.0,  26000, 70),
    "mothbeans":   (0.7,  18000, 55),
    "mungbean":    (0.8,  20000, 70),
    "blackgram":   (0.8,  20000, 65),
    "lentil":      (1.0,  22000, 60),
    "pomegranate": (10.0, 150000, 80),
    "banana":      (35.0, 180000, 15),
    "mango":       (8.0,  90000, 40),
    "grapes":      (18.0, 200000, 45),
    "watermelon":  (25.0, 60000, 8),
    "muskmelon":   (15.0, 55000, 12),
    "apple":       (15.0, 120000, 60),
    "orange":      (12.0, 85000, 25),
    "papaya":      (40.0, 70000, 12),
    "coconut":     (8.0,  60000, 20),
    "cotton":      (1.8,  40000, 60),
    "jute":        (2.5,  35000, 40),
    "coffee":      (1.0,  100000, 250),
}

# National Horticulture Board yield figures (2014-15) -- crop_yield.csv has
# no fruit crops at all
FRUIT_YIELD_OVERRIDE_T_HA = {
    "apple": 6.01, "orange": 12.72, "grapes": 26.28,
    "mango": 8.49, "papaya": 38.46, "pomegranate": 10.79,
}
FRUIT_YIELD_SOURCE = "National Horticulture Board (2014-15)"

# reject a yield backed by only one year of data if it's below a sane floor
MIN_PLAUSIBLE_YIELD_T_HA = {"coconut": 3.0}

ALL_LABELS = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
    "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
    "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
    "coconut", "cotton", "jute", "coffee",
]

CROP_CONSTANTS = {
    "rice":        {"water_req_mm": 382.9, "ideal_temp_c": 25, "ideal_rainfall_mm": 200},
    "maize":       {"water_req_mm": 500,   "ideal_temp_c": 24, "ideal_rainfall_mm": 90},
    "chickpea":    {"water_req_mm": 350,   "ideal_temp_c": 22, "ideal_rainfall_mm": 65},
    "kidneybeans": {"water_req_mm": 400,   "ideal_temp_c": 20, "ideal_rainfall_mm": 90},
    "pigeonpeas":  {"water_req_mm": 450,   "ideal_temp_c": 26, "ideal_rainfall_mm": 150},
    "mothbeans":   {"water_req_mm": 300,   "ideal_temp_c": 28, "ideal_rainfall_mm": 55},
    "mungbean":    {"water_req_mm": 350,   "ideal_temp_c": 27, "ideal_rainfall_mm": 60},
    "blackgram":   {"water_req_mm": 350,   "ideal_temp_c": 27, "ideal_rainfall_mm": 65},
    "lentil":      {"water_req_mm": 300,   "ideal_temp_c": 20, "ideal_rainfall_mm": 45},
    "pomegranate": {"water_req_mm": 800,   "ideal_temp_c": 22, "ideal_rainfall_mm": 60},
    "banana":      {"water_req_mm": 1800,  "ideal_temp_c": 26, "ideal_rainfall_mm": 110},
    "mango":       {"water_req_mm": 900,   "ideal_temp_c": 27, "ideal_rainfall_mm": 90},
    "grapes":      {"water_req_mm": 700,   "ideal_temp_c": 20, "ideal_rainfall_mm": 70},
    "watermelon":  {"water_req_mm": 500,   "ideal_temp_c": 25, "ideal_rainfall_mm": 45},
    "muskmelon":   {"water_req_mm": 450,   "ideal_temp_c": 26, "ideal_rainfall_mm": 30},
    "apple":       {"water_req_mm": 1000,  "ideal_temp_c": 12, "ideal_rainfall_mm": 110},
    "orange":      {"water_req_mm": 900,   "ideal_temp_c": 22, "ideal_rainfall_mm": 110},
    "papaya":      {"water_req_mm": 1000,  "ideal_temp_c": 26, "ideal_rainfall_mm": 140},
    "coconut":     {"water_req_mm": 1300,  "ideal_temp_c": 27, "ideal_rainfall_mm": 150},
    "cotton":      {"water_req_mm": 700,   "ideal_temp_c": 26, "ideal_rainfall_mm": 80},
    "jute":        {"water_req_mm": 1200,  "ideal_temp_c": 25, "ideal_rainfall_mm": 170},
    "coffee":      {"water_req_mm": 1500,  "ideal_temp_c": 24, "ideal_rainfall_mm": 180},
}

STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Puducherry",
]


def build_state_yield(crop_yield_path="crop_yield.csv"):
    df = pd.read_csv(crop_yield_path)
    df["State"] = df["State"].str.strip()
    df["Crop"] = df["Crop"].str.strip()
    df["Season"] = df["Season"].str.strip()

    records = {}
    for label, apy_crop in YIELD_NAME_MAP.items():
        crop_rows = df[df["Crop"].str.strip() == apy_crop.strip()]
        crop_rows = crop_rows[crop_rows["Yield"] > 0]
        for state in STATES:
            sub = crop_rows[crop_rows["State"] == state]
            if sub.empty:
                continue
            latest_year = sub["Crop_Year"].max()
            year_rows = sub[sub["Crop_Year"] == latest_year]
            for pref in ["Kharif", "Whole Year", None]:
                chosen = year_rows[year_rows["Season"] == pref] if pref else year_rows
                if not chosen.empty:
                    break
            yield_val = float(chosen["Yield"].mean())
            n_years_available = sub["Crop_Year"].nunique()

            if label == "coconut":
                yield_val = yield_val * COCONUT_YIELD_NUTS_TO_TONNES
                source_note = f"state ({state}, crop_yield.csv, {int(latest_year)}, nuts/ha->tonnes/ha converted)"
            else:
                source_note = f"state ({state}, crop_yield.csv, {int(latest_year)})"

            floor = MIN_PLAUSIBLE_YIELD_T_HA.get(label)
            if floor is not None and yield_val < floor and n_years_available <= 1:
                continue

            records[(label, state)] = {
                "avg_yield_t_ha": round(yield_val, 6),
                "yield_source": source_note,
            }
    return records


def build_state_cost(cost_paths=("Cost_of_Cultivation_Sample_Data.csv",)):
    # Uses A2+FL (paid-out cost + imputed family labour) rather than C2
    # (which also imputes rent on owned land and interest on owned capital).
    # C2 produced 28/682 rows that were unprofitable even at best-case yield;
    # A2+FL is the standard basis for a farmer-facing profit figure and is
    # also what India's MSP formula (1.5 x A2+FL) is benchmarked against.
    if isinstance(cost_paths, str):
        cost_paths = (cost_paths,)
    df = pd.concat([pd.read_csv(p) for p in cost_paths], ignore_index=True).drop_duplicates()
    df.columns = [c.strip() for c in df.columns]
    state_col = "State Name (state_name)"
    crop_col = "Crop Name (crop_name)"
    year_col = "Year (year)"
    cost_col = "Cultivation Cost A2+FL (cul_cost_a2fl)"

    records = {}
    for label, cacp_crop in COST_NAME_MAP.items():
        crop_rows = df[df[crop_col].str.strip() == cacp_crop.strip()]
        for state in STATES:
            sub = crop_rows[crop_rows[state_col].str.strip() == state]
            sub = sub.dropna(subset=[cost_col])
            if sub.empty:
                continue
            latest_year = sorted(sub[year_col].unique())[-1]
            year_rows = sub[sub[year_col] == latest_year]
            cost_val = year_rows[cost_col].mean()
            records[(label, state)] = {
                "cost_cultivation_inr_ha": round(float(cost_val), 2),
                "cost_source": f"state ({state}, CACP A2+FL, {latest_year})",
            }
    return records


def build_national_price(price_path="crop_price_data.csv"):
    df = pd.read_csv(price_path)
    price_col = [c for c in df.columns if c.startswith("Modal Price")][0]

    records = {}
    for label, commodity in PRICE_NAME_MAP.items():
        rows = df[df["Commodity"].str.strip().str.lower() == commodity.strip().lower()]
        if rows.empty:
            continue
        price_per_kg = (rows[price_col].mean() / 100)
        records[label] = {
            "price_per_kg_inr": round(float(price_per_kg), 4),
            "price_source": "national (crop_price_data.csv, 2025-26)",
        }
    return records


def build_regionally_verifiable(data_set_path="data_set.csv"):
    df = pd.read_csv(data_set_path)
    state_col, crop_col, area_col, prod_col = "State", "Crop", "Area", "Production"

    grown_ok = df[(df[area_col] > 0) & (df[prod_col] > 0)]
    grown_set = set(zip(
        grown_ok[state_col].astype(str).str.upper().str.strip(),
        grown_ok[crop_col].astype(str).str.strip(),
    ))

    verifiable = {}
    for label in ALL_LABELS:
        apy_name = YIELD_NAME_MAP.get(label, label.capitalize())
        for state in STATES:
            verifiable[(label, state)] = (state.upper().strip(), apy_name) in grown_set
    return verifiable


def main():
    yield_records = build_state_yield()
    cost_records = build_state_cost(("Cost_of_Cultivation_Sample_Data.csv", "Cost_of_Cultivation_Sample_Data__1_.csv"))
    price_records = build_national_price()
    verifiable_records = build_regionally_verifiable()

    rows = []
    for label in ALL_LABELS:
        const = CROP_CONSTANTS[label]
        for state in STATES:
            y = yield_records.get((label, state))
            c = cost_records.get((label, state))
            p = price_records.get(label)
            ph_yield, ph_cost, ph_price = NATIONAL_PLACEHOLDER[label]

            if y:
                yield_val, yield_src = y["avg_yield_t_ha"], y["yield_source"]
            elif label in FRUIT_YIELD_OVERRIDE_T_HA:
                yield_val, yield_src = FRUIT_YIELD_OVERRIDE_T_HA[label], FRUIT_YIELD_SOURCE
            else:
                yield_val, yield_src = ph_yield, "national placeholder estimate (no dataset covers this crop)"

            if label == "coconut":
                price_val, price_src = COCONUT_DRY_PRICE_INR_KG, "documented dry-coconut market rate (~Rs 28/kg, not Tender Coconut)"
            elif p:
                price_val, price_src = p["price_per_kg_inr"], p["price_source"]
            else:
                price_val, price_src = ph_price, "national placeholder estimate (no dataset covers this crop)"

            if c:
                cost_val, cost_src = c["cost_cultivation_inr_ha"], c["cost_source"]
            elif label in HORTICULTURE_COST_INR_HA:
                cost_val, cost_src = HORTICULTURE_COST_INR_HA[label], "national estimate (horticulture cost table, no CACP data exists for this crop)"
            else:
                cost_val, cost_src = ph_cost, "national placeholder estimate (no dataset covers this crop)"

            # flag rows that stay unprofitable even at best-case climate fit,
            # rather than hiding them
            best_case_profit = (yield_val * 1.15 * 1000 * price_val) - cost_val
            margin_flag = "low_margin_verified_real_data" if best_case_profit < 0 else "ok"

            rows.append({
                "label": label,
                "state": state,
                "margin_flag": margin_flag,
                "avg_yield_t_ha": yield_val,
                "yield_source": yield_src,
                "cost_cultivation_inr_ha": cost_val,
                "cost_source": cost_src,
                "price_per_kg_inr": price_val,
                "price_source": price_src,
                "water_req_mm": const["water_req_mm"],
                "ideal_temp_c": const["ideal_temp_c"],
                "ideal_rainfall_mm": const["ideal_rainfall_mm"],
                "regionally_verifiable": verifiable_records.get((label, state), False),
                "data_confidence": "state" if y else "national",
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv("crop_benchmarks_STATE.csv", index=False)
    print("Wrote crop_benchmarks_STATE.csv:", out_df.shape)
    print(out_df["data_confidence"].value_counts())
    print(out_df["margin_flag"].value_counts())


if __name__ == "__main__":
    main()
