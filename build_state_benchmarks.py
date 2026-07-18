"""
build_state_benchmarks.py
==========================
"""

import pandas as pd
import numpy as np


YIELD_NAME_MAP = {           # label -> name used in crop_yield.csv "Crop" column
    "rice": "Rice", "maize": "Maize", "chickpea": "Gram",
    "pigeonpeas": "Arhar/Tur", "mothbeans": "Moth",
    "mungbean": "Moong(Green Gram)", "blackgram": "Urad", "lentil": "Masoor",
    "banana": "Banana", "coconut": "Coconut ", "cotton": "Cotton(lint)",
    "jute": "Jute",
}

COST_NAME_MAP = {            # label -> name used in Cost_of_Cultivation "Crop Name (crop_name)"
    "rice": "Paddy", "maize": "Maize", "chickpea": "Gram",
    "pigeonpeas": "Tur (Arhar)", "blackgram": "Urad", "mungbean": "Moong",
    "lentil": "Masur", "cotton": "Cotton", "jute": "Jute",
}

PRICE_NAME_MAP = {           # label -> "Commodity" in crop_price_data.csv
    "apple": "Apple", "banana": "Banana", "blackgram": "Black Gram",
    "chickpea": "Bengal Gram", "coffee": "Coffee",
    "cotton": "Cotton", "grapes": "Grapes", "jute": "Jute", "lentil": "Lentil",
    "maize": "Maize", "mango": "Mango", "mungbean": "Green Gram",
    "muskmelon": "Musk Melon", "orange": "Orange", "papaya": "Papaya",
    "pigeonpeas": "Arhar", "pomegranate": "Pomegranate", "rice": "Rice",
    "watermelon": "Water Melon",
    # coconut deliberately excluded here -- crop_price_data.csv only has "Tender
    # Coconut" (young drinking coconut), a different product from the mature/dry
    # coconut this crop label represents. See COCONUT_DRY_PRICE_INR_KG override below.
    # kidneybeans / mothbeans genuinely have no reliable national price series here
}


COCONUT_DRY_PRICE_INR_KG = 28.0


COCONUT_YIELD_NUTS_TO_TONNES = 0.0012


HORTICULTURE_COST_INR_HA = {
    "banana": 56000,
    "watermelon": 50000,
    "muskmelon": 56000,
    "coconut": 60000,   # CACP has no coconut rows either; documented plantation-cost estimate
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


FRUIT_YIELD_OVERRIDE_T_HA = {
    "apple": 6.01,
    "orange": 12.72,   # sourced from broader "Citrus" category -- disclose in paper
    "grapes": 26.28,
    "mango": 8.49,
    "papaya": 38.46,
    "pomegranate": 10.79,
}
FRUIT_YIELD_SOURCE = "Indian Horticulture Database / National Horticulture Board (2014-15)"

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

STATES = [  # the 31 states/UTs your app's dropdown covers -- edit to match regional_soil_MASTER.csv exactly
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir",
    "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra",
    "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Puducherry",
]


def build_state_yield(crop_yield_path="crop_yield.csv"):
    """Real state-level yield: most recent year available per (state, crop),
    preferring the Kharif season row when more than one season exists that year."""
    df = pd.read_csv(crop_yield_path)
    df["State"] = df["State"].str.strip()
    df["Crop"] = df["Crop"].str.strip()
    df["Season"] = df["Season"].str.strip()

    records = {}
    for label, apy_crop in YIELD_NAME_MAP.items():
        crop_rows = df[df["Crop"].str.strip() == apy_crop.strip()]
        crop_rows = crop_rows[crop_rows["Yield"] > 0]  # drop zero-yield rows (bad/missing data, not a real signal)
        for state in STATES:
            sub = crop_rows[crop_rows["State"] == state]
            if sub.empty:
                continue
            latest_year = sub["Crop_Year"].max()
            year_rows = sub[sub["Crop_Year"] == latest_year]
            # prefer Kharif > Whole Year > whatever else is there
            for pref in ["Kharif", "Whole Year", None]:
                chosen = year_rows[year_rows["Season"] == pref] if pref else year_rows
                if not chosen.empty:
                    break
            yield_val = float(chosen["Yield"].mean())

            if label == "coconut":
                # source reports nuts/ha, not tonnes/ha -- convert (see COCONUT_YIELD_NUTS_TO_TONNES)
                yield_val = yield_val * COCONUT_YIELD_NUTS_TO_TONNES
                source_note = f"state ({state}, crop_yield.csv, {int(latest_year)}, nuts/ha->tonnes/ha converted)"
            else:
                source_note = f"state ({state}, crop_yield.csv, {int(latest_year)})"

            records[(label, state)] = {
                "avg_yield_t_ha": round(yield_val, 6),
                "yield_source": source_note,
            }
    return records


def build_state_cost(cost_path="Cost_of_Cultivation_Sample_Data.csv"):
    """Real state-level cost of cultivation using CACP C2 (full economic cost),
    most recent year available per (state, crop)."""
    df = pd.read_csv(cost_path)
    df.columns = [c.strip() for c in df.columns]
    state_col = "State Name (state_name)"
    crop_col = "Crop Name (crop_name)"
    year_col = "Year (year)"
    c2_col = "Cultivation Cost C2 (cul_cost_c2)"

    records = {}
    for label, cacp_crop in COST_NAME_MAP.items():
        crop_rows = df[df[crop_col].str.strip() == cacp_crop.strip()]
        for state in STATES:
            sub = crop_rows[crop_rows[state_col].str.strip() == state]
            sub = sub.dropna(subset=[c2_col])
            if sub.empty:
                continue
            latest_year = sorted(sub[year_col].unique())[-1]
            year_rows = sub[sub[year_col] == latest_year]
            cost_val = year_rows[c2_col].mean()
            records[(label, state)] = {
                "cost_cultivation_inr_ha": round(float(cost_val), 2),
                "cost_source": f"state ({state}, CACP C2, {latest_year})",
            }
    return records


def build_national_price(price_path="crop_price_data.csv"):
    """National price (no state split available in source data) -- averages
    the two most recent years present (2025 & 2026) per commodity."""
    df = pd.read_csv(price_path)
    price_col = [c for c in df.columns if c.startswith("Modal Price")][0]

    records = {}
    for label, commodity in PRICE_NAME_MAP.items():
        rows = df[df["Commodity"].str.strip().str.lower() == commodity.strip().lower()]
        if rows.empty:
            continue
        price_per_kg = (rows[price_col].mean() / 100)  # Rs/Quintal -> Rs/kg
        records[label] = {
            "price_per_kg_inr": round(float(price_per_kg), 4),
            "price_source": "national (crop_price_data.csv, 2025-26)",
        }
    return records


def build_regionally_verifiable(data_set_path="data_set.csv"):
    """Cross-check each (label, state) against your main crop dataset:
    True only if Area>0 and Production>0 exists for that state+crop.
    NOTE: adjust the column names below (state_col/crop_col/area_col/prod_col)
    to match whichever data_set.csv you're currently using in the repo --
    they've changed across our sessions."""
    df = pd.read_csv(data_set_path)
    # --- EDIT THESE FOUR NAMES to match your current data_set.csv columns ---
    state_col, crop_col, area_col, prod_col = "STATES", "Crop", "Area", "Production"
    # -------------------------------------------------------------------------
    grown_ok = df[(df[area_col] > 0) & (df[prod_col] > 0)]
    grown_set = set(zip(
        grown_ok[state_col].astype(str).str.upper().str.strip(),
        grown_ok[crop_col].astype(str).str.strip(),
    ))

    CROP_TO_APY_NAME = {**YIELD_NAME_MAP}  # reuse the same mapping as a best-effort check
    verifiable = {}
    for label in ALL_LABELS:
        apy_name = CROP_TO_APY_NAME.get(label, label.capitalize())
        for state in STATES:
            verifiable[(label, state)] = (state.upper().strip(), apy_name) in grown_set
    return verifiable


def main():
    yield_records = build_state_yield()
    cost_records = build_state_cost()
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

            # --- YIELD: real state data > fruit horticulture override > placeholder ---
            if y:
                yield_val, yield_src = y["avg_yield_t_ha"], y["yield_source"]
            elif label in FRUIT_YIELD_OVERRIDE_T_HA:
                yield_val, yield_src = FRUIT_YIELD_OVERRIDE_T_HA[label], FRUIT_YIELD_SOURCE
            else:
                yield_val, yield_src = ph_yield, "national placeholder estimate (no dataset covers this crop)"

            # --- PRICE: coconut override > real national price > placeholder ---
            if label == "coconut":
                price_val, price_src = COCONUT_DRY_PRICE_INR_KG, "documented dry-coconut market rate (~Rs 28/kg, not Tender Coconut)"
            elif p:
                price_val, price_src = p["price_per_kg_inr"], p["price_source"]
            else:
                price_val, price_src = ph_price, "national placeholder estimate (no dataset covers this crop)"

            # --- COST: real CACP state data > horticulture cost table > placeholder ---
            if c:
                cost_val, cost_src = c["cost_cultivation_inr_ha"], c["cost_source"]
            elif label in HORTICULTURE_COST_INR_HA:
                cost_val, cost_src = HORTICULTURE_COST_INR_HA[label], "national estimate (horticulture cost table, no CACP data exists for this crop)"
            else:
                cost_val, cost_src = ph_cost, "national placeholder estimate (no dataset covers this crop)"

            row = {
                "label": label,
                "state": state,
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
            }
            rows.append(row)

    out_df = pd.DataFrame(rows)

    out_df.to_csv("crop_benchmarks_STATE.csv", index=False)
    print("Wrote crop_benchmarks_STATE.csv:", out_df.shape)
    print(out_df["data_confidence"].value_counts())
    print(out_df["regionally_verifiable"].value_counts())


if __name__ == "__main__":
    main()
