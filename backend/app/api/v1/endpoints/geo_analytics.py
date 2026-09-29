"""
Geographic & Socio-Economic Stratification Analytics API.
Serves official State and District benchmarks (UDISE+, AISHE, NITI Aayog MPI).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Query, HTTPException

BASE_DIR = Path(__file__).resolve().parents[5]
GEO_INDEX_PATH = BASE_DIR / "data" / "macro" / "state_district_indices.json"

router = APIRouter(prefix="/analytics/geo", tags=["geo-analytics"])

def _load_geo_data() -> Dict[str, Any]:
    if not GEO_INDEX_PATH.exists():
        raise HTTPException(status_code=500, detail="Geographic index dataset not found.")
    with open(GEO_INDEX_PATH, "r") as f:
        return json.load(f)["states"]

@router.get("/states", summary="Get State-level education and poverty benchmarks")
def get_states_overview() -> List[Dict[str, Any]]:
    states_data = _load_geo_data()
    results = []
    for state_name, info in states_data.items():
        results.append({
            "state_name": state_name,
            "state_code": info["state_code"],
            "literacy_rate": info["literacy_rate"],
            "female_literacy_rate": info["female_literacy_rate"],
            "ger_higher_ed": info["ger_higher_ed"],
            "school_dropout_rate_secondary": info["school_dropout_rate_secondary"],
            "mpi_poverty_pct": info["mpi_poverty_pct"],
            "per_capita_income_inr": info["avg_annual_per_capita_income_inr"],
            "district_count": len(info["districts"])
        })
    return sorted(results, key=lambda x: x["mpi_poverty_pct"], reverse=True)

@router.get("/districts", summary="Get District-level indicators for a specific State")
def get_districts_by_state(state: str = Query(..., description="State name (e.g. Bihar, Uttar Pradesh)")) -> List[Dict[str, Any]]:
    states_data = _load_geo_data()
    if state not in states_data:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found in registry.")

    s_info = states_data[state]
    districts = []
    for d_name, d_info in s_info["districts"].items():
        districts.append({
            "district_name": d_name,
            "state_name": state,
            "state_code": s_info["state_code"],
            "mpi_headcount_pct": d_info["mpi_headcount_pct"],
            "is_aspirational": d_info["is_aspirational"],
            "rurality_pct": d_info["rurality_pct"],
            "pupil_teacher_ratio": d_info["ptr_ratio"]
        })
    return sorted(districts, key=lambda x: x["mpi_headcount_pct"], reverse=True)

@router.get("/heatmap", summary="Get Pan-India State Dropout Vulnerability Heatmap")
def get_national_heatmap() -> Dict[str, Any]:
    states_data = _load_geo_data()
    heatmap_items = []
    for state_name, info in states_data.items():
        # Composite State Vulnerability Score in [0, 100]
        vulnerability_score = (
            (100.0 - info["literacy_rate"]) * 0.35 +
            info["mpi_poverty_pct"] * 0.35 +
            info["school_dropout_rate_secondary"] * 0.30
        )
        heatmap_items.append({
            "state_name": state_name,
            "state_code": info["state_code"],
            "vulnerability_score": round(vulnerability_score, 1),
            "risk_level": "High" if vulnerability_score > 35 else ("Medium" if vulnerability_score > 20 else "Low"),
            "literacy_rate": info["literacy_rate"],
            "mpi_poverty_pct": info["mpi_poverty_pct"],
            "ger_higher_ed": info["ger_higher_ed"]
        })
    return {
        "total_states_monitored": len(heatmap_items),
        "states": sorted(heatmap_items, key=lambda x: x["vulnerability_score"], reverse=True)
    }
