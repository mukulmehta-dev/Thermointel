import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import json
import math
import gc
import asyncio
import concurrent.futures
import joblib
import pandas as pd
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from . import db
from . import ml_features


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# ENVIRONMENT
# =========================================================

backend_env = os.path.join(os.path.dirname(BASE_DIR), ".env")
if os.path.exists(backend_env):
    load_dotenv(backend_env)
load_dotenv()

FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY")

if not FIRMS_MAP_KEY:
    raise RuntimeError(
        "FIRMS_MAP_KEY is missing. Add it to backend/.env"
    )


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="ThermoIntel API",
    version="0.7.0",
    description=(
        "AI-assisted industrial thermal intelligence "
        "backend for SIH 26162."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()


# =========================================================
# ML MODEL LOADING
# =========================================================

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models",
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "thermal_classifier.joblib",
)

BOOSTER_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "thermal_classifier.json",
)

LABEL_ENCODER_PATH = os.path.join(
    MODELS_DIR,
    "label_encoder.joblib",
)

CLASSES_PATH = os.path.join(
    MODELS_DIR,
    "classes.json",
)

ML_MODEL = None
ML_CLASSES = None


def load_ml_model():
    global ML_MODEL, ML_CLASSES

    # Prefer native XGBoost booster JSON (runs with zero scikit-learn dependency)
    if os.path.exists(BOOSTER_MODEL_PATH):
        try:
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(BOOSTER_MODEL_PATH)
            ML_MODEL = booster

            if os.path.exists(CLASSES_PATH):
                with open(CLASSES_PATH, "r", encoding="utf-8") as file:
                    ML_CLASSES = json.load(file)
            else:
                ML_CLASSES = [
                    "Industrial Fire Candidate",
                    "Potential Industrial Thermal Source",
                    "Thermal Anomaly",
                ]

            print(
                "Trained ML booster model "
                "loaded successfully (native JSON)."
            )
            return

        except Exception as error:
            print(
                "Failed to load native booster model: "
                f"{error}. Falling back to joblib model."
            )

    if os.path.exists(MODEL_PATH):
        try:
            ML_MODEL = joblib.load(
                MODEL_PATH
            )

            if os.path.exists(CLASSES_PATH):
                with open(CLASSES_PATH, "r", encoding="utf-8") as file:
                    ML_CLASSES = json.load(file)
            elif os.path.exists(LABEL_ENCODER_PATH):
                label_encoder = joblib.load(
                    LABEL_ENCODER_PATH
                )
                ML_CLASSES = list(label_encoder.classes_)
            else:
                ML_CLASSES = [
                    "Industrial Fire Candidate",
                    "Potential Industrial Thermal Source",
                    "Thermal Anomaly",
                ]

            print(
                "Trained ML classifier "
                "loaded successfully (joblib)."
            )

        except Exception as error:
            print(
                "Failed to load ML model: "
                f"{error}"
            )

            ML_MODEL = None
            ML_CLASSES = None

    else:
        print(
            "No trained ML model found. "
            "Run train_model.py to create one. "
            "Falling back to rule-based "
            "classification only."
        )


load_ml_model()


def predict_ml_classification(event):
    if ML_MODEL is None or not ML_CLASSES:
        return None, None

    try:
        row = pd.DataFrame(
            [
                {
                    "brightness": event["brightness"],
                    "confidence": event["confidence"],
                    "frp": event["frp"],
                    "intelligence_score": event[
                        "intelligence_score"
                    ],
                    "industrial_likelihood": event[
                        "industrial_likelihood"
                    ],
                    "landcover": event.get(
                        "landcover",
                        "unknown",
                    ),
                }
            ]
        )

        features = ml_features.build_features(
            row
        )

        if hasattr(ML_MODEL, "predict_proba"):
            prediction_index = int(
                ML_MODEL.predict(
                    features
                )[0]
            )
            probabilities = ML_MODEL.predict_proba(
                features
            )[0]
        else:
            import xgboost as xgb
            dmat = xgb.DMatrix(features)
            probabilities = ML_MODEL.predict(dmat)[0]
            prediction_index = int(probabilities.argmax())

        if 0 <= prediction_index < len(ML_CLASSES):
            predicted_label = ML_CLASSES[prediction_index]
        else:
            predicted_label = "Thermal Anomaly"

        confidence = round(
            float(
                max(probabilities)
            )
            * 100,
            1,
        )

        return predicted_label, confidence

    except Exception:
        return None, None


# =========================================================
# BACKGROUND POLLING
# =========================================================

async def background_poll_loop():
    while True:
        try:
            await asyncio.to_thread(
                build_india_events
            )

        except Exception:
            pass

        await asyncio.sleep(
            BACKGROUND_POLL_INTERVAL_SECONDS
        )


@app.on_event("startup")
async def start_background_poller():
    asyncio.create_task(
        background_poll_loop()
    )


# =========================================================
# CONFIGURATION
# =========================================================

FIRMS_SOURCE = "VIIRS_NOAA20_NRT"

INDIA_BBOX = "68,6,98,37"

FIRMS_DAYS = 3

CLUSTER_RADIUS_KM = 5.0

LANDCOVER_RADIUS_M = 400

LANDCOVER_GRID_DECIMALS = 2

MAX_LANDCOVER_LOOKUPS_PER_REQUEST = 8

LANDCOVER_LOOKUP_TIME_BUDGET_SECONDS = 15

LANDCOVER_LOOKUP_WORKERS = 2

BACKGROUND_POLL_INTERVAL_SECONDS = 30 * 60

MIN_DAYS_SEEN_FOR_BASELINE = 2

SPIKE_MULTIPLIER_THRESHOLD = 1.5

FIRMS_URL = (
    "https://firms.modaps.eosdis.nasa.gov/"
    "api/area/csv/"
    f"{FIRMS_MAP_KEY}/"
    f"{FIRMS_SOURCE}/"
    f"{INDIA_BBOX}/"
    f"{FIRMS_DAYS}"
)

# Multiple Overpass servers.
# If one is unavailable or times out, the next one is tried.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

OVERPASS_HEADERS = {
    "User-Agent": "ThermoIntel/0.7"
}

# In-memory cache for landcover lookups.
# Keyed by rounded (lat, lon) grid cells to avoid
# repeated Overpass calls for nearby events.
LANDCOVER_CACHE = {}


# =========================================================
# INDIA GEOJSON
# =========================================================

INDIA_GEOJSON_PATH = os.path.join(
    BASE_DIR,
    "india.geojson",
)


def load_india_geometry():
    if not os.path.exists(INDIA_GEOJSON_PATH):
        raise RuntimeError(
            "india.geojson was not found at: "
            f"{INDIA_GEOJSON_PATH}"
        )

    try:
        with open(
            INDIA_GEOJSON_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            geojson = json.load(file)

        geojson_type = geojson.get("type")

        if geojson_type == "FeatureCollection":
            geometries = []

            for feature in geojson.get("features", []):
                geometry = feature.get("geometry")

                if geometry:
                    geometries.append(
                        shape(geometry)
                    )

            del geojson
            gc.collect()

            if not geometries:
                raise ValueError(
                    "FeatureCollection contains no geometries."
                )

            if len(geometries) == 1:
                return geometries[0]

            return unary_union(geometries)

        if geojson_type == "Feature":
            geometry = geojson.get("geometry")

            del geojson
            gc.collect()

            if not geometry:
                raise ValueError(
                    "GeoJSON Feature has no geometry."
                )

            return shape(geometry)

        if geojson_type in {
            "Polygon",
            "MultiPolygon",
        }:
            geom = shape(geojson)
            del geojson
            gc.collect()
            return geom

        raise ValueError(
            f"Unsupported GeoJSON type: {geojson_type}"
        )

    except Exception as error:
        raise RuntimeError(
            "Unable to load india.geojson: "
            f"{error}"
        )


INDIA_GEOMETRY = load_india_geometry()
PREPARED_INDIA_GEOMETRY = prep(INDIA_GEOMETRY)


def is_inside_india(latitude, longitude):
    point = Point(
        float(longitude),
        float(latitude),
    )

    return (
        PREPARED_INDIA_GEOMETRY.contains(point)
        or PREPARED_INDIA_GEOMETRY.touches(point)
    )


# =========================================================
# LOCAL LAND USE LAYERS (offline, no Overpass needed)
# =========================================================

LANDCOVER_LAYER_FILES = {
    "industrial": "industrial_landuse.geojson",
    "forest": "forest_landuse.geojson",
    "farmland": "farmland_landuse.geojson",
}

LOCAL_LANDCOVER_GRID_SIZE = 0.1

LOCAL_LANDCOVER_PROXIMITY_DEGREES = 0.005

LOCAL_LANDCOVER_LAYERS = {}


def load_local_landcover_layer(
    category,
    filename,
):
    geojson_path = os.path.join(
        BASE_DIR,
        filename,
    )

    if not os.path.exists(geojson_path):
        print(
            f"No {filename} found. "
            f"'{category}' land-cover checks "
            "will rely on live Overpass "
            "calls only."
        )

        return

    try:
        with open(
            geojson_path,
            "r",
            encoding="utf-8",
        ) as file:
            geojson = json.load(file)

        geometries = []

        for feature in geojson.get(
            "features",
            [],
        ):
            geometry = feature.get(
                "geometry"
            )

            if not geometry:
                continue

            try:
                geometries.append(
                    shape(geometry)
                )

            except Exception:
                continue

        del geojson
        gc.collect()

        if not geometries:
            print(
                f"{filename} loaded but "
                "contained no usable "
                "geometries."
            )

            return

        grid_index = {}

        for polygon_index, geometry in enumerate(
            geometries
        ):
            try:
                (
                    min_lon,
                    min_lat,
                    max_lon,
                    max_lat,
                ) = geometry.bounds

            except Exception:
                continue

            min_cell_lat = math.floor(
                min_lat
                / LOCAL_LANDCOVER_GRID_SIZE
            )

            max_cell_lat = math.floor(
                max_lat
                / LOCAL_LANDCOVER_GRID_SIZE
            )

            min_cell_lon = math.floor(
                min_lon
                / LOCAL_LANDCOVER_GRID_SIZE
            )

            max_cell_lon = math.floor(
                max_lon
                / LOCAL_LANDCOVER_GRID_SIZE
            )

            for cell_lat in range(
                min_cell_lat,
                max_cell_lat + 1,
            ):
                for cell_lon in range(
                    min_cell_lon,
                    max_cell_lon + 1,
                ):
                    key = (
                        cell_lat,
                        cell_lon,
                    )

                    grid_index.setdefault(
                        key,
                        [],
                    ).append(polygon_index)

        LOCAL_LANDCOVER_LAYERS[category] = {
            "geometries": geometries,
            "grid_index": grid_index,
        }

        print(
            f"Loaded {len(geometries)} "
            f"'{category}' land-use polygons "
            "from local file "
            f"({len(grid_index)} grid cells "
            "indexed)."
        )

    except Exception as error:
        print(
            f"Failed to load {filename}: "
            f"{error}"
        )


def load_all_local_landcover_layers():
    for category, filename in (
        LANDCOVER_LAYER_FILES.items()
    ):
        load_local_landcover_layer(
            category,
            filename,
        )


load_all_local_landcover_layers()


def is_point_in_local_layer(
    category,
    latitude,
    longitude,
):
    layer = LOCAL_LANDCOVER_LAYERS.get(
        category
    )

    if not layer:
        return False

    geometries = layer["geometries"]

    grid_index = layer["grid_index"]

    cell_lat = math.floor(
        latitude
        / LOCAL_LANDCOVER_GRID_SIZE
    )

    cell_lon = math.floor(
        longitude
        / LOCAL_LANDCOVER_GRID_SIZE
    )

    candidate_indices = set()

    for lat_offset in (-1, 0, 1):
        for lon_offset in (-1, 0, 1):
            key = (
                cell_lat + lat_offset,
                cell_lon + lon_offset,
            )

            candidate_indices.update(
                grid_index.get(
                    key,
                    [],
                )
            )

    if not candidate_indices:
        return False

    point = Point(
        longitude,
        latitude,
    )

    for polygon_index in candidate_indices:
        try:
            geometry = geometries[
                polygon_index
            ]

            if (
                geometry.distance(point)
                <= LOCAL_LANDCOVER_PROXIMITY_DEGREES
            ):
                return True

        except Exception:
            continue

    return False


def get_local_landcover(
    latitude,
    longitude,
):
    for category in (
        "industrial",
        "forest",
        "farmland",
    ):
        if is_point_in_local_layer(
            category,
            latitude,
            longitude,
        ):
            return category

    return None


# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "ThermoIntel API",
        "version": "0.7.0",
        "india_boundary": True,
        "intelligence_layer": True,
        "clustering_layer": True,
        "industrial_context_layer": True,
        "landcover_classification_layer": True,
    }


# =========================================================
# FIRMS DATA FETCH
# =========================================================

def fetch_firms_data():
    try:
        response = requests.get(
            FIRMS_URL,
            timeout=30,
        )

        response.raise_for_status()

    except requests.RequestException as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to retrieve NASA FIRMS "
                f"data: {error}"
            ),
        )

    if not response.text.strip():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            io.StringIO(response.text)
        )

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=(
                "NASA FIRMS returned data that "
                f"could not be parsed: {error}"
            ),
        )


# =========================================================
# CONFIDENCE NORMALIZATION
# =========================================================

def normalize_confidence(value):
    if pd.isna(value):
        return 50

    if isinstance(value, str):
        value = value.lower().strip()

        if value == "h":
            return 90

        if value == "n":
            return 70

        if value == "l":
            return 45

    try:
        numeric = float(value)

        return max(
            0,
            min(
                100,
                numeric,
            ),
        )

    except Exception:
        return 50


# =========================================================
# INTELLIGENCE SCORE
# =========================================================

def calculate_intelligence_score(
    brightness,
    frp,
    confidence,
):
    brightness_score = (
        (brightness - 290) / 70
    ) * 100

    brightness_score = max(
        0,
        min(
            100,
            brightness_score,
        ),
    )

    frp_score = (
        frp / 20
    ) * 100

    frp_score = max(
        0,
        min(
            100,
            frp_score,
        ),
    )

    intelligence_score = (
        brightness_score * 0.40
        + frp_score * 0.40
        + confidence * 0.20
    )

    return round(
        max(
            0,
            min(
                100,
                intelligence_score,
            ),
        ),
        1,
    )


# =========================================================
# INDUSTRIAL LIKELIHOOD
# =========================================================

def calculate_industrial_likelihood(
    brightness,
    frp,
    confidence,
):
    score = 0

    if brightness >= 340:
        score += 40

    elif brightness >= 325:
        score += 30

    elif brightness >= 310:
        score += 20

    else:
        score += 10

    if frp >= 20:
        score += 40

    elif frp >= 10:
        score += 30

    elif frp >= 5:
        score += 20

    elif frp >= 2:
        score += 10

    if confidence >= 80:
        score += 20

    elif confidence >= 70:
        score += 15

    elif confidence >= 50:
        score += 10

    return min(
        100,
        score,
    )


# =========================================================
# THERMAL CLASSIFICATION (INTENSITY-BASED)
# =========================================================

def classify_event(row):
    brightness = float(
        row.get(
            "bright_ti4",
            0,
        )
        or 0
    )

    frp = float(
        row.get(
            "frp",
            0,
        )
        or 0
    )

    confidence = normalize_confidence(
        row.get("confidence")
    )

    if (
        brightness >= 340
        and frp >= 20
        and confidence >= 70
    ):
        return (
            "Industrial Fire Candidate",
            "HIGH",
        )

    if (
        brightness >= 325
        or frp >= 10
    ):
        return (
            "Potential Industrial Thermal Source",
            "MEDIUM",
        )

    return (
        "Thermal Anomaly",
        "LOW",
    )


# =========================================================
# LANDCOVER LOOKUP (OVERPASS)
# =========================================================

def landcover_grid_key(latitude, longitude):
    return (
        round(latitude, LANDCOVER_GRID_DECIMALS),
        round(longitude, LANDCOVER_GRID_DECIMALS),
    )


def query_landcover(latitude, longitude):
    local_result = get_local_landcover(
        latitude,
        longitude,
    )

    if local_result:
        return local_result

    # If forest/farmland local files aren't
    # present yet, fall back to a throttled
    # live Overpass check as a stopgap.
    has_forest_file = (
        "forest" in LOCAL_LANDCOVER_LAYERS
    )

    has_farmland_file = (
        "farmland" in LOCAL_LANDCOVER_LAYERS
    )

    if has_forest_file and has_farmland_file:
        return None

    query = f"""
    [out:json][timeout:10];
    (
      nwr["landuse"="forest"]
        (around:{LANDCOVER_RADIUS_M},{latitude},{longitude});

      nwr["natural"="wood"]
        (around:{LANDCOVER_RADIUS_M},{latitude},{longitude});

      nwr["landuse"~"farmland|meadow|orchard"]
        (around:{LANDCOVER_RADIUS_M},{latitude},{longitude});
    );
    out tags 15;
    """

    for overpass_url in OVERPASS_URLS[:1]:
        try:
            response = requests.post(
                overpass_url,
                data=query,
                timeout=8,
                headers=OVERPASS_HEADERS,
            )

            response.raise_for_status()

            payload = response.json()

            found_forest = False
            found_farmland = False

            for element in payload.get(
                "elements",
                [],
            ):
                tags = element.get(
                    "tags"
                ) or {}

                if (
                    tags.get("landuse") == "forest"
                    or tags.get("natural") == "wood"
                ):
                    found_forest = True

                if tags.get("landuse") in (
                    "farmland",
                    "meadow",
                    "orchard",
                ):
                    found_farmland = True

            if found_forest and not has_forest_file:
                return "forest"

            if found_farmland and not has_farmland_file:
                return "farmland"

            return None

        except requests.RequestException:
            continue

        except ValueError:
            continue

    return None


def get_landcover(latitude, longitude):
    key = landcover_grid_key(
        latitude,
        longitude,
    )

    if key in LANDCOVER_CACHE:
        return LANDCOVER_CACHE[key]

    result = query_landcover(
        latitude,
        longitude,
    )

    LANDCOVER_CACHE[key] = result

    return result


def build_landcover_lookup(coordinate_pairs):
    unique_keys = {}

    for latitude, longitude in coordinate_pairs:
        key = landcover_grid_key(
            latitude,
            longitude,
        )

        if key not in unique_keys:
            unique_keys[key] = (
                latitude,
                longitude,
            )

    lookup = {}

    remaining = {}

    # Phase 1: local land-use file check
    # (industrial, forest, farmland). This is
    # instant and offline, so every location
    # gets checked, no throttling.
    for key, (latitude, longitude) in unique_keys.items():
        if key in LANDCOVER_CACHE:
            lookup[key] = LANDCOVER_CACHE[key]
            continue

        local_result = get_local_landcover(
            latitude,
            longitude,
        )

        if local_result:
            lookup[key] = local_result

            LANDCOVER_CACHE[key] = local_result

        else:
            remaining[key] = (
                latitude,
                longitude,
            )

    # Phase 2: Overpass fallback, only used
    # for whichever local files aren't present
    # yet. This part stays throttled since it's
    # a live call to a rate-limited public
    # service.
    pending_items = list(
        remaining.items()
    )[:MAX_LANDCOVER_LOOKUPS_PER_REQUEST]

    skipped_items = list(
        remaining.items()
    )[MAX_LANDCOVER_LOOKUPS_PER_REQUEST:]

    for key, _ in skipped_items:
        lookup[key] = None

    if not pending_items:
        return lookup

    deadline_seconds = (
        LANDCOVER_LOOKUP_TIME_BUDGET_SECONDS
    )

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=LANDCOVER_LOOKUP_WORKERS
    )

    future_to_key = {
        executor.submit(
            get_landcover,
            latitude,
            longitude,
        ): key
        for key, (latitude, longitude) in pending_items
    }

    done_futures, pending_futures = (
        concurrent.futures.wait(
            future_to_key.keys(),
            timeout=deadline_seconds,
        )
    )

    for future in done_futures:
        key = future_to_key[future]

        try:
            lookup[key] = future.result()

        except Exception:
            lookup[key] = None

    for future in pending_futures:
        key = future_to_key[future]

        lookup[key] = None

    executor.shutdown(
        wait=False,
        cancel_futures=True,
    )

    return lookup


# =========================================================
# LANDCOVER-AWARE CLASSIFICATION
# =========================================================

def apply_landcover_context(
    classification,
    industrial_likelihood,
    landcover,
):
    if landcover == "industrial":
        if classification == "Thermal Anomaly":
            return "Potential Industrial Thermal Source"

        return classification

    if (
        landcover == "forest"
        and industrial_likelihood < 55
    ):
        return "Likely Forest Fire (Non-Industrial)"

    if (
        landcover == "farmland"
        and industrial_likelihood < 55
    ):
        return "Likely Agricultural Burning (Non-Industrial)"

    return classification


# =========================================================
# BASELINE / SPIKE DETECTION
# =========================================================

def compute_baseline_status(
    intelligence_score,
    baseline_entry,
):
    if (
        baseline_entry is None
        or baseline_entry["days_seen"]
        < MIN_DAYS_SEEN_FOR_BASELINE
    ):
        return "NEW_LOCATION", None

    baseline_score = (
        baseline_entry["baseline_score"] or 0
    )

    if baseline_score <= 0:
        return "NEW_LOCATION", None

    ratio = round(
        intelligence_score / baseline_score,
        2,
    )

    if ratio >= SPIKE_MULTIPLIER_THRESHOLD:
        return "ANOMALOUS_SPIKE", ratio

    return "CONSISTENT_WITH_HISTORY", ratio


# =========================================================
# INTELLIGENCE EXPLANATION
# =========================================================

def generate_intelligence_reason(
    brightness,
    frp,
    confidence,
    classification,
    landcover,
):
    reasons = []

    if brightness >= 340:
        reasons.append(
            "very high thermal brightness"
        )

    elif brightness >= 325:
        reasons.append(
            "elevated thermal brightness"
        )

    if frp >= 20:
        reasons.append(
            "very high fire radiative power"
        )

    elif frp >= 10:
        reasons.append(
            "high fire radiative power"
        )

    elif frp >= 5:
        reasons.append(
            "moderate fire radiative power"
        )

    if confidence >= 80:
        reasons.append(
            "high detection confidence"
        )

    elif confidence >= 70:
        reasons.append(
            "reliable detection confidence"
        )

    if landcover == "industrial":
        reasons.append(
            "located within mapped industrial land use"
        )

    elif landcover == "forest":
        reasons.append(
            "located within forested land cover"
        )

    elif landcover == "farmland":
        reasons.append(
            "located within agricultural land cover"
        )

    if not reasons:
        return (
            "Low-intensity thermal detection "
            "with limited evidence of an "
            "industrial source."
        )

    reason_text = ", ".join(reasons)

    return (
        f"{classification} based on "
        f"{reason_text}."
    )


# =========================================================
# TRANSFORM FIRMS ROW
# =========================================================

def transform_event(row, index, landcover, baseline_entry):
    classification, severity = classify_event(row)

    latitude = float(
        row["latitude"]
    )

    longitude = float(
        row["longitude"]
    )

    brightness = float(
        row.get(
            "bright_ti4",
            0,
        )
        or 0
    )

    frp = float(
        row.get(
            "frp",
            0,
        )
        or 0
    )

    confidence = normalize_confidence(
        row.get("confidence")
    )

    intelligence_score = (
        calculate_intelligence_score(
            brightness,
            frp,
            confidence,
        )
    )

    industrial_likelihood = (
        calculate_industrial_likelihood(
            brightness,
            frp,
            confidence,
        )
    )

    classification = apply_landcover_context(
        classification,
        industrial_likelihood,
        landcover,
    )

    baseline_status, baseline_ratio = (
        compute_baseline_status(
            intelligence_score,
            baseline_entry,
        )
    )

    intelligence_reason = (
        generate_intelligence_reason(
            brightness,
            frp,
            confidence,
            classification,
            landcover,
        )
    )

    acquisition_date = str(
        row.get(
            "acq_date",
            "",
        )
    )

    acquisition_time = str(
        row.get(
            "acq_time",
            "",
        )
    )

    satellite = str(
        row.get(
            "satellite",
            "VIIRS",
        )
    )

    event = {
        "id": f"firms-{index}",
        "latitude": latitude,
        "longitude": longitude,
        "brightness": round(
            brightness,
            2,
        ),
        "confidence": round(
            confidence,
            1,
        ),
        "frp": round(
            frp,
            2,
        ),
        "classification": classification,
        "severity": severity,
        "landcover": landcover or "unknown",
        "intelligence_score": intelligence_score,
        "industrial_likelihood": industrial_likelihood,
        "intelligence_reason": intelligence_reason,
        "location": (
            f"{latitude:.3f}, "
            f"{longitude:.3f}"
        ),
        "source": satellite,
        "timestamp": (
            f"{acquisition_date} "
            f"{acquisition_time}"
        ),
        "cluster_id": None,
        "cluster_size": 1,
        "cluster_peak_frp": round(
            frp,
            2,
        ),
        "cluster_avg_score": intelligence_score,
        "cluster_max_score": intelligence_score,
        "cluster_severity": severity,
        "baseline_status": baseline_status,
        "baseline_ratio": baseline_ratio,
        "ml_classification": None,
        "ml_confidence": None,
    }

    ml_classification, ml_confidence = (
        predict_ml_classification(event)
    )

    event["ml_classification"] = (
        ml_classification
    )

    event["ml_confidence"] = ml_confidence

    return event


# =========================================================
# HAVERSINE DISTANCE
# =========================================================

def haversine_distance_km(
    latitude1,
    longitude1,
    latitude2,
    longitude2,
):
    earth_radius_km = 6371.0

    lat1 = math.radians(latitude1)
    lat2 = math.radians(latitude2)

    delta_lat = math.radians(
        latitude2 - latitude1
    )

    delta_lon = math.radians(
        longitude2 - longitude1
    )

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )

    return earth_radius_km * c


# =========================================================
# SEVERITY RANK
# =========================================================

def severity_rank(severity):
    ranks = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
    }

    return ranks.get(
        severity,
        0,
    )


# =========================================================
# SPATIAL CLUSTERING
# =========================================================

def cluster_events(events):
    if not events:
        return events

    total_events = len(events)

    parent = list(
        range(total_events)
    )

    rank = [
        0
        for _ in range(total_events)
    ]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[
                parent[x]
            ]

            x = parent[x]

        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a == root_b:
            return

        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b

        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a

        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    for i in range(total_events):
        event_a = events[i]

        for j in range(
            i + 1,
            total_events,
        ):
            event_b = events[j]

            distance = haversine_distance_km(
                event_a["latitude"],
                event_a["longitude"],
                event_b["latitude"],
                event_b["longitude"],
            )

            if distance <= CLUSTER_RADIUS_KM:
                union(i, j)

    grouped = {}

    for index in range(total_events):
        root = find(index)

        if root not in grouped:
            grouped[root] = []

        grouped[root].append(
            events[index]
        )

    clusters = list(
        grouped.values()
    )

    clusters.sort(
        key=lambda cluster: (
            max(
                event["intelligence_score"]
                for event in cluster
            ),
            len(cluster),
            max(
                event["frp"]
                for event in cluster
            ),
        ),
        reverse=True,
    )

    for cluster_index, cluster in enumerate(
        clusters,
        start=1,
    ):
        cluster_id = (
            f"cluster-{cluster_index}"
        )

        cluster_size = len(cluster)

        peak_frp = max(
            event["frp"]
            for event in cluster
        )

        average_score = (
            sum(
                event["intelligence_score"]
                for event in cluster
            )
            / cluster_size
        )

        maximum_score = max(
            event["intelligence_score"]
            for event in cluster
        )

        highest_severity = max(
            (
                event["severity"]
                for event in cluster
            ),
            key=severity_rank,
        )

        cluster_bonus = min(
            15,
            (cluster_size - 1) * 2,
        )

        cluster_score = round(
            min(
                100,
                maximum_score + cluster_bonus,
            ),
            1,
        )

        for event in cluster:
            event["cluster_id"] = cluster_id

            event["cluster_size"] = cluster_size

            event["cluster_peak_frp"] = round(
                peak_frp,
                2,
            )

            event["cluster_avg_score"] = round(
                average_score,
                1,
            )

            event["cluster_max_score"] = (
                cluster_score
            )

            event["cluster_severity"] = (
                highest_severity
            )

    return events


# =========================================================
# BUILD EVENTS
# =========================================================

def build_india_events():
    dataframe = fetch_firms_data()

    if dataframe.empty:
        return {
            "raw_count": 0,
            "filtered_count": 0,
            "events": [],
        }

    required_columns = [
        "latitude",
        "longitude",
        "bright_ti4",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=502,
            detail=(
                "NASA FIRMS response is "
                "missing required fields: "
                f"{missing_columns}"
            ),
        )

    dataframe = dataframe.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    )

    raw_count = len(dataframe)

    india_rows = []

    for index, row in dataframe.iterrows():
        try:
            latitude = float(
                row["latitude"]
            )

            longitude = float(
                row["longitude"]
            )

            if is_inside_india(
                latitude,
                longitude,
            ):
                india_rows.append(
                    (
                        index,
                        row,
                    )
                )

        except Exception:
            continue

    filtered_count = len(
        india_rows
    )

    landcover_lookup = build_landcover_lookup(
        (
            float(row["latitude"]),
            float(row["longitude"]),
        )
        for _, row in india_rows
    )

    try:
        baseline_lookup = (
            db.get_location_baselines()
        )
    except Exception:
        baseline_lookup = {}

    events = []

    for index, row in india_rows:
        try:
            key = landcover_grid_key(
                float(row["latitude"]),
                float(row["longitude"]),
            )

            landcover = landcover_lookup.get(
                key
            )

            baseline_entry = baseline_lookup.get(
                key
            )

            event = transform_event(
                row,
                index,
                landcover,
                baseline_entry,
            )

            events.append(event)

        except Exception:
            continue

    events = cluster_events(events)

    try:
        db.insert_events(events)
    except Exception:
        pass

    return {
        "raw_count": raw_count,
        "filtered_count": filtered_count,
        "events": events,
    }


# =========================================================
# FIRMS EVENTS ENDPOINT
# =========================================================

@app.get("/api/firms/events")
def get_firms_events(
    limit: int = 1000,
):
    result = build_india_events()

    max_limit = max(
        1,
        min(
            limit,
            5000,
        ),
    )

    events = result["events"][:max_limit]

    return {
        "source": FIRMS_SOURCE,
        "days": FIRMS_DAYS,
        "raw_count": result["raw_count"],
        "filtered_count": result["filtered_count"],
        "count": len(events),
        "events": events,
        "retrieved_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }


# =========================================================
# CLUSTERS ENDPOINT
# =========================================================

@app.get("/api/firms/clusters")
def get_firms_clusters():
    result = build_india_events()

    events = result["events"]

    clusters_map = {}

    for event in events:
        cluster_id = event.get(
            "cluster_id"
        )

        if not cluster_id:
            continue

        if cluster_id not in clusters_map:
            clusters_map[cluster_id] = []

        clusters_map[
            cluster_id
        ].append(event)

    clusters = []

    for (
        cluster_id,
        cluster_events_list,
    ) in clusters_map.items():

        representative = max(
            cluster_events_list,
            key=lambda event: (
                event["intelligence_score"],
                event["frp"],
            ),
        )

        latitude = (
            sum(
                event["latitude"]
                for event in cluster_events_list
            )
            / len(cluster_events_list)
        )

        longitude = (
            sum(
                event["longitude"]
                for event in cluster_events_list
            )
            / len(cluster_events_list)
        )

        clusters.append(
            {
                "cluster_id": cluster_id,
                "latitude": round(
                    latitude,
                    5,
                ),
                "longitude": round(
                    longitude,
                    5,
                ),
                "event_count": len(
                    cluster_events_list
                ),
                "severity": (
                    representative[
                        "cluster_severity"
                    ]
                ),
                "intelligence_score": (
                    representative[
                        "cluster_max_score"
                    ]
                ),
                "average_score": (
                    representative[
                        "cluster_avg_score"
                    ]
                ),
                "peak_frp": (
                    representative[
                        "cluster_peak_frp"
                    ]
                ),
                "representative_event": (
                    representative["id"]
                ),
            }
        )

    clusters.sort(
        key=lambda cluster: (
            cluster["intelligence_score"],
            cluster["event_count"],
        ),
        reverse=True,
    )

    return {
        "source": FIRMS_SOURCE,
        "days": FIRMS_DAYS,
        "event_count": len(events),
        "cluster_count": len(clusters),
        "clusters": clusters,
        "retrieved_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }


# =========================================================
# INDUSTRIAL CONTEXT
# =========================================================

def find_nearby_industrial_context(
    latitude,
    longitude,
    radius_km=10.0,
):
    radius_m = max(
        500,
        min(
            float(radius_km) * 1000,
            25000,
        ),
    )

    query = f"""
    [out:json][timeout:15];
    (
      nwr["landuse"="industrial"]
        (around:{radius_m},{latitude},{longitude});

      nwr["industrial"]
        (around:{radius_m},{latitude},{longitude});

      nwr["man_made"~"works|kiln|storage_tank|silo"]
        (around:{radius_m},{latitude},{longitude});
    );
    out center tags;
    """

    errors = []

    for overpass_url in OVERPASS_URLS:
        try:
            response = requests.post(
                overpass_url,
                data=query,
                timeout=18,
                headers=OVERPASS_HEADERS,
            )

            response.raise_for_status()

            payload = response.json()

            candidates = []

            for element in payload.get(
                "elements",
                [],
            ):
                tags = element.get(
                    "tags"
                ) or {}

                center = element.get(
                    "center"
                ) or {}

                feature_lat = element.get(
                    "lat",
                    center.get("lat"),
                )

                feature_lon = element.get(
                    "lon",
                    center.get("lon"),
                )

                if (
                    feature_lat is None
                    or feature_lon is None
                ):
                    continue

                feature_lat = float(
                    feature_lat
                )

                feature_lon = float(
                    feature_lon
                )

                distance = (
                    haversine_distance_km(
                        float(latitude),
                        float(longitude),
                        feature_lat,
                        feature_lon,
                    )
                )

                name = (
                    tags.get("name")
                    or tags.get("official_name")
                    or tags.get("operator")
                    or "Unnamed industrial feature"
                )

                feature_type = (
                    tags.get("industrial")
                    or tags.get("landuse")
                    or tags.get("man_made")
                    or "industrial"
                )

                candidates.append(
                    {
                        "name": name,
                        "type": feature_type,
                        "distance_km": round(
                            distance,
                            2,
                        ),
                        "latitude": round(
                            feature_lat,
                            5,
                        ),
                        "longitude": round(
                            feature_lon,
                            5,
                        ),
                        "osm_type": element.get(
                            "type"
                        ),
                        "osm_id": element.get(
                            "id"
                        ),
                    }
                )

            candidates.sort(
                key=lambda item:
                item["distance_km"]
            )

            unique = []
            seen = set()

            for candidate in candidates:
                key = (
                    candidate["name"],
                    candidate["type"],
                    round(
                        candidate["latitude"],
                        4,
                    ),
                    round(
                        candidate["longitude"],
                        4,
                    ),
                )

                if key in seen:
                    continue

                seen.add(key)

                unique.append(candidate)

                if len(unique) >= 10:
                    break

            return {
                "query": {
                    "latitude": float(
                        latitude
                    ),
                    "longitude": float(
                        longitude
                    ),
                    "radius_km": round(
                        radius_m / 1000,
                        2,
                    ),
                },
                "nearby_count": len(
                    unique
                ),
                "nearest": (
                    unique[0]
                    if unique
                    else None
                ),
                "features": unique,
                "source": (
                    "OpenStreetMap Overpass"
                ),
                "status": "success",
                "retrieved_at": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            }

        except requests.RequestException as error:
            errors.append(
                f"{overpass_url}: {error}"
            )

        except ValueError as error:
            errors.append(
                f"{overpass_url}: "
                f"Invalid JSON: {error}"
            )

    # Do NOT crash the API if every Overpass
    # server is temporarily unavailable.
    return {
        "query": {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "radius_km": round(
                radius_m / 1000,
                2,
            ),
        },
        "nearby_count": 0,
        "nearest": None,
        "features": [],
        "source": "OpenStreetMap Overpass",
        "status": "unavailable",
        "message": (
            "Industrial context is temporarily "
            "unavailable."
        ),
        "errors": errors,
        "retrieved_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }


# =========================================================
# INDUSTRIAL CONTEXT ENDPOINT
# =========================================================

@app.get("/api/firms/context")
def get_firms_context(
    latitude: float,
    longitude: float,
    radius_km: float = 10.0,
):
    if not (
        -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid latitude or longitude."
            ),
        )

    if radius_km <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "radius_km must be greater "
                "than zero."
            ),
        )

    return find_nearby_industrial_context(
        latitude,
        longitude,
        radius_km,
    )


# =========================================================
# PERSISTENT THERMAL SOURCES
# =========================================================

@app.get("/api/firms/persistent")
def get_persistent_sources(
    min_days_active: int = 2,
    lookback_days: int = 30,
    limit: int = 50,
):
    sources = db.get_persistent_sources(
        min_days_active=min_days_active,
        lookback_days=lookback_days,
        limit=limit,
    )

    return {
        "min_days_active": min_days_active,
        "lookback_days": lookback_days,
        "count": len(sources),
        "sources": sources,
        "retrieved_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }


# =========================================================
# TRENDS
# =========================================================

@app.get("/api/firms/trends")
def get_trends(
    lookback_days: int = 14,
):
    trends = db.get_daily_trends(
        lookback_days=lookback_days
    )

    return {
        "lookback_days": lookback_days,
        "days_with_data": len(trends),
        "total_stored_events": (
            db.get_total_stored_events()
        ),
        "trends": trends,
        "retrieved_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "name": "ThermoIntel API",
        "status": "running",
        "version": "0.7.0",
        "india_boundary": True,
        "intelligence_layer": True,
        "clustering_layer": True,
        "industrial_context_layer": True,
        "landcover_classification_layer": True,
        "firms_days": FIRMS_DAYS,
        "cluster_radius_km": CLUSTER_RADIUS_KM,
        "endpoints": [
            "/api/health",
            "/api/firms/events",
            "/api/firms/clusters",
            "/api/firms/context",
            "/api/firms/persistent",
            "/api/firms/trends",
        ],
    }