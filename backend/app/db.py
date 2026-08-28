import os
import sqlite3
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DB_PATH = os.path.join(
    BASE_DIR,
    "thermointel.db",
)


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_connection()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS thermal_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_key TEXT UNIQUE,
            latitude REAL,
            longitude REAL,
            grid_lat REAL,
            grid_lon REAL,
            brightness REAL,
            confidence REAL,
            frp REAL,
            classification TEXT,
            severity TEXT,
            landcover TEXT,
            intelligence_score REAL,
            industrial_likelihood REAL,
            satellite TEXT,
            acq_date TEXT,
            acq_time TEXT,
            first_seen TEXT,
            last_seen TEXT
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_thermal_events_grid
        ON thermal_events (grid_lat, grid_lon)
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_thermal_events_acq_date
        ON thermal_events (acq_date)
        """
    )

    connection.commit()
    connection.close()


def build_event_key(event):
    return (
        f"{round(event['latitude'], 4)}:"
        f"{round(event['longitude'], 4)}:"
        f"{event.get('timestamp', '')}:"
        f"{event.get('source', '')}"
    )


def insert_events(events):
    if not events:
        return

    now = datetime.now(
        timezone.utc
    ).isoformat()

    connection = get_connection()

    for event in events:
        try:
            timestamp = event.get(
                "timestamp",
                "",
            ).strip()

            acq_date, _, acq_time = (
                timestamp.partition(" ")
            )

            event_key = build_event_key(
                event
            )

            grid_lat = round(
                event["latitude"],
                2,
            )

            grid_lon = round(
                event["longitude"],
                2,
            )

            connection.execute(
                """
                INSERT INTO thermal_events (
                    event_key, latitude, longitude,
                    grid_lat, grid_lon, brightness,
                    confidence, frp, classification,
                    severity, landcover, intelligence_score,
                    industrial_likelihood, satellite,
                    acq_date, acq_time, first_seen, last_seen
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(event_key) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    classification = excluded.classification,
                    severity = excluded.severity,
                    landcover = excluded.landcover,
                    intelligence_score = excluded.intelligence_score,
                    industrial_likelihood = excluded.industrial_likelihood
                """,
                (
                    event_key,
                    event["latitude"],
                    event["longitude"],
                    grid_lat,
                    grid_lon,
                    event["brightness"],
                    event["confidence"],
                    event["frp"],
                    event["classification"],
                    event["severity"],
                    event.get("landcover", "unknown"),
                    event["intelligence_score"],
                    event["industrial_likelihood"],
                    event["source"],
                    acq_date,
                    acq_time,
                    now,
                    now,
                ),
            )

        except Exception:
            continue

    connection.commit()
    connection.close()


def get_persistent_sources(
    min_days_active=2,
    lookback_days=30,
    limit=50,
):
    connection = get_connection()

    rows = connection.execute(
        """
        SELECT
            grid_lat,
            grid_lon,
            COUNT(DISTINCT acq_date) AS days_active,
            COUNT(*) AS detection_count,
            MAX(intelligence_score) AS max_score,
            AVG(intelligence_score) AS avg_score,
            MAX(industrial_likelihood) AS max_industrial_likelihood,
            MIN(acq_date) AS first_detected,
            MAX(acq_date) AS last_detected,
            GROUP_CONCAT(DISTINCT classification) AS classifications,
            GROUP_CONCAT(DISTINCT landcover) AS landcovers
        FROM thermal_events
        WHERE acq_date >= date('now', ?)
        GROUP BY grid_lat, grid_lon
        HAVING days_active >= ?
        ORDER BY days_active DESC, max_score DESC
        LIMIT ?
        """,
        (
            f"-{lookback_days} days",
            min_days_active,
            limit,
        ),
    ).fetchall()

    connection.close()

    results = []

    for row in rows:
        results.append(
            {
                "latitude": row["grid_lat"],
                "longitude": row["grid_lon"],
                "days_active": row["days_active"],
                "detection_count": row["detection_count"],
                "max_intelligence_score": round(
                    row["max_score"] or 0,
                    1,
                ),
                "avg_intelligence_score": round(
                    row["avg_score"] or 0,
                    1,
                ),
                "max_industrial_likelihood": row[
                    "max_industrial_likelihood"
                ],
                "first_detected": row["first_detected"],
                "last_detected": row["last_detected"],
                "classifications": (
                    row["classifications"] or ""
                ).split(","),
                "landcovers": (
                    row["landcovers"] or ""
                ).split(","),
            }
        )

    return results


def get_daily_trends(lookback_days=14):
    connection = get_connection()

    rows = connection.execute(
        """
        SELECT
            acq_date,
            severity,
            COUNT(*) AS event_count
        FROM thermal_events
        WHERE acq_date >= date('now', ?)
        AND acq_date != ''
        GROUP BY acq_date, severity
        ORDER BY acq_date ASC
        """,
        (f"-{lookback_days} days",),
    ).fetchall()

    connection.close()

    trends = {}

    for row in rows:
        date_key = row["acq_date"]

        if date_key not in trends:
            trends[date_key] = {
                "date": date_key,
                "HIGH": 0,
                "MEDIUM": 0,
                "LOW": 0,
                "total": 0,
            }

        severity = row["severity"]

        if severity in trends[date_key]:
            trends[date_key][severity] = (
                row["event_count"]
            )

        trends[date_key]["total"] += (
            row["event_count"]
        )

    return sorted(
        trends.values(),
        key=lambda item: item["date"],
    )


def get_total_stored_events():
    connection = get_connection()

    row = connection.execute(
        "SELECT COUNT(*) AS total FROM thermal_events"
    ).fetchone()

    connection.close()

    return row["total"] if row else 0


def get_all_events_for_training():
    connection = get_connection()

    rows = connection.execute(
        """
        SELECT
            brightness,
            confidence,
            frp,
            intelligence_score,
            industrial_likelihood,
            landcover,
            classification,
            severity
        FROM thermal_events
        """
    ).fetchall()

    connection.close()

    return [dict(row) for row in rows]


def get_location_baselines():
    connection = get_connection()

    rows = connection.execute(
        """
        SELECT
            grid_lat,
            grid_lon,
            AVG(intelligence_score) AS baseline_score,
            AVG(frp) AS baseline_frp,
            COUNT(DISTINCT acq_date) AS days_seen,
            COUNT(*) AS detection_count
        FROM thermal_events
        GROUP BY grid_lat, grid_lon
        """
    ).fetchall()

    connection.close()

    baselines = {}

    for row in rows:
        key = (
            row["grid_lat"],
            row["grid_lon"],
        )

        baselines[key] = {
            "baseline_score": row["baseline_score"],
            "baseline_frp": row["baseline_frp"],
            "days_seen": row["days_seen"],
            "detection_count": row["detection_count"],
        }

    return baselines