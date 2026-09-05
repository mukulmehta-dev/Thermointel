import { useEffect, useMemo, useState } from "react";
import {
  CircleMarker,
  MapContainer,
  Popup,
  TileLayer,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./app.css";
import Globe3D from "./Globe3D";

type Severity = "HIGH" | "MEDIUM" | "LOW";
type SeverityFilter = "ALL" | Severity;

type BaselineStatus =
  | "NEW_LOCATION"
  | "CONSISTENT_WITH_HISTORY"
  | "ANOMALOUS_SPIKE";

type WorkstationTab =
  | "map"
  | "intelligence"
  | "analytics"
  | "reports"
  | "settings";

type ThermalEvent = {
  id: string;
  latitude: number;
  longitude: number;
  brightness: number;
  confidence: number;
  frp: number;
  classification: string;
  severity: Severity;
  landcover?: string;
  intelligence_score: number;
  industrial_likelihood: number;
  intelligence_reason: string;
  location: string;
  source: string;
  timestamp: string;
  cluster_id?: string | null;
  cluster_size?: number;
  cluster_peak_frp?: number;
  cluster_avg_score?: number;
  cluster_max_score?: number;
  cluster_severity?: Severity;
  baseline_status?: BaselineStatus;
  baseline_ratio?: number | null;
  ml_classification?: string | null;
  ml_confidence?: number | null;
};

type EventsResponse = {
  source: string;
  raw_count?: number;
  filtered_count?: number;
  count: number;
  events: ThermalEvent[];
  retrieved_at: string;
};

type IndustrialFeature = {
  name: string;
  type: string;
  distance_km: number;
  latitude: number;
  longitude: number;
  osm_type?: string;
  osm_id?: number;
};

type IndustrialContextResponse = {
  query: {
    latitude: number;
    longitude: number;
    radius_km: number;
  };
  nearby_count: number;
  nearest: IndustrialFeature | null;
  features: IndustrialFeature[];
  source: string;
  status?: string;
  retrieved_at: string;
};

type TrendDay = {
  date: string;
  HIGH: number;
  MEDIUM: number;
  LOW: number;
  total: number;
};

type TrendsResponse = {
  lookback_days: number;
  days_with_data: number;
  total_stored_events: number;
  trends: TrendDay[];
  retrieved_at: string;
};

type PersistentSource = {
  latitude: number;
  longitude: number;
  days_active: number;
  detection_count: number;
  max_intelligence_score: number;
  avg_intelligence_score: number;
  max_industrial_likelihood: number;
  first_detected: string;
  last_detected: string;
  classifications: string[];
  landcovers: string[];
};

type PersistentResponse = {
  min_days_active: number;
  lookback_days: number;
  count: number;
  sources: PersistentSource[];
  retrieved_at: string;
};

const API_URL = "http://127.0.0.1:8001";

function severityClass(severity: Severity) {
  return severity.toLowerCase();
}

function markerColor(severity: Severity) {
  switch (severity) {
    case "HIGH":
      return "#ff4545";
    case "MEDIUM":
      return "#ffd34d";
    default:
      return "#45df8d";
  }
}

function scoreClass(score: number) {
  if (score >= 70) return "score-high";
  if (score >= 40) return "score-medium";
  return "score-low";
}

function formatFeatureType(type: string) {
  if (!type) return "Industrial";

  return type
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatFeatureName(name: string, type: string) {
  const normalized = (name || "").trim().toLowerCase();

  if (
    !normalized ||
    normalized === "unnamed industrial feature" ||
    normalized === "unnamed feature" ||
    normalized === "unknown"
  ) {
    const formattedType = formatFeatureType(type);

    return `Unnamed ${formattedType.toLowerCase()}`;
  }

  return name;
}

function getFeatureTypeClass(type: string) {
  const normalized = type.toLowerCase();

  if (
    normalized.includes("mine") ||
    normalized.includes("coal")
  ) {
    return "mine";
  }

  if (
    normalized.includes("power") ||
    normalized.includes("plant")
  ) {
    return "power";
  }

  if (
    normalized.includes("steel") ||
    normalized.includes("works")
  ) {
    return "works";
  }

  if (
    normalized.includes("port") ||
    normalized.includes("harbour")
  ) {
    return "port";
  }

  return "industrial";
}

function formatTrendLabel(dateString: string) {
  if (!dateString) return "--";

  const parts = dateString.split("-");

  if (parts.length !== 3) return dateString;

  return `${parts[1]}/${parts[2]}`;
}

function baselineStatusClass(status?: BaselineStatus) {
  if (status === "ANOMALOUS_SPIKE") return "spike";
  if (status === "CONSISTENT_WITH_HISTORY") return "consistent";
  return "new";
}

function baselineStatusLabel(status?: BaselineStatus) {
  if (status === "ANOMALOUS_SPIKE") return "Anomalous Spike";
  if (status === "CONSISTENT_WITH_HISTORY") return "Consistent with History";
  return "New Location";
}

// Illustrative facility silhouettes — not live imagery.
// Chosen over a live external image service deliberately:
// this project already hit real reliability issues depending
// on a third-party service (Overpass) for live data, and this
// panel isn't backed by real per-site photography anyway.
function FacilityIcon({ typeClass }: { typeClass: string }) {
  const paths: Record<string, string> = {
    mine: "M4 42h56L48 20l-8 10-6-14-8 16-6-8z",
    power: "M20 8h10l-4 14h8L22 44l3-16h-8z M34 20h18v22H34z",
    works: "M6 44V22l10-8v8l10-8v8l10-8v8l10-8v30z",
    port: "M8 40h48M14 40V24l8-4 8 4v16M34 40V20l8-4 8 4v20M4 44h56",
    industrial: "M8 44V18l12 8V18l12 8V18l12 8V44z M44 44V26h8v18z",
  };

  const path = paths[typeClass] || paths.industrial;

  return (
    <svg viewBox="0 0 64 48" width="100%" height="100%">
      <path
        d={path}
        fill="none"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

function App() {
  const [activeTab, setActiveTab] =
    useState<WorkstationTab>("map");

  const [events, setEvents] =
    useState<ThermalEvent[]>([]);

  const [selectedEvent, setSelectedEvent] =
    useState<ThermalEvent | null>(null);

  const [industrialContext, setIndustrialContext] =
    useState<IndustrialContextResponse | null>(null);

  const [industrialLoading, setIndustrialLoading] =
    useState(false);

  const [industrialError, setIndustrialError] =
    useState("");

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  const [lastUpdated, setLastUpdated] =
    useState("");

  const [searchTerm, setSearchTerm] =
    useState("");

  const [severityFilter, setSeverityFilter] =
    useState<SeverityFilter>("ALL");

  const [trends, setTrends] =
    useState<TrendDay[]>([]);

  const [trendsTotal, setTrendsTotal] =
    useState(0);

  const [trendsLoading, setTrendsLoading] =
    useState(true);

  const [persistentSources, setPersistentSources] =
    useState<PersistentSource[]>([]);

  const [persistentLoading, setPersistentLoading] =
    useState(true);

  const [refreshIntervalMinutes, setRefreshIntervalMinutes] =
    useState(5);

  async function loadEvents() {
    try {
      setError("");

      const response = await fetch(
        `${API_URL}/api/firms/events`
      );

      if (!response.ok) {
        throw new Error(
          `API returned ${response.status}`
        );
      }

      const data: EventsResponse =
        await response.json();

      setEvents(
        Array.isArray(data.events)
          ? data.events
          : []
      );

      setLastUpdated(
        data.retrieved_at || ""
      );
    } catch (err) {
      console.error(
        "ThermoIntel API error:",
        err
      );

      setError(
        "Unable to connect to ThermoIntel backend."
      );
    } finally {
      setLoading(false);
    }
  }

  async function loadTrends() {
    try {
      const response = await fetch(
        `${API_URL}/api/firms/trends?lookback_days=14`
      );

      if (!response.ok) {
        throw new Error(
          `API returned ${response.status}`
        );
      }

      const data: TrendsResponse =
        await response.json();

      setTrends(
        Array.isArray(data.trends)
          ? data.trends
          : []
      );

      setTrendsTotal(
        data.total_stored_events || 0
      );
    } catch (err) {
      console.error(
        "Trends API error:",
        err
      );
    } finally {
      setTrendsLoading(false);
    }
  }

  async function loadPersistentSources() {
    try {
      const response = await fetch(
        `${API_URL}/api/firms/persistent?min_days_active=2&lookback_days=30&limit=10`
      );

      if (!response.ok) {
        throw new Error(
          `API returned ${response.status}`
        );
      }

      const data: PersistentResponse =
        await response.json();

      setPersistentSources(
        Array.isArray(data.sources)
          ? data.sources
          : []
      );
    } catch (err) {
      console.error(
        "Persistent sources API error:",
        err
      );
    } finally {
      setPersistentLoading(false);
    }
  }

  async function loadIndustrialContext(
    event: ThermalEvent
  ) {
    try {
      setIndustrialLoading(true);
      setIndustrialError("");
      setIndustrialContext(null);

      const params = new URLSearchParams({
        latitude: String(event.latitude),
        longitude: String(event.longitude),
        radius_km: "10",
      });

      const response = await fetch(
        `${API_URL}/api/firms/context?${params.toString()}`
      );

      if (!response.ok) {
        throw new Error(
          `API returned ${response.status}`
        );
      }

      const data: IndustrialContextResponse =
        await response.json();

      setIndustrialContext(data);
    } catch (err) {
      console.error(
        "Industrial context API error:",
        err
      );

      setIndustrialError(
        "Unable to retrieve industrial context."
      );
    } finally {
      setIndustrialLoading(false);
    }
  }

  function handleEventSelection(
    event: ThermalEvent
  ) {
    setSelectedEvent(event);
    loadIndustrialContext(event);
  }

  function closeEventPanel() {
    setSelectedEvent(null);
    setIndustrialContext(null);
    setIndustrialError("");
  }

  function refreshAll() {
    loadEvents();
    loadTrends();
    loadPersistentSources();
  }

  useEffect(() => {
    refreshAll();

    if (refreshIntervalMinutes <= 0) {
      return;
    }

    const interval = setInterval(
      refreshAll,
      refreshIntervalMinutes * 60 * 1000
    );

    return () => clearInterval(interval);
  }, [refreshIntervalMinutes]);

  const highCount = useMemo(
    () =>
      events.filter(
        (event) => event.severity === "HIGH"
      ).length,
    [events]
  );

  const mediumCount = useMemo(
    () =>
      events.filter(
        (event) => event.severity === "MEDIUM"
      ).length,
    [events]
  );

  const lowCount = useMemo(
    () =>
      events.filter(
        (event) => event.severity === "LOW"
      ).length,
    [events]
  );

  const averageScore = useMemo(() => {
    if (!events.length) return 0;

    return Math.round(
      events.reduce(
        (total, event) =>
          total +
          (event.intelligence_score || 0),
        0
      ) / events.length
    );
  }, [events]);

  const highestRiskEvent = useMemo(() => {
    if (!events.length) return null;

    return [...events].sort(
      (a, b) =>
        (b.intelligence_score || 0) -
        (a.intelligence_score || 0)
    )[0];
  }, [events]);

  const filteredEvents = useMemo(() => {
    const query = searchTerm
      .trim()
      .toLowerCase();

    return events.filter((event) => {
      const matchesSeverity =
        severityFilter === "ALL" ||
        event.severity === severityFilter;

      const searchableText = [
        event.location,
        event.classification,
        event.source,
        event.severity,
      ]
        .join(" ")
        .toLowerCase();

      const matchesSearch =
        !query ||
        searchableText.includes(query);

      return (
        matchesSeverity &&
        matchesSearch
      );
    });
  }, [
    events,
    searchTerm,
    severityFilter,
  ]);

  const rankedEvents = useMemo(() => {
    return [...events]
      .sort(
        (a, b) =>
          (b.intelligence_score || 0) -
          (a.intelligence_score || 0)
      )
      .slice(0, 5);
  }, [events]);

  const maxTrendTotal = useMemo(() => {
    if (!trends.length) return 1;

    return Math.max(
      1,
      ...trends.map((day) => day.total)
    );
  }, [trends]);

  const classificationBreakdown = useMemo(() => {
    const counts: Record<string, number> = {};

    events.forEach((event) => {
      counts[event.classification] =
        (counts[event.classification] || 0) + 1;
    });

    return Object.entries(counts).sort(
      (a, b) => b[1] - a[1]
    );
  }, [events]);

  const landcoverBreakdown = useMemo(() => {
    const counts: Record<string, number> = {};

    events.forEach((event) => {
      const key = event.landcover || "unknown";
      counts[key] = (counts[key] || 0) + 1;
    });

    return Object.entries(counts).sort(
      (a, b) => b[1] - a[1]
    );
  }, [events]);

  const baselineBreakdown = useMemo(() => {
    const counts: Record<string, number> = {
      NEW_LOCATION: 0,
      CONSISTENT_WITH_HISTORY: 0,
      ANOMALOUS_SPIKE: 0,
    };

    events.forEach((event) => {
      const key = event.baseline_status || "NEW_LOCATION";

      if (key in counts) {
        counts[key] += 1;
      }
    });

    return counts;
  }, [events]);

  function exportEventsAsCsv() {
    const headers = [
      "id",
      "latitude",
      "longitude",
      "classification",
      "severity",
      "brightness",
      "frp",
      "confidence",
      "intelligence_score",
      "industrial_likelihood",
      "landcover",
      "baseline_status",
      "timestamp",
    ];

    const rows = filteredEvents.map((event) => [
      event.id,
      event.latitude,
      event.longitude,
      event.classification,
      event.severity,
      event.brightness,
      event.frp,
      event.confidence,
      event.intelligence_score,
      event.industrial_likelihood,
      event.landcover || "unknown",
      event.baseline_status || "",
      event.timestamp,
    ]);

    const csvContent = [headers, ...rows]
      .map((row) =>
        row
          .map((value) => `"${String(value).replace(/"/g, '""')}"`)
          .join(",")
      )
      .join("\n");

    const blob = new Blob([csvContent], {
      type: "text/csv;charset=utf-8;",
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, "-");

    link.href = url;
    link.download = `thermointel-events-${timestamp}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  return (
    <div className="workstation-shell">

      {/* ============ FULL-BLEED MAP LAYER ============ */}

      <div className="workstation-map-layer">

        <MapContainer
          center={[22.5, 79]}
          zoom={5}
          minZoom={4}
          maxZoom={12}
          scrollWheelZoom={true}
          zoomControl={false}
          className="workstation-map"
        >

          <TileLayer
            attribution="&copy; OpenStreetMap contributors &copy; CARTO"
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />

          {filteredEvents.map((event) => {

            const color = markerColor(event.severity);

            const radius = Math.min(
              14,
              Math.max(5, 5 + event.frp / 3)
            );

            return (
              <CircleMarker
                key={event.id}
                center={[event.latitude, event.longitude]}
                radius={radius}
                pathOptions={{
                  color,
                  fillColor: color,
                  fillOpacity: 0.85,
                  weight: 2,
                }}
                eventHandlers={{
                  click: () => handleEventSelection(event),
                }}
              >

                <Popup>
                  <div className="map-popup">

                    <div
                      className={`popup-severity ${severityClass(
                        event.severity
                      )}`}
                    >
                      {event.severity}
                    </div>

                    <strong>{event.classification}</strong>

                    <div>📍 {event.location}</div>
                    <div>Intelligence Score: {event.intelligence_score}</div>
                    <div>Industrial Likelihood: {event.industrial_likelihood}%</div>
                    <div>Brightness: {event.brightness} K</div>
                    <div>FRP: {event.frp} MW</div>
                    <div>Confidence: {event.confidence}%</div>

                    {event.ml_classification && (
                      <div>
                        AI Prediction: {event.ml_classification} (
                        {event.ml_confidence}%)
                      </div>
                    )}

                    {event.baseline_status && (
                      <div>
                        Baseline: {baselineStatusLabel(event.baseline_status)}
                      </div>
                    )}

                  </div>
                </Popup>

              </CircleMarker>
            );
          })}

        </MapContainer>

      </div>

      {/* ============ TOP NAV ============ */}

      <nav className="workstation-topnav glass">

        <div className="topnav-brand">
          <div className="topnav-brand-icon">
            <svg viewBox="0 0 40 40" width="22" height="22">
              <polygon
                points="20,3 35,11.5 35,28.5 20,37 5,28.5 5,11.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              />
              <circle cx="20" cy="20" r="3.2" fill="currentColor" />
              <path
                d="M20 20 L20 9 M20 20 L28.5 25 M20 20 L11.5 25"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
              <circle cx="20" cy="9" r="1.8" fill="currentColor" opacity="0.8" />
            </svg>
          </div>
          <div>
            <div className="topnav-brand-name">
              <span className="brand-part-a">Thermo</span>
              <span className="brand-part-b">Intel</span>
            </div>
            <div className="topnav-brand-subtitle">Industrial Thermal Intelligence</div>
          </div>
        </div>

        <div className="topnav-tabs">

          <button
            className={`topnav-tab ${activeTab === "map" ? "active" : ""}`}
            onClick={() => setActiveTab("map")}
          >
            LIVE MAP
          </button>

          <button
            className={`topnav-tab ${activeTab === "intelligence" ? "active" : ""}`}
            onClick={() => setActiveTab("intelligence")}
          >
            INTELLIGENCE
          </button>

          <button
            className={`topnav-tab ${activeTab === "analytics" ? "active" : ""}`}
            onClick={() => setActiveTab("analytics")}
          >
            ANALYTICS
          </button>

          <button
            className={`topnav-tab ${activeTab === "reports" ? "active" : ""}`}
            onClick={() => setActiveTab("reports")}
          >
            REPORTS
          </button>

          <button
            className={`topnav-tab ${activeTab === "settings" ? "active" : ""}`}
            onClick={() => setActiveTab("settings")}
          >
            SETTINGS
          </button>

        </div>

        <div className="topnav-status">
          <span className="pulse" />
          <div>
            <div className="topnav-status-label">
              {loading ? "SYNCING" : "DATA STREAM ACTIVE"}
            </div>
            <div className="topnav-status-meta">VIIRS &bull; NOAA-20</div>
          </div>
          <button
            className="topnav-refresh"
            onClick={refreshAll}
            disabled={loading}
          >
            &#8635;
          </button>
        </div>

      </nav>

      {error && (
        <div className="workstation-error-banner glass">
          &#9888; {error}
        </div>
      )}

      {/* ============ LIVE MAP TAB: FLOATING PANELS ============ */}

      {activeTab === "map" && (
        <>

          <aside className="workstation-panel panel-left glass">

            <div className="panel-section">
              <div className="panel-section-title">DATA SOURCES</div>

              <div className="panel-source-row">
                <span>NASA FIRMS</span>
                <span className="source-live-badge">LIVE</span>
              </div>
              <div className="panel-source-row">
                <span>OpenStreetMap</span>
                <span className="source-check">&#10003;</span>
              </div>
              <div className="panel-source-row">
                <span>India Boundary</span>
                <span className="source-check">&#10003;</span>
              </div>
            </div>

            <div className="panel-section">
              <div className="panel-section-title">FILTERS</div>

              <div className="panel-filter-label">Severity</div>
              <select
                className="panel-select"
                value={severityFilter}
                onChange={(event) =>
                  setSeverityFilter(event.target.value as SeverityFilter)
                }
              >
                <option value="ALL">All</option>
                <option value="HIGH">High</option>
                <option value="MEDIUM">Medium</option>
                <option value="LOW">Low</option>
              </select>

              <div className="panel-filter-label">Search</div>
              <div className="panel-search-wrapper">
                <input
                  className="panel-search-input"
                  type="text"
                  value={searchTerm}
                  onChange={(event) => setSearchTerm(event.target.value)}
                  placeholder="Location, classification..."
                />
                {searchTerm && (
                  <button
                    className="panel-search-clear"
                    onClick={() => setSearchTerm("")}
                    type="button"
                  >
                    &times;
                  </button>
                )}
              </div>
            </div>

            <div className="panel-section panel-stats">

              <div className="panel-stat-row">
                <span className="panel-stat-label">TOTAL EVENTS</span>
                <span className="panel-stat-value">
                  {loading ? "\u2014" : events.length}
                </span>
              </div>

              <div className="panel-stat-row danger">
                <span className="panel-stat-label">HIGH SEVERITY</span>
                <span className="panel-stat-value">
                  {loading ? "\u2014" : highCount}
                </span>
              </div>

              <div className="panel-stat-row warning">
                <span className="panel-stat-label">MEDIUM SEVERITY</span>
                <span className="panel-stat-value">
                  {loading ? "\u2014" : mediumCount}
                </span>
              </div>

              <div className="panel-stat-row good">
                <span className="panel-stat-label">LOW SEVERITY</span>
                <span className="panel-stat-value">
                  {loading ? "\u2014" : lowCount}
                </span>
              </div>

              <div className="panel-stat-row intel">
                <span className="panel-stat-label">AVG INTELLIGENCE</span>
                <span className="panel-stat-value">
                  {loading ? "\u2014" : averageScore}
                </span>
              </div>

            </div>

          </aside>

          <div className="workstation-panel panel-bottom-left glass">

            <div className="panel-section-title">
              THERMAL ACTIVITY &bull; LAST 14 DAYS
            </div>

            {trendsLoading && (
              <div className="panel-empty-note">Loading trend history...</div>
            )}

            {!trendsLoading && trends.length === 0 && (
              <div className="panel-empty-note">
                Not enough history yet.
              </div>
            )}

            {!trendsLoading && trends.length > 0 && (
              <div className="mini-trend-chart">
                {trends.map((day) => (
                  <div className="mini-trend-bar-group" key={day.date}>
                    <div className="mini-trend-bar-stack">
                      <div
                        className="mini-trend-segment low"
                        style={{ height: `${(day.LOW / maxTrendTotal) * 100}%` }}
                      />
                      <div
                        className="mini-trend-segment medium"
                        style={{ height: `${(day.MEDIUM / maxTrendTotal) * 100}%` }}
                      />
                      <div
                        className="mini-trend-segment high"
                        style={{ height: `${(day.HIGH / maxTrendTotal) * 100}%` }}
                      />
                    </div>
                    <div className="mini-trend-label">
                      {formatTrendLabel(day.date)}
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div className="mini-legend">
              <span><span className="legend-dot high-dot" /> High</span>
              <span><span className="legend-dot medium-dot" /> Medium</span>
              <span><span className="legend-dot low-dot" /> Low</span>
            </div>

          </div>

          {selectedEvent && (

            <aside className="workstation-panel panel-right glass">

              <button className="close-panel" onClick={closeEventPanel}>
                &times;
              </button>

              <div
                className={`event-severity ${severityClass(selectedEvent.severity)}`}
              >
                {selectedEvent.severity}
              </div>

              <h2 className="panel-event-title">{selectedEvent.classification}</h2>

              <div className="event-location">
                📍 {selectedEvent.location}
              </div>

              {industrialLoading && (
                <div className="facility-thumb-loading">
                  Scanning nearby infrastructure...
                </div>
              )}

              {!industrialLoading && industrialContext?.nearest && (
                <div className="facility-thumb">
                  <div
                    className={`facility-icon-box ${getFeatureTypeClass(
                      industrialContext.nearest.type
                    )}`}
                  >
                    <FacilityIcon
                      typeClass={getFeatureTypeClass(
                        industrialContext.nearest.type
                      )}
                    />
                  </div>
                  <div className="facility-thumb-caption">
                    <strong>
                      {formatFeatureName(
                        industrialContext.nearest.name,
                        industrialContext.nearest.type
                      )}
                    </strong>
                    <span>
                      {industrialContext.nearest.distance_km} km away &bull; Illustrative
                    </span>
                  </div>
                </div>
              )}

              <div className="intelligence-box">
                <div className="intel-box-header">
                  <span>INTELLIGENCE SCORE</span>
                  <strong>{selectedEvent.intelligence_score}</strong>
                </div>
                <div className="score-bar">
                  <div
                    className={`score-fill ${scoreClass(
                      selectedEvent.intelligence_score
                    )}`}
                    style={{
                      width: `${Math.min(100, selectedEvent.intelligence_score)}%`,
                    }}
                  />
                </div>
              </div>

              <div className="likelihood-box">
                <div>
                  <span>INDUSTRIAL LIKELIHOOD</span>
                  <strong>{selectedEvent.industrial_likelihood}%</strong>
                </div>
                <div className="likelihood-bar">
                  <div
                    style={{
                      width: `${Math.min(
                        100,
                        selectedEvent.industrial_likelihood
                      )}%`,
                    }}
                  />
                </div>
              </div>

              <div className="reason-box">
                <div className="reason-title">WHY THIS EVENT WAS FLAGGED</div>
                <div className="reason-text">{selectedEvent.intelligence_reason}</div>
              </div>

              {selectedEvent.ml_classification && (
                <div className="ml-prediction-box">
                  <div className="ml-prediction-header">
                    <span>AI MODEL PREDICTION</span>
                    <strong>{selectedEvent.ml_confidence}%</strong>
                  </div>
                  <div className="ml-prediction-value">
                    {selectedEvent.ml_classification}
                  </div>
                </div>
              )}

              {selectedEvent.baseline_status && (
                <div
                  className={`baseline-badge ${baselineStatusClass(
                    selectedEvent.baseline_status
                  )}`}
                >
                  <span className="baseline-badge-label">
                    {baselineStatusLabel(selectedEvent.baseline_status)}
                  </span>
                  {selectedEvent.baseline_ratio != null && (
                    <span className="baseline-badge-ratio">
                      {selectedEvent.baseline_ratio}x baseline
                    </span>
                  )}
                </div>
              )}

              <div className="event-details">
                <div>
                  <span>Brightness</span>
                  <strong>{selectedEvent.brightness} K</strong>
                </div>
                <div>
                  <span>FRP</span>
                  <strong>{selectedEvent.frp} MW</strong>
                </div>
                <div>
                  <span>Confidence</span>
                  <strong>{selectedEvent.confidence}%</strong>
                </div>
                <div>
                  <span>Satellite</span>
                  <strong>{selectedEvent.source}</strong>
                </div>
              </div>

              {industrialError && (
                <div className="industrial-error">&#9888; {industrialError}</div>
              )}

              <div className="event-time">
                Acquisition: {selectedEvent.timestamp}
              </div>

            </aside>

          )}

          <div className="workstation-panel panel-bottom-right glass">

            <div className="panel-section-title">ORBITAL CONTEXT</div>

            <div className="globe-visual">
              <Globe3D />
            </div>

            <div className="globe-caption">
              India monitored via VIIRS &bull; NOAA-20 polar orbit
            </div>

          </div>

        </>
      )}

      {/* ============ INTELLIGENCE TAB: DETAILED VIEWS ============ */}

      {activeTab === "intelligence" && (

        <div className="workstation-scroll-view">

          <div className="intel-view-inner">

            <section className="persistent-card">

              <div className="persistent-header">
                <div>
                  <div className="persistent-title">PERSISTENT THERMAL SOURCES</div>
                  <div className="persistent-subtitle">
                    Locations with repeated detections across multiple days
                  </div>
                </div>
                <div className="persistent-badge">{persistentSources.length} FOUND</div>
              </div>

              {persistentLoading && (
                <div className="persistent-empty">Scanning for persistent sources...</div>
              )}

              {!persistentLoading && persistentSources.length === 0 && (
                <div className="persistent-empty">
                  No location has repeated across multiple days yet.
                </div>
              )}

              {!persistentLoading && persistentSources.length > 0 && (
                <>
                  <div className="persistent-table-row persistent-table-heading">
                    <span>#</span>
                    <span>LOCATION</span>
                    <span>DAYS</span>
                    <span>PEAK SCORE</span>
                    <span>CLASSIFICATION</span>
                  </div>

                  {persistentSources.map((source, index) => (
                    <div
                      className="persistent-table-row persistent-feature-row"
                      key={`${source.latitude}-${source.longitude}`}
                    >
                      <span className="persistent-rank">{index + 1}</span>
                      <span className="persistent-coords">
                        {source.latitude.toFixed(2)}, {source.longitude.toFixed(2)}
                      </span>
                      <span className="persistent-days">{source.days_active}d</span>
                      <span className="persistent-score">
                        {source.max_intelligence_score}
                      </span>
                      <span className="persistent-classification">
                        {source.classifications.filter(Boolean).join(", ")}
                      </span>
                    </div>
                  ))}
                </>
              )}

            </section>

            <section className="ranking-card">

              <div className="ranking-header">
                <div>
                  <div className="ranking-title">PRIORITY INTELLIGENCE</div>
                  <div className="ranking-subtitle">
                    Highest-risk thermal events ranked by intelligence score
                  </div>
                </div>
                <div className="ranking-badge">TOP 5</div>
              </div>

              <div className="ranking-list">
                <div className="ranking-row ranking-heading">
                  <span>#</span>
                  <span>EVENT</span>
                  <span>SEVERITY</span>
                  <span>SCORE</span>
                  <span>INDUSTRIAL</span>
                </div>

                {rankedEvents.map((event, index) => (
                  <button
                    className="ranking-row ranking-event"
                    key={event.id}
                    onClick={() => {
                      setActiveTab("map");
                      handleEventSelection(event);
                    }}
                  >
                    <span className="rank-number">{index + 1}</span>
                    <span className="ranking-event-name">
                      <strong>{event.classification}</strong>
                      <small>{event.location}</small>
                    </span>
                    <span>
                      <span
                        className={`severity-badge ${severityClass(event.severity)}`}
                      >
                        {event.severity}
                      </span>
                    </span>
                    <span
                      className={`ranking-score ${scoreClass(
                        event.intelligence_score
                      )}`}
                    >
                      {event.intelligence_score}
                    </span>
                    <span className="ranking-industrial">
                      {event.industrial_likelihood}%
                    </span>
                  </button>
                ))}
              </div>

            </section>

            <section className="events-card">

              <div className="events-header">
                <div>
                  <div className="events-title">DETECTED THERMAL EVENTS</div>
                  <div className="events-subtitle">
                    {filteredEvents.length} matching events
                    {searchTerm ? ` for "${searchTerm}"` : ""}
                  </div>
                </div>
              </div>

              <div className="events-table">
                <div className="table-row table-heading">
                  <span>SEVERITY</span>
                  <span>LOCATION</span>
                  <span>CLASSIFICATION</span>
                  <span>INTELLIGENCE</span>
                  <span>INDUSTRIAL</span>
                </div>

                {filteredEvents.map((event) => (
                  <button
                    className="table-row table-event"
                    key={event.id}
                    onClick={() => {
                      setActiveTab("map");
                      handleEventSelection(event);
                    }}
                  >
                    <span>
                      <span
                        className={`severity-badge ${severityClass(event.severity)}`}
                      >
                        {event.severity}
                      </span>
                    </span>
                    <span>{event.location}</span>
                    <span>{event.classification}</span>
                    <span>
                      <span
                        className={`table-score ${scoreClass(
                          event.intelligence_score
                        )}`}
                      >
                        {event.intelligence_score}
                      </span>
                    </span>
                    <span>{event.industrial_likelihood}%</span>
                  </button>
                ))}

                {!loading && filteredEvents.length === 0 && (
                  <div className="empty-state">
                    No thermal events match the current search/filter.
                  </div>
                )}
              </div>

            </section>

            <footer className="footer">
              <span>ThermoIntel &bull; Satellite intelligence prototype</span>
              <span>
                {lastUpdated ? `Last update: ${lastUpdated}` : "Waiting for data..."}
              </span>
            </footer>

          </div>

        </div>

      )}

      {/* ============ ANALYTICS TAB ============ */}

      {activeTab === "analytics" && (

        <div className="workstation-scroll-view">
          <div className="intel-view-inner">

            <section className="analytics-card">
              <div className="analytics-card-title">CLASSIFICATION BREAKDOWN</div>
              <div className="analytics-card-subtitle">
                Distribution of all {events.length} currently loaded detections
              </div>

              <div className="breakdown-list">
                {classificationBreakdown.map(([label, count]) => (
                  <div className="breakdown-row" key={label}>
                    <span className="breakdown-label">{label}</span>
                    <div className="breakdown-bar-track">
                      <div
                        className="breakdown-bar-fill"
                        style={{
                          width: `${(count / Math.max(1, events.length)) * 100}%`,
                        }}
                      />
                    </div>
                    <span className="breakdown-count">{count}</span>
                  </div>
                ))}

                {classificationBreakdown.length === 0 && (
                  <div className="panel-empty-note">No data loaded yet.</div>
                )}
              </div>
            </section>

            <section className="analytics-card">
              <div className="analytics-card-title">LAND-COVER CONTEXT</div>
              <div className="analytics-card-subtitle">
                Where detections fall relative to mapped land use
              </div>

              <div className="breakdown-list">
                {landcoverBreakdown.map(([label, count]) => (
                  <div className="breakdown-row" key={label}>
                    <span className="breakdown-label">{label}</span>
                    <div className="breakdown-bar-track">
                      <div
                        className={`breakdown-bar-fill landcover-${label}`}
                        style={{
                          width: `${(count / Math.max(1, events.length)) * 100}%`,
                        }}
                      />
                    </div>
                    <span className="breakdown-count">{count}</span>
                  </div>
                ))}
              </div>
            </section>

            <section className="analytics-card">
              <div className="analytics-card-title">BASELINE STATUS</div>
              <div className="analytics-card-subtitle">
                How detections compare to each location's own history
              </div>

              <div className="baseline-stat-grid">
                <div className="baseline-stat-box new">
                  <div className="baseline-stat-value">
                    {baselineBreakdown.NEW_LOCATION}
                  </div>
                  <div className="baseline-stat-label">New Location</div>
                </div>
                <div className="baseline-stat-box consistent">
                  <div className="baseline-stat-value">
                    {baselineBreakdown.CONSISTENT_WITH_HISTORY}
                  </div>
                  <div className="baseline-stat-label">Consistent</div>
                </div>
                <div className="baseline-stat-box spike">
                  <div className="baseline-stat-value">
                    {baselineBreakdown.ANOMALOUS_SPIKE}
                  </div>
                  <div className="baseline-stat-label">Anomalous Spike</div>
                </div>
              </div>
            </section>

            <footer className="footer">
              <span>ThermoIntel &bull; Satellite intelligence prototype</span>
              <span>
                {lastUpdated ? `Last update: ${lastUpdated}` : "Waiting for data..."}
              </span>
            </footer>

          </div>
        </div>

      )}

      {/* ============ REPORTS TAB ============ */}

      {activeTab === "reports" && (

        <div className="workstation-scroll-view">
          <div className="intel-view-inner">

            <section className="analytics-card">
              <div className="analytics-card-title">EXPORT CURRENT VIEW</div>
              <div className="analytics-card-subtitle">
                Download the {filteredEvents.length} currently filtered events as a spreadsheet-ready CSV file
              </div>

              <button
                className="export-button"
                onClick={exportEventsAsCsv}
                disabled={filteredEvents.length === 0}
              >
                &#8681; Export {filteredEvents.length} Events as CSV
              </button>
            </section>

            <section className="analytics-card">
              <div className="analytics-card-title">SUMMARY REPORT</div>
              <div className="analytics-card-subtitle">
                Snapshot of the current observation window
              </div>

              <div className="report-summary-grid">
                <div className="report-summary-item">
                  <span>Total Detections</span>
                  <strong>{events.length}</strong>
                </div>
                <div className="report-summary-item">
                  <span>High Severity</span>
                  <strong>{highCount}</strong>
                </div>
                <div className="report-summary-item">
                  <span>Medium Severity</span>
                  <strong>{mediumCount}</strong>
                </div>
                <div className="report-summary-item">
                  <span>Low Severity</span>
                  <strong>{lowCount}</strong>
                </div>
                <div className="report-summary-item">
                  <span>Avg Intelligence Score</span>
                  <strong>{averageScore}</strong>
                </div>
                <div className="report-summary-item">
                  <span>Persistent Sources Tracked</span>
                  <strong>{persistentSources.length}</strong>
                </div>
                <div className="report-summary-item">
                  <span>Days of History Stored</span>
                  <strong>{trendsTotal > 0 ? trends.length : 0}</strong>
                </div>
                <div className="report-summary-item">
                  <span>Total Events in Database</span>
                  <strong>{trendsTotal}</strong>
                </div>
              </div>
            </section>

            <footer className="footer">
              <span>ThermoIntel &bull; Satellite intelligence prototype</span>
              <span>
                {lastUpdated ? `Last update: ${lastUpdated}` : "Waiting for data..."}
              </span>
            </footer>

          </div>
        </div>

      )}

      {/* ============ SETTINGS TAB ============ */}

      {activeTab === "settings" && (

        <div className="workstation-scroll-view">
          <div className="intel-view-inner">

            <section className="analytics-card">
              <div className="analytics-card-title">DATA REFRESH</div>
              <div className="analytics-card-subtitle">
                How often the dashboard pulls new data from the backend
              </div>

              <div className="settings-row">
                <span>Auto-refresh interval</span>
                <select
                  className="panel-select settings-select"
                  value={refreshIntervalMinutes}
                  onChange={(event) =>
                    setRefreshIntervalMinutes(Number(event.target.value))
                  }
                >
                  <option value={1}>Every 1 minute</option>
                  <option value={5}>Every 5 minutes</option>
                  <option value={10}>Every 10 minutes</option>
                  <option value={30}>Every 30 minutes</option>
                  <option value={0}>Off (manual only)</option>
                </select>
              </div>
            </section>

            <section className="analytics-card">
              <div className="analytics-card-title">FILTERS</div>
              <div className="analytics-card-subtitle">
                Reset search and severity filters back to default
              </div>

              <button
                className="export-button"
                onClick={() => {
                  setSearchTerm("");
                  setSeverityFilter("ALL");
                }}
              >
                Reset Filters
              </button>
            </section>

            <section className="analytics-card">
              <div className="analytics-card-title">BACKEND CONNECTION</div>
              <div className="analytics-card-subtitle">
                Current API endpoint this dashboard is connected to
              </div>

              <div className="settings-row">
                <span>API URL</span>
                <code className="settings-code">{API_URL}</code>
              </div>

              <div className="settings-row">
                <span>Connection status</span>
                <span className={error ? "settings-status-bad" : "settings-status-good"}>
                  {error ? "Disconnected" : "Connected"}
                </span>
              </div>
            </section>

            <footer className="footer">
              <span>ThermoIntel &bull; Satellite intelligence prototype</span>
              <span>
                {lastUpdated ? `Last update: ${lastUpdated}` : "Waiting for data..."}
              </span>
            </footer>

          </div>
        </div>

      )}

    </div>
  );
}

export default App;