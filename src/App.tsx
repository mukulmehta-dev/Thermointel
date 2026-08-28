import { useEffect, useMemo, useState } from "react";
import {
  CircleMarker,
  MapContainer,
  Popup,
  TileLayer,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./app.css";

type Severity = "HIGH" | "MEDIUM" | "LOW";
type SeverityFilter = "ALL" | Severity;

type BaselineStatus =
  | "NEW_LOCATION"
  | "CONSISTENT_WITH_HISTORY"
  | "ANOMALOUS_SPIKE";

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

function App() {
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

    const interval = setInterval(
      refreshAll,
      5 * 60 * 1000
    );

    return () => clearInterval(interval);
  }, []);

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

  return (
    <div className="app-shell">

      <aside className="sidebar">

        <div className="brand">

          <div className="brand-icon">
            TI
          </div>

          <div>
            <div className="brand-name">
              ThermoIntel
            </div>

            <div className="brand-subtitle">
              INDUSTRIAL THERMAL INTELLIGENCE
            </div>
          </div>

        </div>

        <div className="sidebar-section">

          <div className="section-label">
            SYSTEM
          </div>

          <div className="nav-item active">
            <span>◉</span>
            Live Monitoring
          </div>

          <div className="nav-item">
            <span>◈</span>
            Thermal Events
          </div>

          <div className="nav-item">
            <span>✦</span>
            Intelligence
          </div>

        </div>

        <div className="sidebar-section">

          <div className="section-label">
            DATA SOURCES
          </div>

          <div className="source-item">
            <span className="status-dot green" />
            NASA FIRMS
          </div>

          <div className="source-item">
            <span className="status-dot green" />
            VIIRS NOAA-20
          </div>

          <div className="source-item">
            <span className="status-dot green" />
            India Boundary
          </div>

        </div>

        <div className="sidebar-section">

          <div className="section-label">
            INTELLIGENCE
          </div>

          <div className="intel-status">
            <span className="intel-dot" />
            Risk scoring active
          </div>

          <div className="intel-status">
            <span className="intel-dot" />
            Industrial analysis active
          </div>

        </div>

        <div className="sidebar-bottom">

          <div className="system-status">
            <span className="status-dot green" />
            SYSTEM ONLINE
          </div>

          <div className="version">
            ThermoIntel v0.7.0
          </div>

        </div>

      </aside>

      <main className="main-content">

        <header className="topbar">

          <div>
            <div className="page-title">
              India Thermal Intelligence
            </div>

            <div className="page-subtitle">
              Satellite thermal anomaly detection and industrial risk analysis
            </div>
          </div>

          <div className="topbar-right">

            <div className="live-indicator">
              <span className="pulse" />
              LIVE
            </div>

            <button
              className="refresh-button"
              onClick={refreshAll}
              disabled={loading}
            >
              ↻ {loading ? "Loading..." : "Refresh"}
            </button>

          </div>

        </header>

        {error && (
          <div className="error-banner">
            ⚠ {error}
          </div>
        )}

        <section className="stats-grid">

          <div className="stat-card">

            <div className="stat-label">
              ACTIVE EVENTS
            </div>

            <div className="stat-value">
              {loading ? "—" : events.length}
            </div>

            <div className="stat-description">
              India-filtered detections
            </div>

          </div>

          <div className="stat-card danger">

            <div className="stat-label">
              HIGH SEVERITY
            </div>

            <div className="stat-value">
              {loading ? "—" : highCount}
            </div>

            <div className="stat-description">
              Priority thermal events
            </div>

          </div>

          <div className="stat-card warning">

            <div className="stat-label">
              MEDIUM SEVERITY
            </div>

            <div className="stat-value">
              {loading ? "—" : mediumCount}
            </div>

            <div className="stat-description">
              Potential industrial sources
            </div>

          </div>

          <div className="stat-card intelligence">

            <div className="stat-label">
              AVG INTELLIGENCE SCORE
            </div>

            <div className="stat-value">
              {loading ? "—" : averageScore}
            </div>

            <div className="stat-description">
              System-wide thermal risk score
            </div>

          </div>

        </section>

        <section className="overview-grid">

          <div className="overview-card">

            <div className="overview-label">
              TOTAL DETECTIONS
            </div>

            <div className="overview-number">
              {loading ? "—" : events.length}
            </div>

            <div className="overview-meta">
              Current India observation window
            </div>

          </div>

          <div className="overview-card">

            <div className="overview-label">
              LOW SEVERITY
            </div>

            <div className="overview-number">
              {loading ? "—" : lowCount}
            </div>

            <div className="overview-meta">
              Lower-intensity anomalies
            </div>

          </div>

          <div className="overview-card risk-card">

            <div className="overview-label">
              HIGHEST CURRENT RISK
            </div>

            <div className="overview-number">
              {loading
                ? "—"
                : highestRiskEvent
                  ? highestRiskEvent.intelligence_score
                  : 0}
            </div>

            <div className="overview-meta">
              Intelligence score
            </div>

          </div>

        </section>

        <section className="trends-card">

          <div className="trends-header">

            <div>
              <div className="trends-title">
                THERMAL ACTIVITY TRENDS
              </div>

              <div className="trends-subtitle">
                Daily detections by severity, last 14 days
              </div>
            </div>

            <div className="trends-total">
              {trendsTotal} STORED
            </div>

          </div>

          {trendsLoading && (
            <div className="trends-empty">
              Loading trend history...
            </div>
          )}

          {!trendsLoading && trends.length === 0 && (
            <div className="trends-empty">
              Not enough history yet. Keep the backend running to build up daily trends.
            </div>
          )}

          {!trendsLoading && trends.length > 0 && (

            <div className="trends-chart">

              {trends.map((day) => (

                <div
                  className="trend-bar-group"
                  key={day.date}
                >

                  <div className="trend-bar-stack">

                    <div
                      className="trend-bar-segment low"
                      style={{
                        height: `${(day.LOW / maxTrendTotal) * 100}%`,
                      }}
                    />

                    <div
                      className="trend-bar-segment medium"
                      style={{
                        height: `${(day.MEDIUM / maxTrendTotal) * 100}%`,
                      }}
                    />

                    <div
                      className="trend-bar-segment high"
                      style={{
                        height: `${(day.HIGH / maxTrendTotal) * 100}%`,
                      }}
                    />

                  </div>

                  <div className="trend-bar-total">
                    {day.total}
                  </div>

                  <div className="trend-bar-label">
                    {formatTrendLabel(day.date)}
                  </div>

                </div>

              ))}

            </div>

          )}

        </section>

        <section className="persistent-card">

          <div className="persistent-header">

            <div>
              <div className="persistent-title">
                PERSISTENT THERMAL SOURCES
              </div>

              <div className="persistent-subtitle">
                Locations with repeated detections across multiple days
              </div>
            </div>

            <div className="persistent-badge">
              {persistentSources.length} FOUND
            </div>

          </div>

          {persistentLoading && (
            <div className="persistent-empty">
              Scanning for persistent sources...
            </div>
          )}

          {!persistentLoading &&
            persistentSources.length === 0 && (

              <div className="persistent-empty">
                No location has repeated across multiple days yet. This builds up as the backend keeps running.
              </div>

          )}

          {!persistentLoading &&
            persistentSources.length > 0 && (

              <>

                <div className="persistent-table-row persistent-table-heading">

                  <span>#</span>
                  <span>LOCATION</span>
                  <span>DAYS</span>
                  <span>PEAK SCORE</span>
                  <span>CLASSIFICATION</span>

                </div>

                {persistentSources.map(
                  (source, index) => (

                    <div
                      className="persistent-table-row persistent-feature-row"
                      key={`${source.latitude}-${source.longitude}`}
                    >

                      <span className="persistent-rank">
                        {index + 1}
                      </span>

                      <span className="persistent-coords">
                        {source.latitude.toFixed(2)},{" "}
                        {source.longitude.toFixed(2)}
                      </span>

                      <span className="persistent-days">
                        {source.days_active}d
                      </span>

                      <span className="persistent-score">
                        {source.max_intelligence_score}
                      </span>

                      <span className="persistent-classification">
                        {source.classifications
                          .filter(Boolean)
                          .join(", ")}
                      </span>

                    </div>

                  )
                )}

              </>

          )}

        </section>

        <section className="search-panel">

          <div className="search-panel-header">

            <div>
              <div className="search-panel-title">
                SEARCH EVENTS
              </div>

              <div className="search-panel-subtitle">
                Filter thermal detections by location, classification or severity
              </div>
            </div>

            <div className="search-result-count">
              {filteredEvents.length} / {events.length}
            </div>

          </div>

          <div className="search-controls">

            <div className="search-input-wrapper">

              <span className="search-icon">
                ⌕
              </span>

              <input
                className="search-input"
                type="text"
                value={searchTerm}
                onChange={(event) =>
                  setSearchTerm(
                    event.target.value
                  )
                }
                placeholder="Search location, classification, satellite..."
              />

              {searchTerm && (
                <button
                  className="clear-search"
                  onClick={() =>
                    setSearchTerm("")
                  }
                  type="button"
                >
                  ×
                </button>
              )}

            </div>

            <div className="severity-select-wrapper">

              <span className="filter-label">
                SEVERITY
              </span>

              <select
                className="severity-select"
                value={severityFilter}
                onChange={(event) =>
                  setSeverityFilter(
                    event.target.value as SeverityFilter
                  )
                }
              >
                <option value="ALL">
                  All Severities
                </option>

                <option value="HIGH">
                  High
                </option>

                <option value="MEDIUM">
                  Medium
                </option>

                <option value="LOW">
                  Low
                </option>

              </select>

            </div>

          </div>

        </section>

        <section className="map-card">

          <div className="map-header">

            <div>
              <div className="map-title">
                INDIA THERMAL MONITORING
              </div>

              <div className="map-subtitle">
                NASA FIRMS satellite detections • India boundary filtered
              </div>
            </div>

            <div className="legend">

              <div>
                <span className="legend-dot high-dot" />
                High
              </div>

              <div>
                <span className="legend-dot medium-dot" />
                Medium
              </div>

              <div>
                <span className="legend-dot low-dot" />
                Low
              </div>

            </div>

          </div>

          <div className="map-container real-map">

            <MapContainer
              center={[22.5, 79]}
              zoom={5}
              minZoom={4}
              maxZoom={12}
              scrollWheelZoom={true}
              className="leaflet-map"
            >

              <TileLayer
                attribution="&copy; OpenStreetMap contributors &copy; CARTO"
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
              />

              {filteredEvents.map((event) => {

                const color =
                  markerColor(
                    event.severity
                  );

                const radius = Math.min(
                  14,
                  Math.max(
                    5,
                    5 + event.frp / 3
                  )
                );

                return (
                  <CircleMarker
                    key={event.id}
                    center={[
                      event.latitude,
                      event.longitude,
                    ]}
                    radius={radius}
                    pathOptions={{
                      color,
                      fillColor: color,
                      fillOpacity: 0.85,
                      weight: 2,
                    }}
                    eventHandlers={{
                      click: () =>
                        handleEventSelection(
                          event
                        ),
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

                        <strong>
                          {event.classification}
                        </strong>

                        <div>
                          📍 {event.location}
                        </div>

                        <div>
                          Intelligence Score:{" "}
                          {event.intelligence_score}
                        </div>

                        <div>
                          Industrial Likelihood:{" "}
                          {event.industrial_likelihood}%
                        </div>

                        <div>
                          Brightness:{" "}
                          {event.brightness} K
                        </div>

                        <div>
                          FRP: {event.frp} MW
                        </div>

                        <div>
                          Confidence:{" "}
                          {event.confidence}%
                        </div>

                        {event.ml_classification && (
                          <div>
                            AI Prediction:{" "}
                            {event.ml_classification} (
                            {event.ml_confidence}%)
                          </div>
                        )}

                        {event.baseline_status && (
                          <div>
                            Baseline:{" "}
                            {baselineStatusLabel(
                              event.baseline_status
                            )}
                          </div>
                        )}

                      </div>

                    </Popup>

                  </CircleMarker>
                );
              })}

            </MapContainer>

            {selectedEvent && (

              <div className="event-panel">

                <button
                  className="close-panel"
                  onClick={closeEventPanel}
                >
                  ×
                </button>

                <div
                  className={`event-severity ${severityClass(
                    selectedEvent.severity
                  )}`}
                >
                  {selectedEvent.severity}
                </div>

                <h2>
                  {selectedEvent.classification}
                </h2>

                <div className="event-location">
                  📍 {selectedEvent.location}
                </div>

                <div className="intelligence-box">

                  <div className="intel-box-header">

                    <span>
                      INTELLIGENCE SCORE
                    </span>

                    <strong>
                      {selectedEvent.intelligence_score}
                    </strong>

                  </div>

                  <div className="score-bar">

                    <div
                      className={`score-fill ${scoreClass(
                        selectedEvent.intelligence_score
                      )}`}
                      style={{
                        width: `${Math.min(
                          100,
                          selectedEvent.intelligence_score
                        )}%`,
                      }}
                    />

                  </div>

                </div>

                <div className="likelihood-box">

                  <div>

                    <span>
                      INDUSTRIAL LIKELIHOOD
                    </span>

                    <strong>
                      {selectedEvent.industrial_likelihood}%
                    </strong>

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

                  <div className="reason-title">
                    WHY THIS EVENT WAS FLAGGED
                  </div>

                  <div className="reason-text">
                    {selectedEvent.intelligence_reason}
                  </div>

                </div>

                {selectedEvent.ml_classification && (

                  <div className="ml-prediction-box">

                    <div className="ml-prediction-header">

                      <span>
                        AI MODEL PREDICTION
                      </span>

                      <strong>
                        {selectedEvent.ml_confidence}%
                      </strong>

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
                      {baselineStatusLabel(
                        selectedEvent.baseline_status
                      )}
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
                    <strong>
                      {selectedEvent.brightness} K
                    </strong>
                  </div>

                  <div>
                    <span>FRP</span>
                    <strong>
                      {selectedEvent.frp} MW
                    </strong>
                  </div>

                  <div>
                    <span>Confidence</span>
                    <strong>
                      {selectedEvent.confidence}%
                    </strong>
                  </div>

                  <div>
                    <span>Satellite</span>
                    <strong>
                      {selectedEvent.source}
                    </strong>
                  </div>

                </div>

                <div className="event-time">
                  Acquisition:{" "}
                  {selectedEvent.timestamp}
                </div>

              </div>

            )}

          </div>

        </section>

        {selectedEvent && (

          <section className="industrial-context-card">

            <div className="industrial-context-header">

              <div>

                <div className="industrial-context-title">
                  INDUSTRIAL CONTEXT
                </div>

                <div className="industrial-context-subtitle">
                  Nearby industrial infrastructure within 10 km
                </div>

              </div>

              {industrialContext && (
                <div className="industrial-context-source">
                  OpenStreetMap
                </div>
              )}

            </div>

            {industrialLoading && (

              <div className="industrial-loading">

                <div className="industrial-loading-spinner" />

                <div>
                  <strong>
                    ANALYZING INDUSTRIAL CONTEXT
                  </strong>

                  <span>
                    Searching nearby infrastructure...
                  </span>
                </div>

              </div>

            )}

            {industrialError &&
              !industrialLoading && (

                <div className="industrial-error">
                  ⚠ {industrialError}
                </div>

            )}

            {industrialContext &&
              !industrialLoading && (

                <>

                  <div className="industrial-query-location">

                    <span>
                      📍
                    </span>

                    <strong>
                      {industrialContext.query.latitude.toFixed(
                        3
                      )}
                      ,{" "}
                      {industrialContext.query.longitude.toFixed(
                        3
                      )}
                    </strong>

                  </div>

                  <div className="industrial-summary-grid">

                    <div className="industrial-summary-card">

                      <span className="industrial-summary-label">
                        NEARBY INFRASTRUCTURE
                      </span>

                      <strong className="industrial-summary-value">
                        {industrialContext.nearby_count}
                      </strong>

                    </div>

                    <div className="industrial-summary-card">

                      <span className="industrial-summary-label">
                        SEARCH RADIUS
                      </span>

                      <strong className="industrial-summary-value">
                        {industrialContext.query.radius_km} km
                      </strong>

                    </div>

                    <div className="industrial-summary-card">

                      <span className="industrial-summary-label">
                        NEAREST FEATURE
                      </span>

                      <strong className="industrial-summary-value">
                        {industrialContext.nearest
                          ? `${industrialContext.nearest.distance_km} km`
                          : "—"}
                      </strong>

                    </div>

                  </div>

                  {industrialContext.nearest && (

                    <div className="nearest-industrial-card">

                      <div className="nearest-industrial-header">

                        <div className="nearest-industrial-icon">
                          ◎
                        </div>

                        <div>

                          <div className="nearest-industrial-label">
                            NEAREST INDUSTRIAL FEATURE
                          </div>

                          <div className="nearest-industrial-name">
                            {formatFeatureName(
                              industrialContext.nearest.name,
                              industrialContext.nearest.type
                            )}
                          </div>

                        </div>

                        <div className="nearest-industrial-distance">
                          {industrialContext.nearest.distance_km} km
                        </div>

                      </div>

                      <div className="nearest-industrial-meta">

                        <span
                          className={`industrial-type-badge ${getFeatureTypeClass(
                            industrialContext.nearest.type
                          )}`}
                        >
                          {formatFeatureType(
                            industrialContext.nearest.type
                          )}
                        </span>

                        <span>
                          from selected thermal event
                        </span>

                      </div>

                    </div>

                  )}

                  {industrialContext.features.length > 0 && (

                    <div className="industrial-list-section">

                      <div className="industrial-list-header">

                        <div>

                          <div className="industrial-list-title">
                            NEARBY INFRASTRUCTURE
                          </div>

                          <div className="industrial-list-subtitle">
                            Ranked by distance from thermal event
                          </div>

                        </div>

                        <div className="industrial-list-count">
                          {industrialContext.features.length} FOUND
                        </div>

                      </div>

                      <div className="industrial-table">

                        <div className="industrial-table-row industrial-table-heading">

                          <span>
                            #
                          </span>

                          <span>
                            INFRASTRUCTURE
                          </span>

                          <span>
                            TYPE
                          </span>

                          <span>
                            DISTANCE
                          </span>

                        </div>

                        {industrialContext.features.map(
                          (
                            feature,
                            index
                          ) => (

                            <div
                              className="industrial-table-row industrial-feature-row"
                              key={`${feature.osm_type}-${feature.osm_id}-${index}`}
                            >

                              <span className="industrial-rank">
                                {String(
                                  index + 1
                                ).padStart(
                                  2,
                                  "0"
                                )}
                              </span>

                              <span className="industrial-feature-name">

                                <strong>
                                  {formatFeatureName(
                                    feature.name,
                                    feature.type
                                  )}
                                </strong>

                              </span>

                              <span>

                                <span
                                  className={`industrial-type-badge ${getFeatureTypeClass(
                                    feature.type
                                  )}`}
                                >
                                  {formatFeatureType(
                                    feature.type
                                  )}
                                </span>

                              </span>

                              <span className="industrial-distance">
                                {feature.distance_km} km
                              </span>

                            </div>

                          )
                        )}

                      </div>

                    </div>

                  )}

                  {industrialContext.features.length === 0 && (

                    <div className="industrial-empty">
                      No mapped industrial infrastructure was found within the selected radius.
                    </div>

                  )}

                </>

            )}

          </section>

        )}

        <section className="ranking-card">

          <div className="ranking-header">

            <div>

              <div className="ranking-title">
                PRIORITY INTELLIGENCE
              </div>

              <div className="ranking-subtitle">
                Highest-risk thermal events ranked by intelligence score
              </div>

            </div>

            <div className="ranking-badge">
              TOP 5
            </div>

          </div>

          <div className="ranking-list">

            <div className="ranking-row ranking-heading">

              <span>#</span>
              <span>EVENT</span>
              <span>SEVERITY</span>
              <span>SCORE</span>
              <span>INDUSTRIAL</span>

            </div>

            {rankedEvents.map(
              (event, index) => (

                <button
                  className="ranking-row ranking-event"
                  key={event.id}
                  onClick={() =>
                    handleEventSelection(
                      event
                    )
                  }
                >

                  <span className="rank-number">
                    {index + 1}
                  </span>

                  <span className="ranking-event-name">

                    <strong>
                      {event.classification}
                    </strong>

                    <small>
                      {event.location}
                    </small>

                  </span>

                  <span>

                    <span
                      className={`severity-badge ${severityClass(
                        event.severity
                      )}`}
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

              )
            )}

          </div>

        </section>

        <section className="events-card">

          <div className="events-header">

            <div>

              <div className="events-title">
                DETECTED THERMAL EVENTS
              </div>

              <div className="events-subtitle">
                {filteredEvents.length} matching events
                {searchTerm
                  ? ` for "${searchTerm}"`
                  : ""}
              </div>

            </div>

          </div>

          <div className="events-table">

            <div className="table-row table-heading">

              <span>
                SEVERITY
              </span>

              <span>
                LOCATION
              </span>

              <span>
                CLASSIFICATION
              </span>

              <span>
                INTELLIGENCE
              </span>

              <span>
                INDUSTRIAL
              </span>

            </div>

            {filteredEvents.map(
              (event) => (

                <button
                  className="table-row table-event"
                  key={event.id}
                  onClick={() =>
                    handleEventSelection(
                      event
                    )
                  }
                >

                  <span>

                    <span
                      className={`severity-badge ${severityClass(
                        event.severity
                      )}`}
                    >
                      {event.severity}
                    </span>

                  </span>

                  <span>
                    {event.location}
                  </span>

                  <span>
                    {event.classification}
                  </span>

                  <span>

                    <span
                      className={`table-score ${scoreClass(
                        event.intelligence_score
                      )}`}
                    >
                      {event.intelligence_score}
                    </span>

                  </span>

                  <span>
                    {event.industrial_likelihood}%
                  </span>

                </button>

              )
            )}

            {!loading &&
              filteredEvents.length === 0 && (

                <div className="empty-state">
                  No thermal events match the current search/filter.
                </div>

              )}

          </div>

        </section>

        <footer className="footer">

          <span>
            ThermoIntel • Satellite intelligence prototype
          </span>

          <span>
            {lastUpdated
              ? `Last update: ${lastUpdated}`
              : "Waiting for data..."}
          </span>

        </footer>

      </main>

    </div>

  );
}

export default App;