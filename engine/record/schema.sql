-- SQLite schema for the experiment record (component E11).

CREATE TABLE IF NOT EXISTS model_maps (
    model_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS series (
    series_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    model_id TEXT NOT NULL,
    series_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plans (
    plan_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    experiment_id TEXT NOT NULL,
    approved INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (plan_id, version)
);

CREATE INDEX IF NOT EXISTS idx_plans_experiment_id ON plans (experiment_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    plan_version INTEGER NOT NULL,
    experiment_id TEXT,
    status TEXT NOT NULL,
    message TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_plan_id ON runs (plan_id);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    check_name TEXT NOT NULL,
    component_id TEXT NOT NULL,
    checkpoint_id TEXT,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    control_json TEXT,
    dataset_id TEXT NOT NULL,
    split TEXT NOT NULL,
    seed INTEGER NOT NULL,
    device TEXT NOT NULL,
    precision TEXT NOT NULL,
    quantized INTEGER NOT NULL,
    is_placeholder INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_results_run_id ON results (run_id);
CREATE INDEX IF NOT EXISTS idx_results_check_name ON results (check_name);
CREATE INDEX IF NOT EXISTS idx_results_metric ON results (metric);
CREATE INDEX IF NOT EXISTS idx_results_checkpoint_id ON results (checkpoint_id);

CREATE TABLE IF NOT EXISTS findings (
    finding_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    check_name TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_findings_run_id ON findings (run_id);

CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    item_order INTEGER NOT NULL,
    contains_harmful INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_examples_run_id ON examples (run_id);

CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reports_run_id ON reports (run_id);
