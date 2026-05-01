from flask import Flask, render_template, request, send_file, redirect, jsonify
import requests
import pandas as pd
import io
import time
import threading

app = Flask(__name__)

PROJECT_ID = "gui-firebase-56d95"
COLLECTION = "csi"
BASE_URL = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/{COLLECTION}"

# ── Cache ──
_cache = {"docs": None, "ts": 0}
_cache_lock = threading.Lock()
CACHE_TTL = 600  # 10 minutes

def get_docs():
    now = time.time()
    with _cache_lock:
        if _cache["docs"] is None or now - _cache["ts"] > CACHE_TTL:
            _cache["docs"] = parse_docs(fetch_all_docs())
            _cache["ts"] = now
        return _cache["docs"]

def fetch_all_docs():
    docs = []
    params = {"pageSize": 300}
    while True:
        r = requests.get(BASE_URL, params=params)
        data = r.json()
        docs += data.get("documents", [])
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
    return docs

def parse_value(v):
    if "integerValue" in v: return int(v["integerValue"])
    if "doubleValue" in v: return float(v["doubleValue"])
    if "stringValue" in v: return v["stringValue"]
    if "arrayValue" in v:
        return [parse_value(i) for i in v["arrayValue"].get("values", [])]
    if "mapValue" in v:
        return {k: parse_value(fv) for k, fv in v["mapValue"]["fields"].items()}
    return None

def parse_docs(raw_docs):
    parsed = []
    for d in raw_docs:
        if "fields" not in d:
            continue
        fields = {k: parse_value(v) for k, v in d["fields"].items()}
        parsed.append({"name": d.get("name", ""), **fields})
    return parsed

def group_by_timestamp(docs):
    groups = {}
    for d in docs:
        ts = d.get("session_timestamp", "unknown")
        groups.setdefault(ts, []).append(d)
    return groups

def build_dataframe(docs):
    rows = []
    for doc in docs:
        if "samples" not in doc:
            continue
        ap_idx = doc.get("ap_index", -1)
        for s in doc.get("samples", []):
            rows.append({
                "timestamp": doc.get("session_timestamp", "unknown"),
                "ap": ap_idx,
                "subcarrier": s.get("subcarrier"),
                "packet": s.get("packet"),
                "real": s.get("real"),
                "imag": s.get("imag"),
                "rssi": s.get("rssi"),
                "amplitude": s.get("amplitude"),
                "angle_rad": s.get("angle_rad")
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(by=["subcarrier", "packet"])

def delete_documents(doc_names):
    for name in doc_names:
        requests.delete(f"https://firestore.googleapis.com/v1/{name}")

def build_chart_data(df):
    if df.empty:
        return {}
    charts = {}

    amp_by_sub = {}
    for ap, group in df.groupby("ap"):
        sub_avg = group.groupby("subcarrier")["amplitude"].mean()
        amp_by_sub[str(ap)] = {
            "subcarriers": sub_avg.index.tolist(),
            "amplitudes": sub_avg.values.tolist()
        }
    charts["amplitude_by_subcarrier"] = amp_by_sub

    rssi_by_packet = {}
    for ap, group in df.groupby("ap"):
        pkt_avg = group.groupby("packet")["rssi"].mean()
        rssi_by_packet[str(ap)] = {
            "packets": pkt_avg.index.tolist(),
            "rssi": pkt_avg.values.tolist()
        }
    charts["rssi_by_packet"] = rssi_by_packet

    phase_by_sub = {}
    for ap, group in df.groupby("ap"):
        sub_avg = group.groupby("subcarrier")["angle_rad"].mean()
        phase_by_sub[str(ap)] = {
            "subcarriers": sub_avg.index.tolist(),
            "phase": sub_avg.values.tolist()
        }
    charts["phase_by_subcarrier"] = phase_by_sub

    for ap, group in df.groupby("ap"):
        pivot = group.pivot_table(
            index="subcarrier", columns="packet",
            values="amplitude", aggfunc="mean"
        )
        charts.setdefault("amplitude_heatmap", {})[str(ap)] = {
            "subcarriers": pivot.index.tolist(),
            "packets": pivot.columns.tolist(),
            "values": pivot.fillna(0).values.tolist()
        }

    charts["stats"] = {
        "total_rows": len(df),
        "aps": df["ap"].unique().tolist(),
        "subcarriers": int(df["subcarrier"].nunique()),
        "packets": int(df["packet"].nunique()),
        "avg_amplitude": round(float(df["amplitude"].mean()), 4),
        "avg_rssi": round(float(df["rssi"].mean()), 2),
    }
    return charts


@app.route("/")
def index():
    docs = get_docs()
    grouped = group_by_timestamp(docs)
    sort_order = request.args.get("sort", "desc")
    timestamps = sorted(grouped.keys(), reverse=(sort_order == "desc"))

    ts_meta = {}
    for ts in timestamps:
        group = grouped[ts]
        ap_set = set()
        sample_count = 0
        for d in group:
            ap_set.add(d.get("ap_index", "?"))
            sample_count += len(d.get("samples", []))
        ts_meta[ts] = {"aps": sorted(list(ap_set)), "samples": sample_count}

    return render_template("index.html",
                           timestamps=timestamps,
                           sort=sort_order,
                           ts_meta=ts_meta)


@app.route("/api/session/<path:timestamp>")
def api_session(timestamp):
    docs = get_docs()
    grouped = group_by_timestamp(docs)
    selected = grouped.get(timestamp, [])
    df = build_dataframe(selected)
    if df.empty:
        return jsonify({"error": "No data found for this session."})
    return jsonify(build_chart_data(df))


@app.route("/api/table/<path:timestamp>")
def api_table(timestamp):
    docs = get_docs()
    grouped = group_by_timestamp(docs)
    selected = grouped.get(timestamp, [])
    df = build_dataframe(selected)
    if df.empty:
        return jsonify({"error": "No data found.", "rows": [], "columns": []})
    return jsonify({
        "columns": df.columns.tolist(),
        "rows": df.fillna("").values.tolist()
    })


@app.route("/api/timestamps")
def api_timestamps():
    docs = get_docs()
    grouped = group_by_timestamp(docs)
    return jsonify({"timestamps": sorted(grouped.keys(), reverse=True)})


@app.route("/download/<path:timestamp>")
def download(timestamp):
    docs = get_docs()
    grouped = group_by_timestamp(docs)
    selected = grouped.get(timestamp, [])
    df = build_dataframe(selected)
    if df.empty:
        return "No valid data found for this timestamp."
    output = io.StringIO()
    df.to_csv(output, index=False)
    mem = io.BytesIO()
    mem.write(output.getvalue().encode())
    mem.seek(0)
    return send_file(mem, mimetype="text/csv", as_attachment=True,
                     download_name=f"{timestamp}_subcarrier.csv")


@app.route("/delete/<path:timestamp>", methods=["POST"])
def delete(timestamp):
    docs = get_docs()
    grouped = group_by_timestamp(docs)
    selected = grouped.get(timestamp, [])
    names = [d.get("name") for d in selected if "name" in d]
    delete_documents(names)
    _cache["docs"] = None
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
