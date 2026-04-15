from flask import Flask, render_template, request, send_file, redirect
import requests
import pandas as pd
import io

app = Flask(__name__)

# ================== CONFIG ==================
PROJECT_ID = "csi-esp"
COLLECTION = "csi"

BASE_URL = f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}/databases/(default)/documents/{COLLECTION}"

# ================== FIRESTORE ==================

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
        # ✅ FIX: skip invalid docs
        if "fields" not in d:
            print("⚠️ Skipping invalid doc:", d)
            continue

        fields = {k: parse_value(v) for k, v in d["fields"].items()}

        parsed.append({
            "name": d.get("name", ""),
            **fields
        })

    return parsed


def group_by_timestamp(docs):
    groups = {}
    for d in docs:
        ts = d.get("session_timestamp", "unknown")
        groups.setdefault(ts, []).append(d)
    return groups


def build_dataframe(docs, sort_by="subcarrier"):
    rows = []

    for doc in docs:
        # ✅ FIX: skip docs without samples
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

    # ✅ Sorting
    if sort_by == "subcarrier":
        df = df.sort_values(by=["subcarrier", "packet"])
    elif sort_by == "packet":
        df = df.sort_values(by=["packet", "subcarrier"])

    return df


def delete_documents(doc_names):
    for name in doc_names:
        url = f"https://firestore.googleapis.com/v1/{name}"
        requests.delete(url)


# ================== ROUTES ==================

@app.route("/")
def index():
    raw = fetch_all_docs()
    docs = parse_docs(raw)
    grouped = group_by_timestamp(docs)

    sort_order = request.args.get("sort", "desc")

    timestamps = list(grouped.keys())
    timestamps.sort(reverse=(sort_order == "desc"))

    return render_template("index.html", timestamps=timestamps, sort=sort_order)


@app.route("/download/<timestamp>")
def download(timestamp):
    sort_by = request.args.get("sort_by", "subcarrier")

    raw = fetch_all_docs()
    docs = parse_docs(raw)
    grouped = group_by_timestamp(docs)

    selected = grouped.get(timestamp, [])
    df = build_dataframe(selected, sort_by=sort_by)

    if df.empty:
        return "No valid data found for this timestamp."

    output = io.StringIO()
    df.to_csv(output, index=False)

    mem = io.BytesIO()
    mem.write(output.getvalue().encode())
    mem.seek(0)

    return send_file(mem,
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name=f"{timestamp}_{sort_by}.csv")


@app.route("/delete/<timestamp>", methods=["POST"])
def delete(timestamp):
    raw = fetch_all_docs()
    docs = parse_docs(raw)
    grouped = group_by_timestamp(docs)

    selected = grouped.get(timestamp, [])
    names = [d.get("name") for d in selected if "name" in d]

    delete_documents(names)

    return redirect("/")


# ================== MAIN ==================

if __name__ == "__main__":
    app.run(debug=True)