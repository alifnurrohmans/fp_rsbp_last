# ======================================================
# app.py – Production-ready ML + Gemini + Neo4j Pipeline
# ======================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import google.generativeai as genai
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import traceback
import os, json
import sys # Tambahkan untuk exit


# ------------------------------------------------------
# FLASK INIT
# ------------------------------------------------------
app = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_exception(e):
    print("\n===== INTERNAL SERVER ERROR =====")
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


# ------------------------------------------------------
# LOAD ML MODELS
# ------------------------------------------------------
# Pastikan file ini ada di direktori yang sama!
try:
    models = joblib.load("career_models.pkl")
    print("✅ Model ML berhasil dimuat.")
except FileNotFoundError:
    print("❌ ERROR: File 'career_models.pkl' tidak ditemukan. Server tidak dapat berjalan.")
    sys.exit(1)


label_cols = [
    "offensive", "blue_team", "malware", "forensics",
    "network", "cloud", "appsec", "threatintel", "grc"
]


# ------------------------------------------------------
# GEMINI CONFIGURATION
# ------------------------------------------------------
# Menggunakan API key yang Anda berikan
GENAI_KEY = os.getenv("GENAI_API_KEY") or "AIzaSyCIhvRt5xHmAzk3aUG3W275UzekSbyBCoQ"
genai.configure(api_key=GENAI_KEY)

# Tambahkan pengecekan konfigurasi LLM (walaupun hanya inisialisasi)
if not GENAI_KEY or GENAI_KEY == "AIzaSyCIhvRt5xHmAzk3aUG3W275UzekSbyBCoQ":
      print("⚠️ PERINGATAN: GENAI_API_KEY belum diatur atau masih menggunakan placeholder.")
      # Tidak perlu exit, biarkan error terjadi saat pemanggilan jika key invalid

# PERBAIKAN UTAMA: Ganti gemini-1.5-pro (yang tidak didukung di endpoint v1beta)
# menjadi gemini-2.5-flash (yang stabil dan didukung)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


# ------------------------------------------------------
# NEO4J CONFIGURATION
# ------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # Cek koneksi secara eksplisit
    driver.verify_connectivity()
    print(f"✅ Koneksi Neo4j berhasil: {NEO4J_URI} dengan user {NEO4J_USER}")
except ServiceUnavailable:
    print(f"❌ ERROR: Neo4j server tidak tersedia di {NEO4J_URI}. Pastikan server berjalan dan firewall terbuka.")
    sys.exit(1)
except AuthError:
    print("❌ ERROR: Autentikasi Neo4j gagal. Cek NEO4J_USER dan NEO4J_PASSWORD.")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: Gagal inisialisasi driver Neo4j: {e}")
    sys.exit(1)


# ------------------------------------------------------
# HELPERS – NEO4J
# ------------------------------------------------------
def neo4j_clear_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def neo4j_insert_graph(nodes, edges):
    with driver.session() as session:
        for n in nodes:
            session.run("MERGE (:Skill {name: $name})", {"name": n})
        for src, dst in edges:
            session.run("""
                MATCH (a:Skill {name: $src}), (b:Skill {name: $dst})
                MERGE (a)-[:LEADS_TO]->(b)
            """, {"src": src, "dst": dst})

# ------------------------------------------------------
# HELPER – LLM LEARNING PATH GENERATION (UPDATED)
# ------------------------------------------------------
def generate_learning_path(top3):
    best_role = top3[0][0]
    best_score = top3[0][1]

    prompt = f"""
Anda adalah konsultan karier, ahli kurikulum, dan perancang roadmap profesional di bidang Cybersecurity.

TUGAS UTAMA:
Buat roadmap pembelajaran teknis yang sangat rinci, spesifik, dan sepenuhnya logis berdasarkan PERAN berikut:

PERAN UTAMA: {best_role}
SKOR KECOCOKAN ML: {best_score:.3f}

====================================================
PANDUAN KHUSUS UNTUK GRAPH (PENTING)
====================================================
Anda WAJIB membuat graph pembelajaran yang sangat kompleks:

1. Minimal 40 edges. Ideal: 50–70 edges.
2. Setiap skill beginner harus:
   - punya 2–4 sub-skill tambahan yang dimasukkan ke graph_nodes
   - dihubungkan ke 1–2 skill intermediate sebagai prasyarat
3. Setiap skill intermediate harus:
   - punya 2–4 sub-skill
   - dihubungkan ke 1–2 skill advanced
4. Setiap skill advanced harus:
   - punya sub-skill atau turunan tingkat pakar
5. Urutan graph WAJIB mengikuti alur:
   Beginner → Intermediate → Advanced → Mastery/Expert (subskills)
6. Graph harus terasa seperti kurikulum nyata dengan layer:
   - Fundamental
   - Core Skills
   - Specialization
   - Professional Practice
   - Mastery

====================================================
PANDUAN OUTPUT
====================================================

ATURAN:
1. Output HARUS berupa JSON valid tanpa teks tambahan.
2. Semua skill di learning_path WAJIB masuk juga ke graph_nodes.
3. Graph_nodes harus mencakup:
   - semua skill beginner
   - semua skill intermediate
   - semua skill advanced
   - subskills tambahan untuk membangun kompleksitas graph
4. Graph_edges harus panjang, kompleks, dan berlapis (40–70 edges).
5. Tambahkan learning_resources (minimal 15 skill dengan link pembelajaran).
6. recommended_projects 3–5 item level beginner → advanced.

FORMAT OUTPUT WAJIB (VALID JSON):

{
  "primary_role": "string",
  "why_suited": "string",
  "learning_path": {
    "beginner": [...],
    "intermediate": [...],
    "advanced": [...]
  },
  "learning_resources": [
    {
      "skill": "string",
      "links": ["https://...", "https://..."]
    }
  ],
  "recommended_certifications": ["string", "string"],
  "recommended_projects": ["string", "string"],
  "graph_nodes": ["string", "string"],
  "graph_edges": [
    ["skill A", "skill B"],
    ["skill B", "skill C"]
  ]
}

PENTING:
- Tidak boleh ada backticks, markdown, atau teks luar JSON.
- Hanya kirim objek JSON valid.


"""

    # Tambahkan safety checks dan timeout untuk koneksi Gemini
    try:
        resp = gemini_model.generate_content(
            prompt, 
            request_options={'timeout': 30} # Tambahkan timeout 30 detik
        )
        raw = resp.text.strip()

    except Exception as e:
        # Jika gagal koneksi (API key invalid, network, dll)
        print(f"🚨 ERROR GEMINI CONNECTION: {e}")
        # Gunakan ConnectionError untuk status 503 di Flask
        raise ConnectionError(f"Koneksi ke Gemini Gagal (Cek API Key & Jaringan): {e}")


    # Extract JSON safely
    start = raw.find("{")
    end = raw.rfind("}") + 1

    if start == -1 or end == -1:
        print(f"🚨 ERROR JSON PARSING:\n{raw}")
        raise ValueError("Gemini output not valid JSON structure. Ditemukan: " + raw)

    parsed = json.loads(raw[start:end])
    return parsed


# ------------------------------------------------------
# ENDPOINT: PREDICT
# ------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # Cek apakah body request adalah JSON
    if not request.is_json:
        return jsonify({"error": "Request body harus berupa JSON"}), 400
        
    data = request.json

    # Must contain q1–q20
    try:
        # Pengecekan yang lebih ketat apakah semua q1-q20 ada
        answers = np.array([data[f"q{i}"] for i in range(1, 21)]).reshape(1, -1)
    except Exception:
        # Ini akan terpicu jika ada q_i yang hilang
        return jsonify({"error": "Input harus berisi jawaban q1 hingga q20"}), 400

    predictions = {}

    for label in label_cols:
        model = models[label]

        try:
            # Menggunakan try/except untuk mengatasi jika model tidak memiliki predict_proba
            prob = float(model.predict_proba(answers)[0][1])
        except AttributeError:
            prob = float(model.predict(answers)[0])
        except Exception as e:
            print(f"🚨 ERROR: Kegagalan prediksi untuk label {label}: {e}")
            raise e

        predictions[label] = prob

    # Sort top-3
    top3 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]

    # Generate LLM learning path
    try:
        roadmap = generate_learning_path(top3)
    except ConnectionError as e:
        # Khusus untuk error koneksi Gemini yang ditangkap di helper
        return jsonify({"error": "Gemini API Connection Failed", "detail": str(e)}), 503
    except Exception as e:
        # Untuk error JSON parsing atau lainnya dari LLM
        return jsonify({"error": "Gemini generation failed", "detail": str(e)}), 500

    # Insert Neo4j graph
    nodes = roadmap.get("graph_nodes", [])
    edges = roadmap.get("graph_edges", [])

    if not isinstance(nodes, list) or not isinstance(edges, list):
        return jsonify({"error": "LLM returned invalid graph structure (nodes/edges not list)"}), 500

    try:
        neo4j_clear_database()
        neo4j_insert_graph(nodes, edges)
    except Exception as e:
        print(f"🚨 ERROR NEO4J: Gagal menulis ke database: {e}")
        # Jika gagal di sini, kembalikan response dengan status 503 (Service Unavailable)
        return jsonify({"error": "Gagal menyimpan graph ke Neo4j (Cek koneksi/kredensial Neo4j)", "detail": str(e)}), 503

    # SUCCESS RESPONSE
    return jsonify({
        "probabilities": predictions,
        "top_3_career_recommendation": top3,
        "learning_path": roadmap
    })


# ------------------------------------------------------
# ENDPOINT: GET GRAPH
# ------------------------------------------------------
@app.route("/graph", methods=["GET"])
def get_graph():
    node_map = {}
    edges = []

    try:
        with driver.session() as session:
            # Nodes
            for r in session.run("MATCH (n:Skill) RETURN n.name AS name"):
                node_map[r["name"]] = {"id": r["name"], "label": r["name"]}

            # Edges
            for r in session.run("""
                MATCH (a:Skill)-[:LEADS_TO]->(b:Skill)
                RETURN a.name AS src, b.name AS dst
            """):
                edges.append({"source": r["src"], "target": r["dst"]})
    except Exception as e:
        print(f"🚨 ERROR NEO4J GET: Gagal membaca database: {e}")
        return jsonify({"error": "Gagal membaca graph dari Neo4j"}), 503

    return jsonify({"nodes": list(node_map.values()), "edges": edges})


# ------------------------------------------------------
# OPTIONAL: CLEAR GRAPH
# ------------------------------------------------------
@app.route("/clear_graph", methods=["POST"])
def clear_graph():
    try:
        neo4j_clear_database()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": f"Gagal clear graph: {e}"}), 500


# ------------------------------------------------------
# START SERVER
# ------------------------------------------------------
if __name__ == "__main__":
    print("\n===========================================")
    print(f"🚀 SERVER FLASK BERJALAN di http://127.0.0.1:5000")
    print("===========================================\n")
    app.run(port=5000, debug=True, use_reloader=False) # Matikan reloader untuk menghindari inisialisasi ganda