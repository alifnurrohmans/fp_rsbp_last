# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import google.generativeai as genai
from neo4j import GraphDatabase
import os, json

# -----------------------------------------
# FLASK INIT
# -----------------------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------------------
# LOAD ML MODELS
# -----------------------------------------
models = joblib.load("career_models.pkl")
label_cols = [
    "offensive", "blue_team", "malware", "forensics",
    "network", "cloud", "appsec", "threatintel", "grc"
]

# -----------------------------------------
# CONFIG GEMINI AI (ganti key)
# -----------------------------------------
GENAI_KEY = os.getenv("GENAI_API_KEY") or "ISI_API_KEY_GEMINI_DISINI"
genai.configure(api_key=GENAI_KEY)

# -----------------------------------------
# CONFIG NEO4J (ganti sesuai env)
# -----------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USER") or "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or "password"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -----------------------------------------
# Helper: Clear and Insert graph functions (as before)
# -----------------------------------------
def neo4j_clear_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def neo4j_insert_graph(nodes, edges):
    with driver.session() as session:
        for node in nodes:
            session.run("MERGE (:Skill {name: $name})", {"name": node})
        for src, dst in edges:
            session.run("""
                MATCH (a:Skill {name:$src}), (b:Skill {name:$dst})
                MERGE (a)-[:LEADS_TO]->(b)
            """, {"src": src, "dst": dst})

# -----------------------------------------
# LLM: generate learning path via Gemini
# -----------------------------------------
def generate_learning_path(top3):
    prompt = f"""
You are a professional cybersecurity career mentor.

Based on these top-3 predicted roles:

1. {top3[0][0]} (score: {top3[0][1]})
2. {top3[1][0]} (score: {top3[1][1]})
3. {top3[2][0]} (score: {top3[2][1]})

Please generate a structured cybersecurity learning roadmap in JSON with this format:

{{
  "primary_role": "...",
  "why_suited": "...",
  "learning_path": {{
      "beginner": ["one-line steps..."],
      "intermediate": ["..."],
      "advanced": ["..."]
  }},
  "recommended_certifications": ["..."],
  "recommended_projects": ["..."],
  "graph_nodes": ["Skill A", "Skill B", ...],
  "graph_edges": [
      ["Skill A", "Skill B"],
      ["Skill B", "Skill C"]
  ]
}}
Only output valid JSON. No explanation.
"""
    model = genai.GenerativeModel("gemini-1.5-pro")
    resp = model.generate_content(prompt)
    text = resp.text.strip()
    # try extract JSON object substring
    start = text.find('{')
    end = text.rfind('}') + 1
    if start == -1 or end == -1:
        raise ValueError("LLM did not return JSON")
    json_text = text[start:end]
    return json.loads(json_text)

# -----------------------------------------
# PREDICT endpoint (ML -> LLM -> Neo4j)
# -----------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    answers = np.array([data[f"q{i}"] for i in range(1, 21)]).reshape(1, -1)

    predictions = {}
    for label in label_cols:
        model = models[label]
        # handle models that may not support predict_proba
        try:
            prob = float(model.predict_proba(answers)[0][1])
        except Exception:
            prob = float(model.predict(answers)[0])
        predictions[label] = prob

    top3 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]

    # generate learning path via Gemini
    try:
        llm_output = generate_learning_path(top3)
    except Exception as e:
        return jsonify({"error":"LLM failed","detail":str(e)}), 500

    # write to Neo4j: clear then insert
    graph_nodes = llm_output.get("graph_nodes", [])
    graph_edges = llm_output.get("graph_edges", [])

    # guard: ensure nodes/edges are lists
    if not isinstance(graph_nodes, list) or not isinstance(graph_edges, list):
        return jsonify({"error":"LLM returned invalid graph_nodes/graph_edges"}), 500

    neo4j_clear_database()
    neo4j_insert_graph(graph_nodes, graph_edges)

    return jsonify({
        "probabilities": predictions,
        "top_3_career_recommendation": top3,
        "learning_path": llm_output
    })

# -----------------------------------------
# NEW: /graph endpoint -> return nodes & edges from Neo4j
# -----------------------------------------
@app.route("/graph", methods=["GET"])
def get_graph():
    """
    Returns JSON:
    {
      "nodes": [{"id":"Skill A","label":"Skill A"}, ...],
      "edges": [{"source":"Skill A","target":"Skill B"}, ...]
    }
    """
    nodes = {}
    edges = []
    with driver.session() as session:
        # get nodes
        res_nodes = session.run("MATCH (n:Skill) RETURN n.name as name")
        for r in res_nodes:
            name = r["name"]
            nodes[name] = {"id": name, "label": name}

        # get edges
        res_edges = session.run("MATCH (a:Skill)-[r:LEADS_TO]->(b:Skill) RETURN a.name as src, b.name as dst")
        for r in res_edges:
            src = r["src"]; dst = r["dst"]
            # Only include nodes that exist
            if src not in nodes:
                nodes[src] = {"id": src, "label": src}
            if dst not in nodes:
                nodes[dst] = {"id": dst, "label": dst}
            edges.append({"source": src, "target": dst})

    return jsonify({"nodes": list(nodes.values()), "edges": edges})

# -----------------------------------------
# Optional: endpoint to clear Neo4j (admin)
# -----------------------------------------
@app.route("/clear_graph", methods=["POST"])
def clear_graph():
    neo4j_clear_database()
    return jsonify({"ok": True})

# -----------------------------------------
# START
# -----------------------------------------
if __name__ == "__main__":
    print("Server running at http://127.0.0.1:5000")
    app.run(port=5000, debug=True)
