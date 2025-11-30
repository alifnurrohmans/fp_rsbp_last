import requests
import json
import time

# URL endpoint Flask Anda
# Pastikan ini cocok dengan log 'Running on http://127.0.0.1:5000' di server Flask Anda
URL = "http://127.0.0.1:5000/predict" 

# Data simulasi: Semua jawaban diatur ke nilai maksimal (5 - Sangat Tertarik)
# Ini mensimulasikan pengguna yang sangat tertarik pada semua aspek cybersecurity.
test_answers = {}
for i in range(1, 21):
    test_answers[f"q{i}"] = 5
    
print(f"==========================================")
print(f"🚀 Memulai Simulasi ke {URL}")
print(f"   Data yang dikirim: 20 pertanyaan, semua skor 5 (Sangat Tertarik)")
print(f"==========================================")

start_time = time.time()

try:
    # Kirim permintaan POST
    response = requests.post(
        URL,
        json=test_answers,
        headers={"Content-Type": "application/json"},
        timeout=60 # Beri waktu hingga 60 detik untuk menunggu Gemini
    )

    # Cek status code
    if response.status_code == 200:
        data = response.json()
        
        print(f"\n✅ SUKSES! Status Code: 200 OK")
        print(f"   Waktu Respons: {time.time() - start_time:.2f} detik")
        print("------------------------------------------")
        
        # Tampilkan hasil utama (sesuai yang diharapkan frontend)
        print("🎯 TOP 3 REKOMENDASI KARIR:")
        for role, score in data.get('top_3_career_recommendation', []):
            print(f"   - {role}: {score:.3f}")
            
        print("\n🤖 RINGKASAN LEARNING PATH (dari Gemini):")
        lp = data.get('learning_path', {})
        print(f"   - Peran Utama: {lp.get('primary_role', 'N/A')}")
        print(f"   - Kenapa Cocok: {lp.get('why_suited', 'N/A')}")
        print(f"   - Skills Beginner: {lp.get('learning_path', {}).get('beginner', ['N/A'])}")

        print("\nℹ️ CEK DATA GRAPH NEO4J:")
        print(f"   - Nodes yang Dibuat Gemini: {len(lp.get('graph_nodes', []))} nodes")
        print(f"   - Edges yang Dibuat Gemini: {len(lp.get('graph_edges', []))} edges")
        
    elif response.status_code == 503:
        # Error yang paling mungkin (Gemini/Neo4j gagal)
        # Ambil pesan error dari JSON response jika ada
        error_detail = response.json().get('detail', response.json().get('error', 'Tidak ada detail error'))
        print(f"\n❌ GAGAL! Status Code: 503 Service Unavailable")
        print(f"   Detail Error: {error_detail}")
        print(f"   Tips: Cek log server Flask untuk '🚨 ERROR GEMINI CONNECTION' atau '🚨 ERROR NEO4J'.")

    else:
        # Error lainnya (misal 400 Bad Request atau 500 Internal Error)
        error_detail = response.json().get('error', 'Cek log server Flask.')
        print(f"\n❌ GAGAL! Status Code: {response.status_code}")
        print(f"   Detail Error: {error_detail}")

except requests.exceptions.ConnectionError:
    print(f"\n❌ GAGAL TOTAL: Tidak dapat terhubung ke server Flask di {URL}.")
    print("   Pastikan server Anda berjalan (python app.py) dan tidak ada blokir firewall.")
except requests.exceptions.Timeout:
    print(f"\n❌ GAGAL TOTAL: Permintaan habis waktu (Timeout).")
    print("   Server Flask Anda mungkin lambat merespons (kemungkinan besar menunggu Gemini).")
except Exception as e:
    print(f"\n❌ ERROR PARSING: Terjadi kesalahan saat memproses respons: {e}")

print("\n==========================================")