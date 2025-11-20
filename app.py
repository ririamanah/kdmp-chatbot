import pickle
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Baca file .env supaya OPENAI_API_KEY terbaca
load_dotenv()
client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Mapping nama file PDF → label modul yang lebih manusiawi
FILENAME_LABELS = {
    "mp_1_latar_belakang_dan_tujuan_koperasi_desa_kelurahan_merah_putih_v1.pdf":
        "MP 1 – Latar Belakang & Tujuan KDMP",
    "mp2_manajemen_keuangan_dan_pendanaan_komplet_edited.pdf":
        "MP 2 – Manajemen Keuangan & Pendanaan",
    "mp_3_manajemen_aset_dan_pengembangan_usaha_kdmp_kkmp.pdf":
        "MP 3 – Manajemen Aset & Pengembangan Usaha",
    "mp_4_monitoring_evaluasi_dan_dukungan_pemerintah_terhadap_koperasi_desa_kelurahan_merah_putih_.pdf":
        "MP 4 – Monitoring, Evaluasi & Dukungan Pemerintah",
    "mp_5_integritas,_risiko_dan_mitigasi_pengelolaan_kdmp_1009_2025_1.pdf":
        "MP 5 – Integritas, Risiko & Mitigasi Pengelolaan KDMP",
    # Tambahkan di sini kalau nanti ada PDF lain
}


def embed_query(query: str):
    """Membuat embedding untuk pertanyaan user."""
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return np.array(resp.data[0].embedding)


def search_similar_chunks(query, vectors, texts, metas, k=5):
    """Mencari k potongan teks yang paling mirip dengan pertanyaan."""
    if len(texts) == 0:
        return []

    q_vec = embed_query(query)

    vecs = vectors
    norms = np.linalg.norm(vecs, axis=1) * np.linalg.norm(q_vec)
    sims = np.dot(vecs, q_vec) / (norms + 1e-10)
    idxs = np.argsort(sims)[::-1][:k]

    results = []
    for i in idxs:
        results.append(
            {
                "text": texts[i],
                "meta": metas[i],
                "score": float(sims[i]),
            }
        )
    return results
    
def format_source(meta: dict) -> str:
    """Bikin teks sumber yang rapi: 'MP 3 – Manajemen Aset..., halaman 5'."""
    filename = meta.get("filename", "dokumen")
    page = meta.get("page", "?")
    label = FILENAME_LABELS.get(filename, filename)
    return f"{label}, halaman {page}"


def build_prompt(question, retrieved_chunks):
    """Menyusun prompt berisi konteks + pertanyaan user."""
    header = (
        "Berikut ini adalah beberapa kutipan dari modul-modul Koperasi Desa/Kelurahan "
        "Merah Putih (KDMP/KKMP):\n\n"
    )

    body = ""
    for idx, item in enumerate(retrieved_chunks, start=1):
        meta = item["meta"]
        filename = meta.get("filename", "dokumen")
        page = meta.get("page", "?")
        body += (
            f"[Bagian {idx} | Sumber: {filename}, halaman {page}]\n"
            f"{item['text']}\n\n"
        )

    instruksi = (
        "Gunakan hanya informasi dari kutipan di atas untuk menjawab pertanyaan pengguna.\n"
        "Jawab SELALU dalam bahasa Indonesia yang jelas, rapi, dan sistematis.\n"
        "Jika informasi yang diminta tidak ada dalam kutipan, katakan dengan jujur "
        "bahwa informasi tersebut tidak ditemukan dalam modul.\n\n"
        f"Pertanyaan pengguna: {question}"
    )

    return header + body + instruksi


def generate_answer(question, vectors, texts, metas):
    retrieved = search_similar_chunks(question, vectors, texts, metas, k=5)
    prompt = build_prompt(question, retrieved)

    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Kamu adalah asisten AI yang menjawab pertanyaan tentang "
                    "Koperasi Desa/Kelurahan Merah Putih (KDMP/KKMP) berdasarkan "
                    "modul pelatihan yang telah disediakan. Jawab selalu dalam "
                    "bahasa Indonesia."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    answer = completion.choices[0].message.content
    return answer, retrieved



@st.cache_resource(show_spinner="Memuat basis pengetahuan KDMP…")
def load_vector_store():
    """Memuat vector_store.pkl yang dibuat oleh ingest.py."""
    try:
        with open("vector_store.pkl", "rb") as f:
            data = pickle.load(f)
        vectors = data["vectors"]
        texts = data["texts"]
        metas = data["metas"]
        return vectors, texts, metas
    except FileNotFoundError:
        return None, None, None


def main():
    st.set_page_config(page_title="ChatKDMP", page_icon="📚")
    st.title("📚 ChatKDMP – Chatbot Modul Koperasi Merah Putih")

    st.write(
        "Aplikasi ini menjawab pertanyaan berdasarkan **modul KDMP/KKMP** "
        "yang sudah dimuat sebelumnya. Silakan ajukan pertanyaan dalam bahasa Indonesia."
    )

    vectors, texts, metas = load_vector_store()
    if vectors is None:
        st.error(
            "Basis pengetahuan belum ditemukan (`vector_store.pkl`). "
            "Jalankan terlebih dahulu: `python ingest.py`."
        )
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Tampilkan riwayat chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Tanyakan sesuatu tentang KDMP/KKMP…")

        if user_input:
        # Simpan pesan user
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Jawab
        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban di modul KDMP…"):
                answer, retrieved = generate_answer(user_input, vectors, texts, metas)
                st.markdown(answer)

                # Tampilkan sumber di bawah jawaban
                if retrieved:
                    unique_sources = []
                    seen = set()
                    for item in retrieved:
                        key = (
                            item["meta"].get("filename"),
                            item["meta"].get("page"),
                        )
                        if key not in seen:
                            seen.add(key)
                            unique_sources.append(format_source(item["meta"]))

                    if unique_sources:
                        st.info(
                            "**Sumber jawaban:**\n\n"
                            + "\n".join(f"- {src}" for src in unique_sources)
                        )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


if __name__ == "__main__":
    main()




