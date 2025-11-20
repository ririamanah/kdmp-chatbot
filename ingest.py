import os
import pickle

from dotenv import load_dotenv
from openai import OpenAI

import numpy as np
from pypdf import PdfReader

# ----------------- Konfigurasi -----------------
load_dotenv()
client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
DATA_DIR = "data"

# Daftar file PDF yang akan dibaca (sesuaikan dengan nama file di folder data)
PDF_FILES = [
    "mp_1_latar_belakang_dan_tujuan_koperasi_desa_kelurahan_merah_putih_v1.pdf",
    "mp2_manajemen_keuangan_dan_pendanaan_komplet_edited.pdf",
    "mp_3_manajemen_aset_dan_pengembangan_usaha_kdmp_kkmp.pdf",
    "mp_4_monitoring_evaluasi_dan_dukungan_pemerintah_terhadap_koperasi_desa_kelurahan_merah_putih_.pdf",
    "mp_5_integritas,_risiko_dan_mitigasi_pengelolaan_kdmp_1009_2025_1.pdf",
    # Kalau ada dokumen tambahan, tambahkan di sini, contoh:
    # "kap_e_learning_pengelolaan_keuangan_kdkmp_rev._1.pdf.pdf",
]


def chunk_text(text, chunk_size=800, overlap=200):
    """Memecah teks panjang menjadi potongan (chunk) per ~chunk_size kata."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        start += max(chunk_size - overlap, 1)
    return chunks


def embed_texts(texts):
    """Membuat embedding untuk list teks menggunakan OpenAI."""
    if not texts:
        return []
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [d.embedding for d in resp.data]


def main():
    all_chunks = []
    all_metas = []

    for filename in PDF_FILES:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"[WARNING] File tidak ditemukan: {path}")
            continue

        print(f"Membaca: {path}")
        reader = PdfReader(path)

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if not page_text:
                continue

            # pecah per halaman menjadi beberapa chunk
            chunks = chunk_text(page_text)
            for ch in chunks:
                all_chunks.append(ch)
                all_metas.append(
                    {
                        "filename": filename,
                        "page": page_num,
                    }
                )

    if not all_chunks:
        print("Tidak ada chunk yang dihasilkan. Periksa file PDF di folder data.")
        return

    print(f"Total chunk: {len(all_chunks)}")
    print("Membuat embedding...")
    vectors = np.array(embed_texts(all_chunks))

    data = {
        "vectors": vectors,
        "texts": all_chunks,
        "metas": all_metas,
    }

    with open("vector_store.pkl", "wb") as f:
        pickle.dump(data, f)

    print("Selesai! Disimpan di vector_store.pkl")


if __name__ == "__main__":
    main()
