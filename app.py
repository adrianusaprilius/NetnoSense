
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import io

st.set_page_config(
    page_title="NetnoSense",
    page_icon="📊",
    layout="wide"
)

st.title("📊 NetnoSense")
st.subheader("AI Sentiment & Netnography Research")
st.caption("Digital Marketing & Consumer Behavior Research")

MODEL = "w11wo/indonesian-roberta-base-sentiment-classifier"

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model=MODEL,
        device=-1
    )

st.sidebar.header("Research Settings")

project = st.sidebar.text_input(
    "Nama penelitian",
    "Food Vlogger Research"
)

source = st.sidebar.selectbox(
    "Sumber data",
    ["TikTok", "YouTube", "Instagram",
     "Google Reviews", "X/Twitter", "Lainnya"]
)

method = st.radio(
    "Metode input",
    ["Input Manual", "Upload Dataset"],
    horizontal=True
)

df = None

if method == "Input Manual":
    text = st.text_area(
        "Masukkan satu komentar per baris",
        height=220
    )
    comments = [
        x.strip() for x in text.splitlines()
        if x.strip()
    ]
    df = pd.DataFrame({"Komentar": comments})

else:
    file = st.file_uploader(
        "Upload file Excel atau CSV",
        type=["xlsx", "csv"]
    )

    if file:
        try:
            if file.name.endswith(".csv"):
                raw = pd.read_csv(file)
            else:
                raw = pd.read_excel(file)

            column = st.selectbox(
                "Pilih kolom komentar",
                raw.columns
            )

            df = raw[[column]].rename(
                columns={column: "Komentar"}
            )

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

clean_duplicates = st.checkbox(
    "Hapus komentar duplikat",
    value=False
)

if st.button("🚀 Analisis Sentimen", type="primary"):

    if df is None or df.empty:
        st.warning("Masukkan data terlebih dahulu.")

    else:
        df = df.dropna().copy()
        df["Komentar"] = df["Komentar"].astype(str).str.strip()
        df = df[df["Komentar"] != ""]

        if clean_duplicates:
            df = df.drop_duplicates()

        if df.empty:
            st.warning("Tidak ada komentar yang dapat dianalisis.")
            st.stop()

        df = df.reset_index(drop=True)

        try:
            with st.spinner("AI sedang menganalisis komentar..."):
                classifier = load_model()

                predictions = classifier(
                    df["Komentar"].tolist(),
                    truncation=True,
                    batch_size=8
                )

                label_map = {
                    "positive": "Positif",
                    "negative": "Negatif",
                    "neutral": "Netral"
                }

                df["Sentimen_AI"] = [
                    label_map.get(
                        p["label"].lower(), p["label"]
                    )
                    for p in predictions
                ]

                df["Skor_Model"] = [
                    round(p["score"], 4)
                    for p in predictions
                ]

                df.insert(
                    0, "ID",
                    [f"C{i+1:04d}" for i in range(len(df))]
                )

                df["Proyek"] = project
                df["Sumber"] = source
                df["Model"] = MODEL
                df["Validasi_Manual"] = "Belum divalidasi"

                st.session_state["results"] = df

        except Exception as e:
            st.error(f"Analisis gagal: {e}")

if "results" in st.session_state:

    result = st.session_state["results"]

    st.divider()
    st.header("Dashboard Hasil Analisis")

    total = len(result)
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Komentar", total)

    for col, label in zip(
        [c2, c3, c4],
        ["Positif", "Negatif", "Netral"]
    ):
        count = (
            result["Sentimen_AI"] == label
        ).sum()

        col.metric(
            label,
            f"{count / total * 100:.1f}%",
            f"{count} komentar",
            delta_color="off"
        )

    summary = result["Sentimen_AI"].value_counts().reset_index()
    summary.columns = ["Sentimen", "Jumlah"]

    fig = px.pie(
        summary,
        names="Sentimen",
        values="Jumlah",
        hole=0.45,
        color="Sentimen",
        color_discrete_map={
            "Positif": "#16a34a",
            "Negatif": "#dc2626",
            "Netral": "#94a3b8"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Validasi Hasil Penelitian")

    edited = st.data_editor(
        result,
        disabled=[
            c for c in result.columns
            if c != "Validasi_Manual"
        ],
        column_config={
            "Validasi_Manual": st.column_config.SelectboxColumn(
                "Validasi Peneliti",
                options=[
                    "Belum divalidasi",
                    "Positif",
                    "Negatif",
                    "Netral",
                    "Campuran",
                    "Tidak dapat ditentukan"
                ]
            )
        },
        use_container_width=True,
        hide_index=True
    )

    buffer = io.BytesIO()

    with pd.ExcelWriter(
        buffer, engine="openpyxl"
    ) as writer:
        edited.to_excel(
            writer,
            sheet_name="Dataset",
            index=False
        )

        summary.to_excel(
            writer,
            sheet_name="Ringkasan_AI",
            index=False
        )

    st.download_button(
        "📥 Download Hasil Excel",
        data=buffer.getvalue(),
        file_name="NetnoSense_Results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.caption(
        "Prediksi AI bersifat awal dan memerlukan "
        "validasi peneliti. Skor model bukan bukti "
        "akurasi ilmiah."
    )
