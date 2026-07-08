import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import libpysal as lps
from esda.moran import Moran, Moran_Local
import warnings

# Mengabaikan warning dari pysal
warnings.filterwarnings("ignore", category=UserWarning, message="The weights matrix is not fully connected")

# -----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & STYLE (CSS INJECTION)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Ketahanan Pangan",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Mengoptimalkan Font agar muat di 3 Kolom */
    html, body, p, li, label, .streamlit-expanderHeader, .stMarkdown { font-size: 0.9rem !important; }

    .metric-card {
        background-color: #ffffff; border: 1px solid #e5e7eb; border-radius: 6px;
        padding: 10px 12px; /* Padding sedikit dirapatkan agar muat 3 kolom */
        box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 12px; font-family: 'Inter', sans-serif;
    }
    .metric-title { color: #4b5563; font-size: 0.75rem; font-weight: 600; margin-bottom: 2px; line-height: 1.2; }
    .metric-unit { color: #9ca3af; font-size: 0.65rem; margin-bottom: 6px; line-height: 1; }
    .metric-value { font-size: 1.35rem; font-weight: 700; color: #111827; margin-bottom: 6px; line-height: 1; }
    .metric-delta { font-size: 0.7rem; font-weight: 600; padding: 2px 6px; border-radius: 4px; display: inline-block; }

    .delta-positive { background-color: #dcfce7; color: #166534; }
    .delta-negative { background-color: #fee2e2; color: #991b1b; }
    .delta-neutral { background-color: #f3f4f6; color: #374151; }

    [data-testid="stMetricValue"] div { font-size: 1.4rem !important; }
    [data-testid="stMetricLabel"] p { font-size: 0.8rem !important; }
    [data-testid="stSidebar"] { color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

# URL GeoJSON & Nama File CSV
URL_GEOJSON = "https://raw.githubusercontent.com/dwibagusd/Multilevel-Ordinal-Logistic-Regression-and-Spatial-Analysis-Food-Security/refs/heads/main/Data/peta_indonesia_comp.json"
CSV_FILENAME = "data_ringan.csv"
MATRIKS_FILENAME = "matriks_bobot_penuh.csv"

# =============================================================================
# KONFIGURASI VARIABEL PREDIKTOR (SUMBER TUNGGAL / SINGLE SOURCE OF TRUTH)
# =============================================================================
# Sesuai KOLOM_PREDIKTOR pada model terbaru. Level 1 = kabupaten/kota,
# Level 2 = provinsi (BANSOS).
#
# PENTING - MOHON DIKONFIRMASI:
# Label, unit, dan arah (is_inverse) untuk NCPR, MISKIN, RLSP, TNPAIR,
# TNPLISTRIK, NAKES, AHH, dan STUNTING saya bawa dari versi dashboard
# sebelumnya (sudah konsisten dengan Bab 2-4 skripsi Anda).
# Untuk ENERGI, PROHE, CBPK, CVHARGA, POU, AMANPANGN, dan PPH saya
# menuliskan TEBAKAN TERBAIK berdasarkan istilah ketahanan pangan yang umum
# (mis. POU = Prevalence of Undernourishment, PPH = skor Pola Pangan
# Harapan, CVHARGA = koefisien variasi harga pangan). Silakan sunting
# langsung dictionary LEVEL1_VARS di bawah ini (kolom "label", "unit",
# "is_inverse") agar sesuai definisi variabel yang sebenarnya Anda pakai.
# clip: "pct" -> dibatasi 0-100, "years" -> dibatasi 0-18, "min0" -> dibatasi >= 0
LEVEL1_VARS = {
    "NCPR":       {"label": "NCPR",                          "unit": "Pangan/Kapita",       "is_inverse": False, "clip": "min0"},
    "ENERGI":     {"label": "Ketersediaan Energi",            "unit": "Kkal/Kapita/Hari",    "is_inverse": False, "clip": "min0"},   # TODO: konfirmasi
    "PROHE":      {"label": "PROHE",                          "unit": "Nilai",               "is_inverse": False, "clip": "min0"},   # TODO: konfirmasi definisi
    "CBPK":       {"label": "CBPK",                           "unit": "Nilai",               "is_inverse": False, "clip": "min0"},   # TODO: konfirmasi definisi
    "MISKIN":     {"label": "Kemiskinan",                     "unit": "% Penduduk",          "is_inverse": True,  "clip": "pct"},
    "CVHARGA":    {"label": "Volatilitas Harga (CV)",         "unit": "% Koef. Variasi",     "is_inverse": True,  "clip": "min0"},   # TODO: konfirmasi
    "POU":        {"label": "Prevalence of Undernourishment", "unit": "% Penduduk",          "is_inverse": True,  "clip": "pct"},    # TODO: konfirmasi
    "RLSP":       {"label": "Lama Sekolah Perempuan",         "unit": "Rata-rata Tahun",     "is_inverse": False, "clip": "years"},
    "TNPAIR":     {"label": "Tanpa Air Bersih",                "unit": "% Rumah Tangga",     "is_inverse": True,  "clip": "pct"},
    "TNPLISTRIK": {"label": "Tanpa Listrik",                  "unit": "% Rumah Tangga",      "is_inverse": True,  "clip": "pct"},
    "NAKES":      {"label": "Tenaga Kesehatan",               "unit": "Rasio Perkapita",     "is_inverse": False, "clip": "min0"},
    "AHH":        {"label": "Harapan Hidup",                  "unit": "Usia (Tahun)",        "is_inverse": False, "clip": "pct", "slider_range": (-10, 10)},
    "AMANPANGN":  {"label": "Keamanan Pangan",                "unit": "% Wilayah/RT",        "is_inverse": False, "clip": "pct"},    # TODO: konfirmasi
    "PPH":        {"label": "Pola Pangan Harapan (PPH)",      "unit": "Skor (0-100)",        "is_inverse": False, "clip": "pct"},    # TODO: konfirmasi
    "STUNTING":   {"label": "Stunting",                       "unit": "% Balita",            "is_inverse": True,  "clip": "pct"},
}
LEVEL2_VAR_KEY = "BANSOS"
LEVEL2_VAR_LABEL = "Bansos"
LEVEL2_VAR_UNIT = "Z-Score"

TARGET_KAB = "IKP"          # variabel respons/komposit tingkat kab/kota untuk Moran's I
KOL_KAB_KOTA = "kab_kota"    # id wilayah kab/kota pada data_ringan.csv (sesuaikan bila berbeda)
KOL_PROVINSI = "provinsi"    # id provinsi pada data_ringan.csv (sesuaikan bila berbeda)


def apply_clip(series, clip_type):
    """Menerapkan batas nilai (clipping) sesuai tipe variabel."""
    if clip_type == "pct":
        return series.clip(0, 100)
    if clip_type == "years":
        return series.clip(0, 18)
    return series.clip(lower=0)  # "min0"


KOORDINAT_PROVINSI = {
    "aceh": {"lat": 4.6951, "lon": 96.7494, "zoom": 6},
    "sumatera utara": {"lat": 2.1154, "lon": 99.5451, "zoom": 6},
    "sumatera barat": {"lat": -0.7399, "lon": 100.8000, "zoom": 6.5},
    "riau": {"lat": 0.2933, "lon": 101.7068, "zoom": 6},
    "jambi": {"lat": -1.6116, "lon": 103.6150, "zoom": 6.5},
    "sumatera selatan": {"lat": -3.3194, "lon": 104.1481, "zoom": 6.5},
    "bengkulu": {"lat": -3.5778, "lon": 102.3464, "zoom": 6.5},
    "lampung": {"lat": -4.5586, "lon": 105.4068, "zoom": 6.5},
    "kepulauan bangka belitung": {"lat": -2.7411, "lon": 106.4406, "zoom": 6.5},
    "kepulauan riau": {"lat": 3.9456, "lon": 108.1429, "zoom": 5.5},
    "dki jakarta": {"lat": -6.2088, "lon": 106.8456, "zoom": 9},
    "jawa barat": {"lat": -6.9204, "lon": 107.6046, "zoom": 7},
    "jawa tengah": {"lat": -7.1510, "lon": 110.1403, "zoom": 7},
    "di yogyakarta": {"lat": -7.7956, "lon": 110.3695, "zoom": 8.5},
    "jawa timur": {"lat": -7.5361, "lon": 112.2384, "zoom": 7},
    "banten": {"lat": -6.4058, "lon": 106.0640, "zoom": 7.5},
    "bali": {"lat": -8.4095, "lon": 115.1889, "zoom": 8},
    "nusa tenggara barat": {"lat": -8.6529, "lon": 117.3616, "zoom": 7},
    "nusa tenggara timur": {"lat": -8.6574, "lon": 121.0794, "zoom": 6},
    "kalimantan barat": {"lat": -0.2787, "lon": 111.4753, "zoom": 5.5},
    "kalimantan tengah": {"lat": -1.6815, "lon": 113.3824, "zoom": 5.5},
    "kalimantan selatan": {"lat": -3.0926, "lon": 115.2838, "zoom": 6},
    "kalimantan timur": {"lat": 0.5387, "lon": 116.4194, "zoom": 5.5},
    "kalimantan utara": {"lat": 3.0731, "lon": 116.0414, "zoom": 5.5},
    "sulawesi utara": {"lat": 0.6247, "lon": 123.9750, "zoom": 6.5},
    "sulawesi tengah": {"lat": -1.4300, "lon": 121.4456, "zoom": 5.5},
    "sulawesi selatan": {"lat": -4.1449, "lon": 120.1150, "zoom": 6},
    "sulawesi tenggara": {"lat": -4.1449, "lon": 122.1746, "zoom": 6},
    "gorontalo": {"lat": 0.6999, "lon": 122.4467, "zoom": 7},
    "sulawesi barat": {"lat": -2.8441, "lon": 119.2321, "zoom": 6.5},
    "maluku": {"lat": -3.2385, "lon": 130.1453, "zoom": 5.5},
    "maluku utara": {"lat": 1.5701, "lon": 127.8088, "zoom": 5.5},
    "papua": {"lat": -4.2699, "lon": 138.0804, "zoom": 5},
    "papua barat": {"lat": -1.3361, "lon": 133.1747, "zoom": 5.5},
    "papua selatan": {"lat": -7.7126, "lon": 139.0433, "zoom": 5.5},
    "papua tengah": {"lat": -4.1610, "lon": 135.9189, "zoom": 5.5},
    "papua pegunungan": {"lat": -4.2541, "lon": 138.9959, "zoom": 5.5},
    "papua barat daya": {"lat": -1.3361, "lon": 132.0, "zoom": 6}
}


def get_map_view(prov_list):
    if not prov_list or len(prov_list) != 1:
        return {"lat": -2.5, "lon": 118}, 4.2  # Default View Indonesia

    prov_key = prov_list[0].lower().strip()
    lat = KOORDINAT_PROVINSI.get(prov_key, {}).get("lat", -2.5)
    lon = KOORDINAT_PROVINSI.get(prov_key, {}).get("lon", 118)
    zoom = KOORDINAT_PROVINSI.get(prov_key, {}).get("zoom", 5.5)
    return {"lat": lat, "lon": lon}, zoom

# -----------------------------------------------------------------------------
# 2. LOAD DATA (CACHE)
# -----------------------------------------------------------------------------
@st.cache_data
def load_tabular_data(file_path):
    return pd.read_csv(file_path)


@st.cache_data
def load_spatial_weights(file_path):
    # Matriks yang disimpan pada matriks_bobot_penuh.csv diasumsikan berupa
    # W_sym: matriks Queen contiguity biner (0/1) & simetris, hasil dari
    # proses penyusunan bobot spasial untuk model CAR (lihat catatan di
    # notebook pemodelan). Untuk kebutuhan Moran's I pada dashboard ini,
    # matriks tersebut di-row-standardize (transform='r') sesuai pendekatan:
    #   w = lps.weights.Queen.from_dataframe(final_data); w.transform = 'r'
    df_matriks = pd.read_csv(file_path, index_col=0)
    w = lps.weights.full2W(df_matriks.values, ids=df_matriks.index.tolist())
    w.transform = 'r'
    return w


@st.cache_data
def load_pymc_weights(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


try:
    df_clean = load_tabular_data(CSV_FILENAME)
    w_spasial = load_spatial_weights(MATRIKS_FILENAME)
    weights = load_pymc_weights("model_weights.json")
except FileNotFoundError as e:
    st.error(f"⚠️ File tidak ditemukan: {e.filename}")
    st.stop()


# Menyiapkan fungsi prediksi PyMC yang akan dipakai di seluruh halaman
def predict_ordinal_probs_pymc(df_input, w, df_asli):
    x_beta = 0
    for col, coef in w["beta"].items():
        if col in df_input.columns:
            mean_val = df_asli[col].mean()
            std_val = df_asli[col].std()
            if std_val == 0:
                std_val = 1e-9
            nilai_z = (df_input[col] - mean_val) / std_val
            x_beta += nilai_z * coef

    mean_z = df_asli[LEVEL2_VAR_KEY].mean()
    std_z = df_asli[LEVEL2_VAR_KEY].std()
    if std_z == 0:
        std_z = 1e-9

    nilai_z_bansos = (df_input[LEVEL2_VAR_KEY] - mean_z) / std_z
    gamma_z = nilai_z_bansos * w["gamma"]

    u_prov = df_input[KOL_PROVINSI].map(w["u_provinsi"]).fillna(0.0)
    phi_kab = df_input[KOL_KAB_KOTA].map(w["phi_kabkota"]).fillna(0.0)

    eta = x_beta + gamma_z + u_prov + phi_kab

    cutpoints = w["cutpoints"]
    prob_cat0 = 1 / (1 + np.exp(-(cutpoints[0] - eta)))
    prob_cat_0_1 = 1 / (1 + np.exp(-(cutpoints[1] - eta)))

    probs = np.column_stack([prob_cat0, prob_cat_0_1 - prob_cat0, 1.0 - prob_cat_0_1])
    return np.argmax(probs, axis=1)


# Inisialisasi Session State (dibangun otomatis dari LEVEL1_VARS)
def reset_simulasi():
    for key in LEVEL1_VARS:
        st.session_state[f"sim_{key}"] = 0
    st.session_state[f"sim_{LEVEL2_VAR_KEY}"] = 0.0


if f"sim_{LEVEL2_VAR_KEY}" not in st.session_state:
    st.session_state[f"sim_{LEVEL2_VAR_KEY}"] = 0.0
for key in LEVEL1_VARS:
    if f"sim_{key}" not in st.session_state:
        st.session_state[f"sim_{key}"] = 0

status_map = {0: "Rentan", 1: "Tahan", 2: "Sangat Tahan"}


# -----------------------------------------------------------------------------
# 3. FUNGSI HALAMAN 1: BAYESIAN MULTILEVEL
# -----------------------------------------------------------------------------
def halaman_bayesian():
    st.sidebar.markdown("### Filter Eksplorasi Peta")
    provinsi_terpilih = st.sidebar.multiselect("Filter Provinsi:", options=sorted(df_clean[KOL_PROVINSI].unique()), key="prov_bayes", placeholder="Semua Provinsi")
    label_terpilih = st.sidebar.multiselect("Filter Status:", options=["Rentan", "Tahan", "Sangat Tahan"], key="label_bayes", placeholder="Semua Status")
    st.sidebar.subheader("Simulasi What-If")

    with st.sidebar.expander("Level 1 (Kabupaten/Kota)", expanded=True):
        for key, cfg in LEVEL1_VARS.items():
            lo, hi = cfg.get("slider_range", (-50, 50))
            step = 1 if (hi - lo) <= 20 else 5
            st.slider(f"{cfg['label']} (%)", lo, hi, 0, key=f"sim_{key}", step=step)

    with st.sidebar.expander("Level 2 (Provinsi)", expanded=True):
        sim_bansos = st.slider(f"{LEVEL2_VAR_LABEL} (Z-Score Absolute)", -2.0, 2.0, 0.0, key=f"sim_{LEVEL2_VAR_KEY}", step=0.1)

    # Menerapkan rumus perubahan interaktif dengan perlindungan batas (clipping)
    df_sim = df_clean.copy()
    for key, cfg in LEVEL1_VARS.items():
        pct = st.session_state[f"sim_{key}"]
        df_sim[key] = apply_clip(df_clean[key] * (1 + pct / 100), cfg["clip"])
    df_sim[LEVEL2_VAR_KEY] = df_clean[LEVEL2_VAR_KEY] + sim_bansos

    st.sidebar.button("Reset", on_click=reset_simulasi, width='stretch')

    pred_awal = predict_ordinal_probs_pymc(df_clean, weights, df_clean)
    pred_sim = predict_ordinal_probs_pymc(df_sim, weights, df_clean)

    df_clean["predik_label"] = pred_awal
    df_sim["predik_label"] = pred_sim
    df_sim["status_ketahanan"] = df_sim["predik_label"].map(status_map)

    def render_custom_metric(col, label, unit_text, var_name, is_inverse=False, is_absolute=False):
        if var_name not in df_clean.columns:
            return
        val_awal = df_clean[var_name].mean()
        val_sim = df_sim[var_name].mean()
        delta = val_sim - val_awal

        formatted_val = f"{val_sim:.2f}"
        if is_absolute:
            delta_str = f"{abs(delta):.2f} Poin"
        else:
            pct_change = (delta / val_awal) * 100 if val_awal != 0 else 0
            delta_str = f"{abs(pct_change):.1f}%"

        if delta > 0.001:
            arrow = "↑"
            delta_class = "delta-negative" if is_inverse else "delta-positive"
        elif delta < -0.001:
            arrow = "↓"
            delta_class = "delta-positive" if is_inverse else "delta-negative"
        else:
            arrow = "→"
            delta_class = "delta-neutral"
            delta_str = "0.0%" if not is_absolute else "0.00 Poin"

        html_content = f"""
        <div class="metric-card">
            <div class="metric-title">{label}</div>
            <div class="metric-unit">{unit_text}</div>
            <div class="metric-value">{formatted_val}</div>
            <div class="metric-delta {delta_class}">{arrow} {delta_str}</div>
        </div>
        """
        col.markdown(html_content, unsafe_allow_html=True)

    col_kiri, col_kanan = st.columns([3, 7])

    with col_kiri:
        # 3 sub-kolom karena jumlah variabel sekarang lebih banyak (15 + BANSOS)
        sub_c1, sub_c2, sub_c3 = st.columns(3)
        sub_cols = [sub_c1, sub_c2, sub_c3]

        for i, (key, cfg) in enumerate(LEVEL1_VARS.items()):
            render_custom_metric(sub_cols[i % 3], cfg["label"], cfg["unit"], key, is_inverse=cfg["is_inverse"])
        render_custom_metric(sub_cols[len(LEVEL1_VARS) % 3], LEVEL2_VAR_LABEL, LEVEL2_VAR_UNIT, LEVEL2_VAR_KEY, is_absolute=True)

    with col_kanan:
        df_filtered_bayes = df_sim.copy()
        if provinsi_terpilih:
            df_filtered_bayes = df_filtered_bayes[df_filtered_bayes[KOL_PROVINSI].isin(provinsi_terpilih)]
        if label_terpilih:
            df_filtered_bayes = df_filtered_bayes[df_filtered_bayes["status_ketahanan"].isin(label_terpilih)]

        if df_filtered_bayes.empty:
            st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang Anda pilih.")
        else:
            center_koor, zoom_val = get_map_view(provinsi_terpilih)
            fig_map = px.choropleth_map(
                df_filtered_bayes, geojson=URL_GEOJSON, locations=KOL_KAB_KOTA, featureidkey="properties.kab_kota",
                color="status_ketahanan", color_discrete_map={"Rentan": "#ef4444", "Tahan": "#fde047", "Sangat Tahan": "#22c55e"},
                map_style="basic", zoom=zoom_val, center=center_koor, opacity=0.8,
                hover_name=KOL_KAB_KOTA, hover_data=[KOL_PROVINSI, "MISKIN"] if "MISKIN" in df_filtered_bayes.columns else [KOL_PROVINSI],
                height=530
            )
            fig_map.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, title=""),
                margin={"r": 0, "t": 35, "l": 0, "b": 0}, paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_map, use_container_width=True)

        rentan_awal = (pred_awal == 0).sum()
        rentan_sim = (pred_sim == 0).sum()
        tahan_awal = (pred_awal == 2).sum()
        tahan_sim = (pred_sim == 2).sum()
        total_wilayah = len(df_clean)

    if rentan_sim < rentan_awal:
        pesan_rentan = f"📉 Berhasil **mengentaskan {rentan_awal - rentan_sim} daerah** dari zona Rentan."
    elif rentan_sim > rentan_awal:
        pesan_rentan = f"⚠️ Waspada! Terdapat **{rentan_sim - rentan_awal} daerah baru** jatuh ke zona Rentan."
    else:
        pesan_rentan = "➖ Tidak ada perubahan jumlah wilayah pada zona Rentan (Kondisi Stagnan)."

    st.info(f"💡 **Dampak Kebijakan Nasional:**\n* {pesan_rentan}\n* Proporsi wilayah berstatus **'Sangat Tahan'** berubah dari **{(tahan_awal/total_wilayah)*100:.1f}%** menjadi **{(tahan_sim/total_wilayah)*100:.1f}%**.")


# -----------------------------------------------------------------------------
# 4B. HALAMAN: SIMULASI PROVINSI (LEVEL 2)
# -----------------------------------------------------------------------------
def halaman_simulasi_provinsi():
    pred_awal_global = predict_ordinal_probs_pymc(df_clean, weights, df_clean)
    df_base = df_clean.copy()
    df_base["status_ketahanan"] = pd.Series(pred_awal_global).map(status_map)

    prov_terpilih = st.sidebar.selectbox("Pilih Provinsi Target:", options=sorted(df_base[KOL_PROVINSI].unique()))

    nilai_awal_bansos = float(df_base[df_base[KOL_PROVINSI] == prov_terpilih][LEVEL2_VAR_KEY].iloc[0])

    with st.sidebar.form("form_provinsi"):
        st.markdown("#### Intervensi Anggaran Bansos Pangan")
        new_bansos = st.number_input(
            "Anggaran Bansos",
            value=nilai_awal_bansos,
            step=0.1
        )
        submit_prov = st.form_submit_button("Enter", use_container_width=True)

    df_sim_prov = df_base.copy()
    df_sim_prov.loc[df_sim_prov[KOL_PROVINSI] == prov_terpilih, LEVEL2_VAR_KEY] = new_bansos

    pred_sim_prov = predict_ordinal_probs_pymc(df_sim_prov, weights, df_clean)
    df_sim_prov["status_baru"] = pd.Series(pred_sim_prov).map(status_map)

    col_kiri, col_kanan = st.columns([2, 8])

    df_prov_only = df_sim_prov[df_sim_prov[KOL_PROVINSI] == prov_terpilih].copy()
    total_kab = len(df_prov_only)

    kab_naik = 0
    kab_turun = 0

    for idx, row in df_prov_only.iterrows():
        if pred_sim_prov[idx] > pred_awal_global[idx]:
            kab_naik += 1
        elif pred_sim_prov[idx] < pred_awal_global[idx]:
            kab_turun += 1

    with col_kiri:
        html_tot = f"""
        <div class="metric-card">
            <div class="metric-title">Total Wilayah Admin</div>
            <div class="metric-unit">Dalam Provinsi</div>
            <div class="metric-value">{total_kab}</div>
            <div class="metric-delta delta-neutral">Kabupaten / Kota</div>
        </div>
        """
        st.markdown(html_tot, unsafe_allow_html=True)

        html_naik = f"""
        <div class="metric-card">
            <div class="metric-title">Meningkat Status</div>
            <div class="metric-unit">Dampak Positif Bansos</div>
            <div class="metric-value">{kab_naik}</div>
            <div class="metric-delta {'delta-positive' if kab_naik > 0 else 'delta-neutral'}">{'↑ Naik Status' if kab_naik > 0 else '→ Stagnan'}</div>
        </div>
        """
        st.markdown(html_naik, unsafe_allow_html=True)

        html_turun = f"""
        <div class="metric-card">
            <div class="metric-title">Menurun Status</div>
            <div class="metric-unit">Dampak Negatif Bansos</div>
            <div class="metric-value">{kab_turun}</div>
            <div class="metric-delta {'delta-negative' if kab_turun > 0 else 'delta-neutral'}">{'↓ Turun Status' if kab_turun > 0 else '→ Stagnan'}</div>
        </div>
        """
        st.markdown(html_turun, unsafe_allow_html=True)

        st.write("---")
        st.markdown(f"**Anggaran Bansos:**\n* Awal: `{nilai_awal_bansos:.2f}`\n* Baru: `{new_bansos:.2f}`")
        if new_bansos != nilai_awal_bansos:
            st.success("Cek tabel di bawah untuk melihat rincian Kab/Kota yang terdampak.")
    with col_kanan:
        center_koor, zoom_val = get_map_view([prov_terpilih])

        fig_map_prov = px.choropleth_map(
            df_prov_only, geojson=URL_GEOJSON, locations=KOL_KAB_KOTA, featureidkey="properties.kab_kota",
            color="status_baru", color_discrete_map={"Rentan": "#ef4444", "Tahan": "#fde047", "Sangat Tahan": "#22c55e"},
            map_style="basic", zoom=zoom_val, center=center_koor, opacity=0.9,
            hover_name=KOL_KAB_KOTA, hover_data={"status_baru": True}, height=450
        )
        fig_map_prov.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, title=""),
            margin={"r": 0, "t": 35, "l": 0, "b": 0}, paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_map_prov, use_container_width=True)

    st.write("---")
    kolom_tabel = [KOL_KAB_KOTA, "status_ketahanan", "status_baru", "MISKIN", "STUNTING", "PPH"]
    kolom_tabel = [c for c in kolom_tabel if c in df_prov_only.columns]
    df_tabel = df_prov_only[kolom_tabel].copy()
    df_tabel.rename(columns={"status_ketahanan": "Status Awal", "status_baru": "Status Baru (Efek Bansos)"}, inplace=True)
    st.dataframe(df_tabel, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# 4C. HALAMAN: SIMULASI LOKAL SPESIFIK (COMPACT UI)
# -----------------------------------------------------------------------------
def halaman_simulasi_lokal():
    pred_awal_global = predict_ordinal_probs_pymc(df_clean, weights, df_clean)
    df_base = df_clean.copy()
    df_base["status_ketahanan"] = pd.Series(pred_awal_global).map(status_map)

    # Dictionary variabel lokal dibangun otomatis dari LEVEL1_VARS (BANSOS tidak
    # termasuk karena itu variabel Level 2 / provinsi)
    dict_variabel_lokal = {key: (cfg["label"], cfg["unit"]) for key, cfg in LEVEL1_VARS.items()}

    st.sidebar.markdown("### Simulasi What-if")
    filter_status = st.sidebar.selectbox("Filter Status Awal:", options=["Semua Status", "Rentan", "Tahan", "Sangat Tahan"])

    if filter_status != "Semua Status":
        pilihan_kab = sorted(df_base[df_base["status_ketahanan"] == filter_status][KOL_KAB_KOTA].unique())
    else:
        pilihan_kab = sorted(df_base[KOL_KAB_KOTA].unique())

    if not pilihan_kab:
        st.sidebar.warning(f"Tidak ada wilayah dengan status {filter_status}.")
        st.stop()

    selected_kab = st.sidebar.selectbox("Pilih Kabupaten/Kota:", options=pilihan_kab)
    prov_terpilih = df_base[df_base[KOL_KAB_KOTA] == selected_kab][KOL_PROVINSI].values[0]
    idx_kab = df_base[df_base[KOL_KAB_KOTA] == selected_kab].index[0]

    if 'kab_aktif' not in st.session_state:
        st.session_state.kab_aktif = None

    def reset_nilai_ke_awal():
        for col in dict_variabel_lokal.keys():
            st.session_state[f"sim_{col}"] = float(df_base.loc[idx_kab, col])

    if st.session_state.kab_aktif != selected_kab:
        st.session_state.kab_aktif = selected_kab
        reset_nilai_ke_awal()

    st.sidebar.write("---")
    if st.sidebar.button("Reset Nilai", use_container_width=True):
        reset_nilai_ke_awal()

    with st.sidebar.form("form_lokal"):
        st.markdown("#### Ubah Nilai")
        for col, (label, unit) in dict_variabel_lokal.items():
            if f"sim_{col}" not in st.session_state:
                st.session_state[f"sim_{col}"] = float(df_base.loc[idx_kab, col])
            st.number_input(f"{label} ({unit})", key=f"sim_{col}", step=1.0)
        submit_simpan = st.form_submit_button("Simpan", use_container_width=True)

    df_sim_local = df_base.copy()
    for col in dict_variabel_lokal.keys():
        df_sim_local.loc[idx_kab, col] = st.session_state[f"sim_{col}"]

    pred_sim_local = predict_ordinal_probs_pymc(df_sim_local, weights, df_clean)

    status_awal_str = df_base.loc[idx_kab, "status_ketahanan"]
    status_baru_str = status_map[pred_sim_local[idx_kab]]
    df_sim_local["status_ketahanan"] = pd.Series(pred_sim_local).map(status_map)

    col_kiri, col_kanan = st.columns([3, 7])

    with col_kiri:
        sub_c1, sub_c2 = st.columns(2)

        var_items = list(dict_variabel_lokal.items())
        for i, (col_key, (label, unit)) in enumerate(var_items):
            target_col = sub_c1 if i % 2 == 0 else sub_c2

            val_awal = float(df_base.loc[idx_kab, col_key])
            val_sim = st.session_state[f"sim_{col_key}"]
            delta = val_sim - val_awal
            is_inverse = LEVEL1_VARS[col_key]["is_inverse"]

            formatted_val = f"{val_sim:.2f}"

            pct_change = (delta / val_awal) * 100 if val_awal != 0 else 0
            delta_str = f"{abs(pct_change):.1f}%"

            if delta > 0.001:
                arrow = "↑"
                delta_class = "delta-negative" if is_inverse else "delta-positive"
            elif delta < -0.001:
                arrow = "↓"
                delta_class = "delta-positive" if is_inverse else "delta-negative"
            else:
                arrow = "→"
                delta_class = "delta-neutral"
                delta_str = "0.0%"

            html_content = f"""
            <div class="metric-card" style="padding: 10px;">
                <div class="metric-title" style="font-size: 0.7rem;">{label}</div>
                <div class="metric-unit" style="font-size: 0.55rem; margin-bottom: 4px;">{unit}</div>
                <div class="metric-value" style="font-size: 1.25rem;">{formatted_val}</div>
                <div class="metric-delta {delta_class}">{arrow} {delta_str}</div>
            </div>
            """
            target_col.markdown(html_content, unsafe_allow_html=True)

    with col_kanan:
        df_map_local = df_sim_local[df_sim_local[KOL_PROVINSI] == prov_terpilih]

        center_koor, zoom_val = get_map_view([prov_terpilih])

        fig_map_local = px.choropleth_map(
            df_map_local, geojson=URL_GEOJSON, locations=KOL_KAB_KOTA, featureidkey="properties.kab_kota",
            color="status_ketahanan", color_discrete_map={"Rentan": "#ef4444", "Tahan": "#fde047", "Sangat Tahan": "#22c55e"},
            map_style="light", zoom=zoom_val, center=center_koor, opacity=0.9,
            hover_name=KOL_KAB_KOTA, hover_data={"status_ketahanan": True},
            height=530
        )
        fig_map_local.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, title=""),
            margin={"r": 0, "t": 35, "l": 0, "b": 0}, paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_map_local, use_container_width=True)

    idx_awal = pred_awal_global[idx_kab]
    idx_baru = pred_sim_local[idx_kab]

    if idx_baru > idx_awal:
        st.success(f"🎉 **Dampak Positif!** Variabel simulasi berhasil meningkatkan status **{selected_kab}** dari **{status_awal_str}** menjadi **{status_baru_str}**.")
    elif idx_baru < idx_awal:
        st.error(f"⚠️ **Waspada!** Perubahan variabel menyebabkan penurunan ketahanan pangan di **{selected_kab}** dari **{status_awal_str}** menjadi **{status_baru_str}**.")
    else:
        st.info(f"💡 **Stagnan:** Nilai yang diterapkan belum merubah status ketahanan pangan **{selected_kab}** (Tetap **{status_awal_str}**).")


# -----------------------------------------------------------------------------
# 4. FUNGSI HALAMAN 2: SPATIAL AUTOCORRELATION
# -----------------------------------------------------------------------------
def halaman_spasial():
    df_spasial = df_clean.copy()
    if TARGET_KAB not in df_spasial.columns:
        st.error(f"Variabel '{TARGET_KAB}' tidak ditemukan di dataset.")
        st.stop()

    y_spasial = df_spasial[TARGET_KAB].values
    moran = Moran(y_spasial, w_spasial)
    moran_loc = Moran_Local(y_spasial, w_spasial)

    signifikan = moran_loc.p_sim < 0.05
    kuadran = moran_loc.q

    df_spasial['cluster_label'] = 'Tidak Signifikan (ns)'
    df_spasial.loc[signifikan & (kuadran == 1), 'cluster_label'] = 'HH (Hotspot)'
    df_spasial.loc[signifikan & (kuadran == 3), 'cluster_label'] = 'LL (Coldspot)'
    df_spasial.loc[signifikan & (kuadran == 4), 'cluster_label'] = 'HL (Outlier)'
    df_spasial.loc[signifikan & (kuadran == 2), 'cluster_label'] = 'LH (Outlier)'

    st.sidebar.markdown("### 🔍 Filter Area Spasial")
    provinsi_terpilih_spasial = st.sidebar.multiselect("Pilih Provinsi:", options=sorted(df_spasial[KOL_PROVINSI].unique()), key="prov_spasial", placeholder="Semua Provinsi")
    kluster_terpilih = st.sidebar.multiselect("Pilih Kluster LISA:", options=sorted(df_spasial["cluster_label"].unique()), key="kluster_spasial", placeholder="Semua Kluster")
    st.sidebar.write("---")
    st.sidebar.info("Gunakan filter di atas untuk mengisolasi titik Hotspot/Coldspot pada provinsi tertentu di peta utama.")

    df_filtered_spasial = df_spasial.copy()
    if provinsi_terpilih_spasial:
        df_filtered_spasial = df_filtered_spasial[df_filtered_spasial[KOL_PROVINSI].isin(provinsi_terpilih_spasial)]
    if kluster_terpilih:
        df_filtered_spasial = df_filtered_spasial[df_filtered_spasial["cluster_label"].isin(kluster_terpilih)]

    col_kiri, col_kanan = st.columns([2, 8])

    with col_kiri:
        html_moran = f"""
        <div class="metric-card"><div class="metric-title">Global Moran's I Index</div><div class="metric-unit">Indeks Autokorelasi</div>
        <div class="metric-value">{moran.I:.4f}</div><div class="metric-delta {'delta-positive' if moran.I > 0 else 'delta-negative'}">{'Korelasi Positif' if moran.I > 0 else 'Dispersi'}</div></div>
        """
        st.markdown(html_moran, unsafe_allow_html=True)

        html_pval = f"""
        <div class="metric-card"><div class="metric-title">P-Value Signifikansi</div><div class="metric-unit">Uji Permutasi</div>
        <div class="metric-value">{moran.p_sim:.4f}</div><div class="metric-delta {'delta-positive' if moran.p_sim < 0.05 else 'delta-neutral'}">{'Signifikan (<0.05)' if moran.p_sim < 0.05 else 'Tidak Signifikan'}</div></div>
        """
        st.markdown(html_pval, unsafe_allow_html=True)

        html_ei = f"""
        <div class="metric-card"><div class="metric-title">Expected Index</div><div class="metric-unit">Nilai Harapan (Acak)</div>
        <div class="metric-value">{moran.EI:.4f}</div><div class="metric-delta delta-neutral">Batas Nol Spasial</div></div>
        """
        st.markdown(html_ei, unsafe_allow_html=True)

        html_zscore = f"""
        <div class="metric-card"><div class="metric-title">Z-Score</div><div class="metric-unit">Standar Deviasi</div>
        <div class="metric-value">{moran.z_sim:.4f}</div><div class="metric-delta {'delta-positive' if abs(moran.z_sim) >= 1.96 else 'delta-neutral'}">{'Signifikan (> ±1.96)' if abs(moran.z_sim) >= 1.96 else 'Tidak Kuat'}</div></div>
        """
        st.markdown(html_zscore, unsafe_allow_html=True)

        if moran.p_sim < 0.05:
            if moran.z_sim > 0:
                kesimpulan = "Clustered"
                delta_kesimpulan = "Mengelompok"
                warna_kesimpulan = "delta-positive"
            else:
                kesimpulan = "Dispersed"
                delta_kesimpulan = "Menyebar"
                warna_kesimpulan = "delta-negative"
        else:
            kesimpulan = "Random"
            delta_kesimpulan = "Acak (Tidak Signifikan)"
            warna_kesimpulan = "delta-neutral"

        html_kesimpulan = f"""
        <div class="metric-card"><div class="metric-title">Kesimpulan Pola</div><div class="metric-unit">Distribusi Spasial</div>
        <div class="metric-value">{kesimpulan}</div><div class="metric-delta {warna_kesimpulan}">{delta_kesimpulan}</div></div>
        """
        st.markdown(html_kesimpulan, unsafe_allow_html=True)

    with col_kanan:
        warna_cluster = {'HH (Hotspot)': '#16a34a', 'LL (Coldspot)': '#dc2626', 'HL (Outlier)': '#86efac', 'LH (Outlier)': '#fca5a5', 'Tidak Signifikan (ns)': '#e5e7eb'}

        if df_filtered_spasial.empty:
            st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang Anda pilih.")
        else:
            center_koor, zoom_val = get_map_view(provinsi_terpilih_spasial)

            fig_lisa = px.choropleth_map(
                df_filtered_spasial, geojson=URL_GEOJSON, locations=KOL_KAB_KOTA, featureidkey="properties.kab_kota",
                color="cluster_label", color_discrete_map=warna_cluster, map_style="basic", zoom=zoom_val,
                center=center_koor, opacity=0.9, hover_name=KOL_KAB_KOTA,
                hover_data={TARGET_KAB: True, "cluster_label": False},
                height=530
            )
            fig_lisa.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, title=""),
                margin={"r": 0, "t": 35, "l": 0, "b": 0}, paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_lisa, use_container_width=True)

    st.write("---")
    st.subheader("📋 Raw Data Kluster Spasial")
    kolom_spasial = [KOL_KAB_KOTA, KOL_PROVINSI, "cluster_label", TARGET_KAB]
    st.dataframe(df_filtered_spasial[kolom_spasial], use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# 5. SETUP NAVIGATION & EKSEKUSI
# -----------------------------------------------------------------------------
page_1 = st.Page(halaman_bayesian, title="Model Bayesian", default=True)
page_3 = st.Page(halaman_simulasi_lokal, title="Simulasi Level 1 (Kabupaten/Kota)")
page_4 = st.Page(halaman_simulasi_provinsi, title="Simulasi Level 2 (Provinsi)")
page_2 = st.Page(halaman_spasial, title="Analisis Spasial")

pg = st.navigation({
    "Menu Analisis Utama": [page_1, page_4, page_3, page_2]
})

pg.run()
