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
    /* 1. Mengecilkan Font Global & Elemen Streamlit Bawaan */
    html, body, p, li, label, .streamlit-expanderHeader, .stMarkdown { 
        font-size: 0.9rem !important; 
    }
    
    /* 2. Styling Custom Metric Card (Lebih Compact) */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 12px 16px; /* Diperkecil agar hemat ruang vertikal */
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        margin-bottom: 12px;
        font-family: 'Inter', sans-serif;
    }
    .metric-title { color: #4b5563; font-size: 0.8rem; font-weight: 600; margin-bottom: 2px; line-height: 1.2; }
    .metric-unit { color: #9ca3af; font-size: 0.7rem; margin-bottom: 8px; line-height: 1; } /* Tambahan Keterangan Satuan */
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #111827; margin-bottom: 8px; line-height: 1; }
    .metric-delta { font-size: 0.75rem; font-weight: 600; padding: 2px 6px; border-radius: 4px; display: inline-block; }
    
    /* Warna Delta */
    .delta-positive { background-color: #dcfce7; color: #166534; }
    .delta-negative { background-color: #fee2e2; color: #991b1b; }
    .delta-neutral { background-color: #f3f4f6; color: #374151; }
    
    /* 3. Menyesuaikan Native Metric Streamlit di Halaman Simulasi Lokal */
    [data-testid="stMetricValue"] div { font-size: 1.5rem !important; }
    [data-testid="stMetricLabel"] p { font-size: 0.8rem !important; }

    /* Fix Sidebar Text Color */
    [data-testid="stSidebar"] { color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

# URL GeoJSON & Nama File CSV
URL_GEOJSON = "https://raw.githubusercontent.com/dwibagusd/Multilevel-Ordinal-Logistic-Regression-and-Spatial-Analysis-Food-Security/refs/heads/main/Data/peta_indonesia_comp.json"
CSV_FILENAME = "data_ringan.csv"
MATRIKS_FILENAME = "matriks_bobot_penuh.csv"

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

# Helper fungsi untuk mengambil center dan zoom
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

# Menyiapkan fungsi prediksi Prediksi PyMC yang akan dipakai di halaman Bayesian
def predict_ordinal_probs_pymc(df_input, w, df_asli):
    x_beta = 0
    for col, coef in w["beta"].items():
        if col in df_input.columns:
            mean_val = df_asli[col].mean()
            std_val = df_asli[col].std()
            if std_val == 0: std_val = 1e-9 
            nilai_z = (df_input[col] - mean_val) / std_val
            x_beta += nilai_z * coef
            
    mean_z = df_asli["anggaran_bansos"].mean()
    std_z = df_asli["anggaran_bansos"].std()
    if std_z == 0: std_z = 1e-9
    
    nilai_z_bansos = (df_input["anggaran_bansos"] - mean_z) / std_z
    gamma_z = nilai_z_bansos * w["gamma"]
    
    u_prov = df_input["provinsi"].map(w["u_provinsi"]).fillna(0.0)
    eta = x_beta + gamma_z + u_prov
    
    cutpoints = w["cutpoints"]
    prob_cat0 = 1 / (1 + np.exp(-(cutpoints[0] - eta)))
    prob_cat_0_1 = 1 / (1 + np.exp(-(cutpoints[1] - eta)))
    
    probs = np.column_stack([prob_cat0, prob_cat_0_1 - prob_cat0, 1.0 - prob_cat_0_1])
    return np.argmax(probs, axis=1)

# Inisialisasi Session State
kunci_slider_float = ["sim_bansos"]
kunci_slider_int = ["sim_ncpr", "sim_pengeluaran_pangan", "sim_kemiskinan", "sim_stunting", "sim_harapan_hidup", "sim_tanpa_listrik", "sim_tanpa_air_bersih", "sim_tenaga_kesehatan", "sim_lama_sekolah_perempuan"]
for key in kunci_slider_float:
    if key not in st.session_state: st.session_state[key] = 0.0
for key in kunci_slider_int:
    if key not in st.session_state: st.session_state[key] = 0
def reset_simulasi():
    for key in kunci_slider_float: st.session_state[key] = 0.0
    for key in kunci_slider_int: st.session_state[key] = 0

status_map = {0: "Rentan", 1: "Tahan", 2: "Sangat Tahan"}


# -----------------------------------------------------------------------------
# 3. FUNGSI HALAMAN 1: BAYESIAN MULTILEVEL
# -----------------------------------------------------------------------------
def halaman_bayesian():
    # ==========================================
    # SIDEBAR KHUSUS HALAMAN BAYESIAN
    # ==========================================
    st.sidebar.subheader(":material/filter_alt: Simulasi What-If")

    # Mengelompokkan Sliders
    with st.sidebar.expander("Level 1 (Kabupaten/Kota)", expanded=True):
        sim_ncpr = st.slider("NCPR (%)", -50, 50, 0, key="sim_ncpr", step=5)
        sim_pengeluaran_pangan = st.slider("Pengeluaran Pangan (%)", -50, 50, 0, key="sim_pengeluaran_pangan", step=5)
        sim_kemiskinan = st.slider("Kemiskinan (%)", -50, 50, 0, key="sim_kemiskinan", step=5)
        sim_stunting = st.slider("Stunting (%)", -50, 50, 0, key="sim_stunting", step=5)
        sim_harapan_hidup = st.slider("Harapan Hidup (%)", -10, 10, 0, key="sim_harapan_hidup", step=1)
        sim_tanpa_listrik = st.slider("Tanpa Listrik (%)", -50, 50, 0, key="sim_tanpa_listrik", step=5)
        sim_tanpa_air_bersih = st.slider("Tanpa Air Bersih (%)", -50, 50, 0, key="sim_tanpa_air_bersih", step=5)
        sim_tenaga_kesehatan = st.slider("Tenaga Kesehatan (%)", -50, 50, 0, key="sim_tenaga_kesehatan", step=5)
        sim_lama_sekolah_perempuan = st.slider("Lama Sekolah Perempuan (%)", -30, 30, 0, key="sim_lama_sekolah_perempuan", step=5)

    with st.sidebar.expander("Level 2 (Provinsi)", expanded=True):
        sim_bansos = st.slider("Bansos (Z-Score Absolute)", -2.0, 2.0, 0.0, key="sim_bansos", step=0.1)

    # Menerapkan rumus perubahan interaktif dengan perlindungan batas (clipping)
    df_sim = df_clean.copy()
    df_sim["ncpr"] = (df_clean["ncpr"] * (1 + sim_ncpr / 100)).clip(lower=0)
    df_sim["pengeluaran_pangan"] = (df_clean["pengeluaran_pangan"] * (1 + sim_pengeluaran_pangan / 100)).clip(0, 100)
    df_sim["kemiskinan"] = (df_clean["kemiskinan"] * (1 + sim_kemiskinan / 100)).clip(0, 100)
    df_sim["stunting"] = (df_clean["stunting"] * (1 + sim_stunting / 100)).clip(0, 100)
    df_sim["harapan_hidup"] = (df_clean["harapan_hidup"] * (1 + sim_harapan_hidup / 100)).clip(0, 100)
    df_sim["tanpa_listrik"] = (df_clean["tanpa_listrik"] * (1 + sim_tanpa_listrik / 100)).clip(0, 100)
    df_sim["tanpa_air_bersih"] = (df_clean["tanpa_air_bersih"] * (1 + sim_tanpa_air_bersih / 100)).clip(0, 100)
    df_sim["tenaga_kesehatan"] = (df_clean["tenaga_kesehatan"] * (1 + sim_tenaga_kesehatan / 100)).clip(lower=0)
    df_sim["lama_sekolah_perempuan"] = (df_clean["lama_sekolah_perempuan"] * (1 + sim_lama_sekolah_perempuan / 100)).clip(0, 18)
    df_sim["anggaran_bansos"] = df_clean["anggaran_bansos"] + sim_bansos

    st.sidebar.button("Reset", on_click=reset_simulasi, width='stretch')

    pred_awal = predict_ordinal_probs_pymc(df_clean, weights, df_clean)
    pred_sim = predict_ordinal_probs_pymc(df_sim, weights, df_clean)

    df_clean["predik_label"] = pred_awal
    df_sim["predik_label"] = pred_sim
    df_sim["status_ketahanan"] = df_sim["predik_label"].map(status_map)

    # Ukuran Font Header (h1, p) diperkecil via inline CSS
    st.markdown("<h1 style='font-size: 2.2rem; margin-bottom: 0;'>Multilevel Bayesian Regression</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem;'>Platform analisis interaktif berbasis pemodelan <b>Bayesian Multilevel</b> skala Nasional.</p>", unsafe_allow_html=True)

    # Menambahkan parameter unit_text
    def render_custom_metric(col, label, unit_text, var_name, is_inverse=False, is_absolute=False):
        if var_name not in df_clean.columns: return
        val_awal = df_clean[var_name].mean()
        val_sim = df_sim[var_name].mean()
        delta = val_sim - val_awal
        
        formatted_val = f"{val_sim:.2f}"
        if is_absolute:
            delta_str = f"{abs(delta):.2f} Poin"
        else:
            pct_change = (delta/val_awal)*100 if val_awal != 0 else 0
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

        # HTML Diperbarui dengan .metric-unit
        html_content = f"""
        <div class="metric-card">
            <div class="metric-title">{label}</div>
            <div class="metric-unit">{unit_text}</div>
            <div class="metric-value">{formatted_val}</div>
            <div class="metric-delta {delta_class}">{arrow} {delta_str}</div>
        </div>
        """
        col.markdown(html_content, unsafe_allow_html=True)

    # st.markdown("#### 📊 Indikator Utama (Rata-Rata Nasional)")
    # ==========================================
    # PEMBAGIAN LAYOUT (2 KOLOM KIRI : 3 KOLOM KANAN)
    # ==========================================
    col_kiri, col_kanan = st.columns([2, 3])
    
    with col_kiri:
        # st.markdown("#### 📊 Rata-Rata Nasional")
        
        # Grid 2 kolom di dalam kolom kiri agar metrik tertata rapi ke bawah
        sub_c1, sub_c2 = st.columns(2)
        
        # Variabel Sisi Kiri
        render_custom_metric(sub_c1, "NCPR", "Rasio Pangan/Kapita", "ncpr")
        render_custom_metric(sub_c1, "Kemiskinan", "% Penduduk", "kemiskinan", is_inverse=True)
        render_custom_metric(sub_c1, "Pengeluaran", "% Total Belanja", "pengeluaran_pangan")
        render_custom_metric(sub_c1, "Tanpa Listrik", "% Rumah Tangga", "tanpa_listrik", is_inverse=True)
        render_custom_metric(sub_c1, "Tanpa Air Bersih", "% Rumah Tangga", "tanpa_air_bersih", is_inverse=True)
        
        # Variabel Sisi Kanan
        render_custom_metric(sub_c2, "Lama Sekolah (Pr)", "Rata-rata Tahun", "lama_sekolah_perempuan")
        render_custom_metric(sub_c2, "Nakes", "Rasio Perkapita", "tenaga_kesehatan")
        render_custom_metric(sub_c2, "Harapan Hidup", "Usia (Tahun)", "harapan_hidup")
        render_custom_metric(sub_c2, "Stunting", "% Balita", "stunting", is_inverse=True)
        render_custom_metric(sub_c2, "Bansos", "Z-Score Absolut", "anggaran_bansos", is_absolute=True)

    with col_kanan:
        # st.markdown("#### 📈 Peta Visualisasi & Eksplorasi Spasial")
        
        # Filter berada tepat di atas peta
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            provinsi_terpilih = st.multiselect("Filter Provinsi:", options=sorted(df_sim["provinsi"].unique()), key="prov_bayes", placeholder="Pilih Provinsi...")
        with col_f2:
            label_terpilih = st.multiselect("Filter Status:", options=sorted(df_sim["status_ketahanan"].unique()), key="label_bayes", placeholder="Pilih Status...")
            
        df_filtered_bayes = df_sim.copy()
        if provinsi_terpilih: df_filtered_bayes = df_filtered_bayes[df_filtered_bayes["provinsi"].isin(provinsi_terpilih)]
        if label_terpilih: df_filtered_bayes = df_filtered_bayes[df_filtered_bayes["status_ketahanan"].isin(label_terpilih)]

        if df_filtered_bayes.empty:
            st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang Anda pilih.")
        else:
            # Peta dirender di kolom kanan
            center_koor, zoom_val = get_map_view(provinsi_terpilih)
            
            fig_map = px.choropleth_map(
                df_filtered_bayes, geojson=URL_GEOJSON, locations="kab_kota", featureidkey="properties.kab_kota", 
                color="status_ketahanan", color_discrete_map={"Rentan": "#ef4444", "Tahan": "#fde047", "Sangat Tahan": "#22c55e"},
                map_style="light", zoom=zoom_val, center=center_koor, opacity=0.8,
                hover_name="kab_kota", hover_data=["provinsi", "kemiskinan"] if "kemiskinan" in df_filtered_bayes.columns else ["provinsi"],
                height=520 
            )
            fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_map, use_container_width=True)     
            
        # Info Box Dampak Kebijakan diletakkan persis di bawah peta
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

        st.info(f"""
        💡 **Dampak Kebijakan Nasional:**
        * {pesan_rentan}
        * Proporsi wilayah berstatus **'Sangat Tahan'** berubah dari **{(tahan_awal/total_wilayah)*100:.1f}%** menjadi **{(tahan_sim/total_wilayah)*100:.1f}%**.
        """)
# -----------------------------------------------------------------------------
# 4. FUNGSI HALAMAN BARU: SIMULASI LOKAL SPESIFIK (COMPACT UI)
# -----------------------------------------------------------------------------
def halaman_simulasi_lokal():
    # Menghitung prediksi dasar (baseline) untuk semua wilayah
    pred_awal_global = predict_ordinal_probs_pymc(df_clean, weights, df_clean)
    df_base = df_clean.copy()
    df_base["status_ketahanan"] = pd.Series(pred_awal_global).map(status_map)

    # Dictionary Variabel
    dict_variabel = {
        "ncpr": "NCPR",
        "kemiskinan": "Kemiskinan (%)",
        "pengeluaran_pangan": "Pengeluaran Pangan (%)",
        "tanpa_listrik": "Tanpa Listrik (%)",
        "tanpa_air_bersih": "Tanpa Air Bersih (%)",
        "lama_sekolah_perempuan": "Lama Sekolah Perempuan",
        "tenaga_kesehatan": "Tenaga Kesehatan",
        "harapan_hidup": "Harapan Hidup",
        "stunting": "Stunting (%)",
        "anggaran_bansos": "Anggaran Bansos (Z-Score)"
    }

    # ==========================================
    # 1. KONTROL SIDEBAR (FILTER & INPUT)
    # ==========================================
    st.sidebar.markdown("### 🎯 Simulasi Lokal Spesifik")
    
    # Filter dalam Filter
    filter_status = st.sidebar.selectbox(
        "🔍 Filter Status Awal:", 
        options=["Semua Status", "Rentan", "Tahan", "Sangat Tahan"]
    )
    
    if filter_status != "Semua Status":
        pilihan_kab = sorted(df_base[df_base["status_ketahanan"] == filter_status]["kab_kota"].unique())
    else:
        pilihan_kab = sorted(df_base["kab_kota"].unique())
        
    if not pilihan_kab:
        st.sidebar.warning(f"Tidak ada wilayah dengan status {filter_status}.")
        st.stop()
        
    selected_kab = st.sidebar.selectbox("📍 Pilih Kabupaten/Kota:", options=pilihan_kab)
    prov_terpilih = df_base[df_base["kab_kota"] == selected_kab]["provinsi"].values[0]
    idx_kab = df_base[df_base["kab_kota"] == selected_kab].index[0]

    # Inisialisasi Session State khusus untuk simulasi lokal agar nilai form dinamis
    if 'kab_aktif' not in st.session_state:
        st.session_state.kab_aktif = None

    # Fungsi Reset Nilai
    def reset_nilai_ke_awal():
        for col in dict_variabel.keys():
            st.session_state[f"sim_{col}"] = float(df_base.loc[idx_kab, col])

    # Jika user mengganti kabupaten, otomatis reset nilai di form ke nilai baseline kabupaten baru
    if st.session_state.kab_aktif != selected_kab:
        st.session_state.kab_aktif = selected_kab
        reset_nilai_ke_awal()

    st.sidebar.write("---")
    
    # Tombol Reset Manual
    if st.sidebar.button("🔄 Reset ke Nilai Awal", use_container_width=True):
        reset_nilai_ke_awal()

    # Form Simulasi (Batch Input)
    with st.sidebar.form("form_lokal"):
        st.caption("Ubah angka di bawah ini, lalu klik Simpan untuk memproses.")
        
        # Looping untuk membuat seluruh input box tanpa harus memilih variabel dulu
        for col, label in dict_variabel.items():
            # Pastikan key ada di session state untuk menghindari error
            if f"sim_{col}" not in st.session_state:
                st.session_state[f"sim_{col}"] = float(df_base.loc[idx_kab, col])
                
            st.number_input(label, key=f"sim_{col}", step=1.0)
            
        submit_simpan = st.form_submit_button("💾 Simpan & Lihat Dampak", use_container_width=True)

    # ==========================================
    # 2. PROSES DATA LOKAL (Berdasarkan Form)
    # ==========================================
    df_sim_local = df_base.copy()
    
    # Terapkan nilai dari form ke dalam dataframe simulasi
    for col in dict_variabel.keys():
        df_sim_local.loc[idx_kab, col] = st.session_state[f"sim_{col}"]
        
    pred_sim_local = predict_ordinal_probs_pymc(df_sim_local, weights, df_clean)
    
    status_awal_str = df_base.loc[idx_kab, "status_ketahanan"]
    status_baru_str = status_map[pred_sim_local[idx_kab]]
    df_sim_local["status_ketahanan"] = pd.Series(pred_sim_local).map(status_map)

    # ==========================================
    # 3. KONTEN UTAMA (PETA DI ATAS)
    # ==========================================
    st.markdown("<h1 style='font-size: 2.2rem; margin-bottom: 0;'>Simulasi Kebijakan Spesifik</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem;'>Fokus Peta: <b>Provinsi {prov_terpilih}</b> (Daerah Target: <b>{selected_kab}</b>)</p>", unsafe_allow_html=True)

    # PETA BERADA DI PALING ATAS
    df_map_local = df_sim_local[df_sim_local["provinsi"] == prov_terpilih]
    center_koor, zoom_val = get_map_view([prov_terpilih])
    fig_map_local = px.choropleth_map(
        df_map_local, geojson=URL_GEOJSON, locations="kab_kota", featureidkey="properties.kab_kota", 
        color="status_ketahanan", color_discrete_map={"Rentan": "#ef4444", "Tahan": "#fde047", "Sangat Tahan": "#22c55e"},
        map_style="light", zoom=zoom_val, center=center_koor, opacity=0.9, 
        hover_name="kab_kota", hover_data={"status_ketahanan": True}, height=420
    )
    fig_map_local.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_map_local, use_container_width=True)

    # ==========================================
    # 4. KETERANGAN DAMPAK & METRIK BAWAH
    # ==========================================
    st.markdown("### 📊 Ringkasan Dampak Intervensi")
    
    # Alert Status Perubahan
    idx_awal = pred_awal_global[idx_kab]
    idx_baru = pred_sim_local[idx_kab]
    
    if idx_baru > idx_awal: 
        st.success(f"🎉 **Dampak Positif!** Perubahan variabel pada form simulasi berhasil meningkatkan status **{selected_kab}** dari **{status_awal_str}** menjadi **{status_baru_str}**.")
    elif idx_baru < idx_awal: 
        st.error(f"⚠️ **Waspada!** Perubahan variabel menyebabkan penurunan ketahanan pangan di **{selected_kab}** dari **{status_awal_str}** menjadi **{status_baru_str}**.")
    else:
        st.info(f"💡 **Stagnan:** Nilai yang Anda terapkan belum cukup untuk mengubah status ketahanan pangan **{selected_kab}** (Tetap **{status_awal_str}**).")

    # st.write("---")
    st.markdown(f"**Detail Perubahan Variabel di {selected_kab}:**")
    
    # Membuat 2 Baris berisi 5 Kolom agar 10 Variabel tampil cantik
    col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
    col_v6, col_v7, col_v8, col_v9, col_v10 = st.columns(5)
    
    kolom_list = [col_v1, col_v2, col_v3, col_v4, col_v5, col_v6, col_v7, col_v8, col_v9, col_v10]
    
    # Menggambar metrik untuk setiap variabel
    for i, (col_key, label) in enumerate(dict_variabel.items()):
        val_awal = float(df_base.loc[idx_kab, col_key])
        val_sim = st.session_state[f"sim_{col_key}"]
        delta = val_sim - val_awal
        
        # Logika reverse color untuk indikator negatif (misal: Kemiskinan naik = merah/inverse)
        is_inverse = col_key in ["kemiskinan", "tanpa_listrik", "tanpa_air_bersih", "stunting"]
        
        kolom_list[i].metric(
            label=label, 
            value=f"{val_sim:.2f}", 
            delta=f"{delta:+.2f}" if delta != 0 else "0.00", 
            delta_color="inverse" if is_inverse else "normal"
        )

# -----------------------------------------------------------------------------
# 4. FUNGSI HALAMAN 2: SPATIAL AUTOCORRELATION
# -----------------------------------------------------------------------------
def halaman_spasial():
    # Menggunakan df_clean karena analisis spasial menggunakan IKP murni (tidak terpengaruh What-If slider)
    df_spasial = df_clean.copy()
    
    if 'ikp' not in df_spasial.columns:
        st.error("Variabel 'ikp' tidak ditemukan di dataset.")
        st.stop()
        
    y_spasial = df_spasial['ikp'].values
    moran = Moran(y_spasial, w_spasial)
    moran_loc = Moran_Local(y_spasial, w_spasial)

    # Identifikasi Kluster Spasial
    signifikan = moran_loc.p_sim < 0.05
    kuadran = moran_loc.q
    
    df_spasial['cluster_label'] = 'Tidak Signifikan (ns)'
    df_spasial.loc[signifikan & (kuadran == 1), 'cluster_label'] = 'HH (Hotspot)'
    df_spasial.loc[signifikan & (kuadran == 3), 'cluster_label'] = 'LL (Coldspot)'
    df_spasial.loc[signifikan & (kuadran == 4), 'cluster_label'] = 'HL (Outlier)'
    df_spasial.loc[signifikan & (kuadran == 2), 'cluster_label'] = 'LH (Outlier)'

    # ==========================================
    # SIDEBAR KHUSUS HALAMAN SPASIAL
    # ==========================================
    # Kita pindahkan filter spasial ke sidebar agar konten utama lebih bersih!
    st.sidebar.markdown("### :material/filter_alt: Filter Area Spasial")
    provinsi_terpilih_spasial = st.sidebar.multiselect("Pilih Provinsi:", options=sorted(df_spasial["provinsi"].unique()), key="prov_spasial", placeholder="Semua Provinsi")
    kluster_terpilih = st.sidebar.multiselect("Pilih Kluster LISA:", options=sorted(df_spasial["cluster_label"].unique()), key="kluster_spasial", placeholder="Semua Kluster")
    
    st.sidebar.write("---")
    st.sidebar.info("Gunakan filter di atas untuk mengisolasi titik Hotspot/Coldspot pada provinsi tertentu di peta utama.")

    # ==========================================
    # KONTEN UTAMA HALAMAN SPASIAL
    # ==========================================
    st.markdown("<h1 style='font-size: 2.2rem; margin-bottom: 0;'>Spatial Autocorrelation</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem;'>Eksplorasi autokorelasi menggunakan <b>Global & Local Moran's I (LISA)</b>.</p>", unsafe_allow_html=True)
    
    # Dibagi menjadi 4 kolom agar semua metrik sejajar di atas peta
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    html_moran = f"""
    <div class="metric-card">
        <div class="metric-title">Global Moran's I Index</div>
        <div class="metric-unit">Indeks Autokorelasi</div>
        <div class="metric-value">{moran.I:.4f}</div>
        <div class="metric-delta {'delta-positive' if moran.I > 0 else 'delta-negative'}">{'Korelasi Positif' if moran.I > 0 else 'Dispersi'}</div>
    </div>
    """
    col_m1.markdown(html_moran, unsafe_allow_html=True)
    
    html_pval = f"""
    <div class="metric-card">
        <div class="metric-title">P-Value Signifikansi</div>
        <div class="metric-unit">Uji Permutasi</div>
        <div class="metric-value">{moran.p_sim:.4f}</div>
        <div class="metric-delta {'delta-positive' if moran.p_sim < 0.05 else 'delta-neutral'}">{'Signifikan (<0.05)' if moran.p_sim < 0.05 else 'Tidak Signifikan'}</div>
    </div>
    """
    col_m2.markdown(html_pval, unsafe_allow_html=True)

    html_ei = f"""
    <div class="metric-card">
        <div class="metric-title">Expected Index</div>
        <div class="metric-unit">Nilai Harapan (Acak)</div>
        <div class="metric-value">{moran.EI:.4f}</div>
        <div class="metric-delta delta-neutral">Batas Nol Spasial</div>
    </div>
    """
    col_m3.markdown(html_ei, unsafe_allow_html=True)

    html_zscore = f"""
    <div class="metric-card">
        <div class="metric-title">Z-Score</div>
        <div class="metric-unit">Standar Deviasi</div>
        <div class="metric-value">{moran.z_sim:.4f}</div>
        <div class="metric-delta {'delta-positive' if abs(moran.z_sim) >= 1.96 else 'delta-neutral'}">{'Signifikan (> ±1.96)' if abs(moran.z_sim) >= 1.96 else 'Tidak Kuat'}</div>
    </div>
    """
    col_m4.markdown(html_zscore, unsafe_allow_html=True)
        
    st.markdown("### Peta Local Moran's I (LISA)")
    
    # Filter Data berdasarkan Sidebar
    df_filtered_spasial = df_spasial.copy()
    if provinsi_terpilih_spasial: df_filtered_spasial = df_filtered_spasial[df_filtered_spasial["provinsi"].isin(provinsi_terpilih_spasial)]
    if kluster_terpilih: df_filtered_spasial = df_filtered_spasial[df_filtered_spasial["cluster_label"].isin(kluster_terpilih)]

    warna_cluster = {
        'HH (Hotspot)': '#16a34a', 'LL (Coldspot)': '#dc2626',
        'HL (Outlier)': '#86efac', 'LH (Outlier)': '#fca5a5',
        'Tidak Signifikan (ns)': '#e5e7eb'
    }
    
    if df_filtered_spasial.empty:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang Anda pilih.")
    else:
        center_koor, zoom_val = get_map_view(provinsi_terpilih_spasial)
            
        fig_lisa = px.choropleth_map(
            df_filtered_spasial, geojson=URL_GEOJSON, locations="kab_kota", featureidkey="properties.kab_kota", 
            color="cluster_label", color_discrete_map=warna_cluster, map_style="light", zoom=zoom_val, 
            center=center_koor, opacity=0.9, hover_name="kab_kota", 
            hover_data={"ikp": True, "cluster_label": False}, height=550
        )
        fig_lisa.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_lisa, use_container_width=True)

    st.subheader("Raw Data")
    kolom_spasial = ["kab_kota", "provinsi", "cluster_label", "ikp"]
    st.dataframe(df_filtered_spasial[kolom_spasial], width='stretch', hide_index=True)


# -----------------------------------------------------------------------------
# 5. SETUP NAVIGATION & EKSEKUSI
# -----------------------------------------------------------------------------
# Mendefinisikan Pages & Navigation (Akan otomatis muncul di urutan paling atas sidebar)
page_1 = st.Page(halaman_bayesian, title="Model Bayesian (Global)", default=True)
page_3 = st.Page(halaman_simulasi_lokal, title="Simulasi Spesifik (Lokal)")
page_2 = st.Page(halaman_spasial, title="Analisis Spasial")

pg = st.navigation({
    "Menu Analisis Utama": [page_1, page_3, page_2]
})

# Eksekusi (Harus ditaruh di akhir file)
pg.run()
