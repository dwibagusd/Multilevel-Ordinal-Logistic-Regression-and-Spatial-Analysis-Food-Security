import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
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
    
    [data-testid="stSidebar"] { color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

URL_GEOJSON = "https://raw.githubusercontent.com/dwibagusd/Multilevel-Ordinal-Logistic-Regression-and-Spatial-Analysis-Food-Security/refs/heads/main/Data/peta_indonesia_comp.json"
CSV_FILENAME = "data_ringan.csv"
MATRIKS_FILENAME = "matriks_bobot_penuh.csv"

KOORDINAT_PROVINSI = {
    "aceh": {"lat": 4.6951, "lon": 96.7494, "zoom": 6}, "sumatera utara": {"lat": 2.1154, "lon": 99.5451, "zoom": 6},
    "sumatera barat": {"lat": -0.7399, "lon": 100.8000, "zoom": 6.5}, "riau": {"lat": 0.2933, "lon": 101.7068, "zoom": 6},
    "jambi": {"lat": -1.6116, "lon": 103.6150, "zoom": 6.5}, "sumatera selatan": {"lat": -3.3194, "lon": 104.1481, "zoom": 6.5},
    "bengkulu": {"lat": -3.5778, "lon": 102.3464, "zoom": 6.5}, "lampung": {"lat": -4.5586, "lon": 105.4068, "zoom": 6.5},
    "kepulauan bangka belitung": {"lat": -2.7411, "lon": 106.4406, "zoom": 6.5}, "kepulauan riau": {"lat": 3.9456, "lon": 108.1429, "zoom": 5.5},
    "dki jakarta": {"lat": -6.2088, "lon": 106.8456, "zoom": 9}, "jawa barat": {"lat": -6.9204, "lon": 107.6046, "zoom": 7},
    "jawa tengah": {"lat": -7.1510, "lon": 110.1403, "zoom": 7}, "di yogyakarta": {"lat": -7.7956, "lon": 110.3695, "zoom": 8.5},
    "jawa timur": {"lat": -7.5361, "lon": 112.2384, "zoom": 7}, "banten": {"lat": -6.4058, "lon": 106.0640, "zoom": 7.5},
    "bali": {"lat": -8.4095, "lon": 115.1889, "zoom": 8}, "nusa tenggara barat": {"lat": -8.6529, "lon": 117.3616, "zoom": 7},
    "nusa tenggara timur": {"lat": -8.6574, "lon": 121.0794, "zoom": 6}, "kalimantan barat": {"lat": -0.2787, "lon": 111.4753, "zoom": 5.5},
    "kalimantan tengah": {"lat": -1.6815, "lon": 113.3824, "zoom": 5.5}, "kalimantan selatan": {"lat": -3.0926, "lon": 115.2838, "zoom": 6},
    "kalimantan timur": {"lat": 0.5387, "lon": 116.4194, "zoom": 5.5}, "kalimantan utara": {"lat": 3.0731, "lon": 116.0414, "zoom": 5.5},
    "sulawesi utara": {"lat": 0.6247, "lon": 123.9750, "zoom": 6.5}, "sulawesi tengah": {"lat": -1.4300, "lon": 121.4456, "zoom": 5.5},
    "sulawesi selatan": {"lat": -4.1449, "lon": 120.1150, "zoom": 6}, "sulawesi tenggara": {"lat": -4.1449, "lon": 122.1746, "zoom": 6},
    "gorontalo": {"lat": 0.6999, "lon": 122.4467, "zoom": 7}, "sulawesi barat": {"lat": -2.8441, "lon": 119.2321, "zoom": 6.5},
    "maluku": {"lat": -3.2385, "lon": 130.1453, "zoom": 5.5}, "maluku utara": {"lat": 1.5701, "lon": 127.8088, "zoom": 5.5},
    "papua": {"lat": -4.2699, "lon": 138.0804, "zoom": 5}, "papua barat": {"lat": -1.3361, "lon": 133.1747, "zoom": 5.5},
    "papua selatan": {"lat": -7.7126, "lon": 139.0433, "zoom": 5.5}, "papua tengah": {"lat": -4.1610, "lon": 135.9189, "zoom": 5.5},
    "papua pegunungan": {"lat": -4.2541, "lon": 138.9959, "zoom": 5.5}, "papua barat daya": {"lat": -1.3361, "lon": 132.0, "zoom": 6}
}

# Helper fungsi untuk mengambil center dan zoom
def get_map_view(prov_list):
    if not prov_list or len(prov_list) != 1: return {"lat": -2.5, "lon": 118}, 4.2  
    prov_key = prov_list[0].lower().strip()
    return {"lat": KOORDINAT_PROVINSI.get(prov_key, {}).get("lat", -2.5), 
            "lon": KOORDINAT_PROVINSI.get(prov_key, {}).get("lon", 118)}, \
           KOORDINAT_PROVINSI.get(prov_key, {}).get("zoom", 5.5)

# -----------------------------------------------------------------------------
# 3. FUNGSI HELPER VISUALISASI PROBABILITAS (STACKED BAR CHART)
# -----------------------------------------------------------------------------
def plot_prob_stacked_bar(p_awal, p_sim):
    fig = go.Figure()
    y_labels = ['Awal', 'Simulasi']
    
    fig.add_trace(go.Bar(
        y=y_labels, x=[p_awal[0]*100, p_sim[0]*100], name='Rentan', orientation='h',
        marker=dict(color='#ef4444'), text=[f"{p_awal[0]*100:.1f}%", f"{p_sim[0]*100:.1f}%"], textposition='auto'
    ))
    fig.add_trace(go.Bar(
        y=y_labels, x=[p_awal[1]*100, p_sim[1]*100], name='Tahan', orientation='h',
        marker=dict(color='#eab308'), text=[f"{p_awal[1]*100:.1f}%", f"{p_sim[1]*100:.1f}%"], textposition='auto'
    ))
    fig.add_trace(go.Bar(
        y=y_labels, x=[p_awal[2]*100, p_sim[2]*100], name='Sangat Tahan', orientation='h',
        marker=dict(color='#22c55e'), text=[f"{p_awal[2]*100:.1f}%", f"{p_sim[2]*100:.1f}%"], textposition='auto'
    ))
    
    fig.update_layout(
        barmode='stack', height=160, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'family': "Inter, sans-serif", 'size': 12},
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 100]),
        yaxis=dict(showgrid=False, zeroline=False, title_standoff=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, title="")
    )
    return fig

# -----------------------------------------------------------------------------
# 4. LOAD DATA & FUNGSI MCMC PROBABILITAS (UPDATED FOR CAR MODEL)
# -----------------------------------------------------------------------------
@st.cache_data
def load_tabular_data(file_path): return pd.read_csv(file_path)

@st.cache_data
def load_spatial_weights(file_path):
    df_matriks = pd.read_csv(file_path, index_col=0)
    w = lps.weights.full2W(df_matriks.values, ids=df_matriks.index.tolist())
    w.transform = 'r' 
    return w

@st.cache_data
def load_pymc_weights(json_path):
    with open(json_path, "r") as f: return json.load(f)

try:
    df_clean = load_tabular_data(CSV_FILENAME)
    w_spasial = load_spatial_weights(MATRIKS_FILENAME)
    weights = load_pymc_weights("model_weights.json")
except FileNotFoundError as e:
    st.error(f"⚠️ File tidak ditemukan: {e.filename}")
    st.stop()

# MODIFIKASI: Menambahkan komponen Spasial CAR (phi) ke dalam rumus eta
def get_ordinal_probs_pymc(df_input, w, df_asli):
    # 1. Fixed Effects Level 1 (pm.math.dot(X_data, beta))
    x_beta = 0
    for col, coef in w["beta"].items():
        if col in df_input.columns:
            mean_val = df_asli[col].mean()
            std_val = df_asli[col].std()
            if std_val == 0: std_val = 1e-9 
            nilai_z = (df_input[col] - mean_val) / std_val
            x_beta += nilai_z * coef
            
    # 2. Fixed Effect Level 2 (gamma * Z_data)
    mean_z = df_asli["anggaran_bansos"].mean()
    std_z = df_asli["anggaran_bansos"].std()
    if std_z == 0: std_z = 1e-9
    
    nilai_z_bansos = (df_input["anggaran_bansos"] - mean_z) / std_z
    gamma_z = nilai_z_bansos * w["gamma"]
    
    # 3. Random Effect Level 2 - Provinsi (u[prov_idx_data])
    u_prov = df_input["provinsi"].map(w["u_provinsi"]).fillna(0.0)
    
    # 4. TERBARU: Random Effect Spasial CAR - Kabupaten/Kota (phi[kab_idx_data])
    phi_spasial = df_input["kab_kota"].map(w["phi_kabupaten"]).fillna(0.0)
    
    # 5. Prediktor Linear Gabungan Final (eta)
    eta = x_beta + gamma_z + u_prov + phi_spasial
    
    # 6. Transformasi ke Probabilitas Ordinal
    cutpoints = w["cutpoints"]
    prob_cat0 = 1 / (1 + np.exp(-(cutpoints[0] - eta)))
    prob_cat_0_1 = 1 / (1 + np.exp(-(cutpoints[1] - eta)))
    
    probs = np.column_stack([prob_cat0, prob_cat_0_1 - prob_cat0, 1.0 - prob_cat_0_1])
    return probs

status_map = {0: "Rentan", 1: "Tahan", 2: "Sangat Tahan"}

kunci_slider_float = ["sim_bansos"]
kunci_slider_int = ["sim_ncpr", "sim_pengeluaran_pangan", "sim_kemiskinan", "sim_stunting", "sim_harapan_hidup", "sim_tanpa_listrik", "sim_tanpa_air_bersih", "sim_tenaga_kesehatan", "sim_lama_sekolah_perempuan"]
for key in kunci_slider_float:
    if key not in st.session_state: st.session_state[key] = 0.0
for key in kunci_slider_int:
    if key not in st.session_state: st.session_state[key] = 0
def reset_simulasi():
    for key in kunci_slider_float: st.session_state[key] = 0.0
    for key in kunci_slider_int: st.session_state[key] = 0

# -----------------------------------------------------------------------------
# FUNGSI HALAMAN 1: BAYESIAN MULTILEVEL (GLOBAL/NASIONAL)
# -----------------------------------------------------------------------------
def halaman_bayesian():
    st.sidebar.markdown("### 🔍 Filter Eksplorasi Peta")
    provinsi_terpilih = st.sidebar.multiselect("Filter Provinsi:", options=sorted(df_clean["provinsi"].unique()), key="prov_bayes", placeholder="Semua Provinsi")
    label_terpilih = st.sidebar.multiselect("Filter Status:", options=["Rentan", "Tahan", "Sangat Tahan"], key="label_bayes", placeholder="Semua Status")
    
    st.sidebar.write("---")
    st.sidebar.markdown("### 🎛️ Simulasi What-If (Nasional)")
    st.sidebar.button("🔄 Reset Semua Simulasi", on_click=reset_simulasi, use_container_width=True)

    with st.sidebar.expander("Level 1 (Kabupaten/Kota)", expanded=True):
        sim_ncpr = st.sidebar.slider("NCPR (%)", -50, 50, 0, key="sim_ncpr", step=5)
        sim_pengeluaran_pangan = st.sidebar.slider("Pengeluaran Pangan (%)", -50, 50, 0, key="sim_pengeluaran_pangan", step=5)
        sim_kemiskinan = st.sidebar.slider("Kemiskinan (%)", -50, 50, 0, key="sim_kemiskinan", step=5)
        sim_stunting = st.sidebar.slider("Stunting (%)", -50, 50, 0, key="sim_stunting", step=5)
        sim_harapan_hidup = st.sidebar.slider("Harapan Hidup (%)", -10, 10, 0, key="sim_harapan_hidup", step=1)
        sim_tanpa_listrik = st.sidebar.slider("Tanpa Listrik (%)", -50, 50, 0, key="sim_tanpa_listrik", step=5)
        sim_tanpa_air_bersih = st.sidebar.slider("Tanpa Air Bersih (%)", -50, 50, 0, key="sim_tanpa_air_bersih", step=5)
        sim_tenaga_kesehatan = st.sidebar.slider("Tenaga Kesehatan (%)", -50, 50, 0, key="sim_tenaga_kesehatan", step=5)
        sim_lama_sekolah_perempuan = st.sidebar.slider("Lama Sekolah Perempuan (%)", -30, 30, 0, key="sim_lama_sekolah_perempuan", step=5)

    with st.sidebar.expander("Level 2 (Provinsi)", expanded=True):
        sim_bansos = st.sidebar.slider("Bansos (Z-Score Absolute)", -2.0, 2.0, 0.0, key="sim_bansos", step=0.1)

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

    probs_awal = get_ordinal_probs_pymc(df_clean, weights, df_clean)
    pred_awal = np.argmax(probs_awal, axis=1)
    
    probs_sim = get_ordinal_probs_pymc(df_sim, weights, df_clean)
    pred_sim = np.argmax(probs_sim, axis=1)

    df_clean["predik_label"] = pred_awal
    df_sim["predik_label"] = pred_sim
    df_sim["status_ketahanan"] = df_sim["predik_label"].map(status_map)

    def render_custom_metric(col, label, unit_text, var_name, is_inverse=False, is_absolute=False):
        if var_name not in df_clean.columns: return
        val_awal = df_clean[var_name].mean()
        val_sim = df_sim[var_name].mean()
        delta = val_sim - val_awal
        
        formatted_val = f"{val_sim:.2f}"
        if is_absolute: delta_str = f"{abs(delta):.2f} Poin"
        else: pct_change = (delta/val_awal)*100 if val_awal != 0 else 0; delta_str = f"{abs(pct_change):.1f}%"
        
        if delta > 0.001: arrow = "↑"; delta_class = "delta-negative" if is_inverse else "delta-positive"
        elif delta < -0.001: arrow = "↓"; delta_class = "delta-positive" if is_inverse else "delta-negative"
        else: arrow = "→"; delta_class = "delta-neutral"; delta_str = "0.0%" if not is_absolute else "0.00 Poin"

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
        sub_c1, sub_c2 = st.columns(2)
        render_custom_metric(sub_c1, "NCPR", "Pangan/Kapita", "ncpr")
        render_custom_metric(sub_c1, "Kemiskinan", "% Penduduk", "kemiskinan", is_inverse=True)
        render_custom_metric(sub_c1, "Pengeluaran", "% Belanja", "pengeluaran_pangan")
        render_custom_metric(sub_c1, "Tanpa Listrik", "% Rumah Tangga", "tanpa_listrik", is_inverse=True)
        render_custom_metric(sub_c1, "Tanpa Air", "% Rumah Tangga", "tanpa_air_bersih", is_inverse=True)
        
        render_custom_metric(sub_c2, "Lama Sekolah", "Rata-rata Tahun", "lama_sekolah_perempuan")
        render_custom_metric(sub_c2, "Nakes", "Rasio Perkapita", "tenaga_kesehatan")
        render_custom_metric(sub_c2, "Harapan Hidup", "Usia (Tahun)", "harapan_hidup")
        render_custom_metric(sub_c2, "Stunting", "% Balita", "stunting", is_inverse=True)
        render_custom_metric(sub_c2, "Bansos", "Z-Score", "anggaran_bansos", is_absolute=True)

    with col_kanan:
        df_filtered_bayes = df_sim.copy()
        if provinsi_terpilih: df_filtered_bayes = df_filtered_bayes[df_filtered_bayes["provinsi"].isin(provinsi_terpilih)]
        if label_terpilih: df_filtered_bayes = df_filtered_bayes[df_filtered_bayes["status_ketahanan"].isin(label_terpilih)]

        if df_filtered_bayes.empty:
            st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang Anda pilih.")
        else:
            center_koor, zoom_val = get_map_view(provinsi_terpilih)
            fig_map = px.choropleth_map(
                df_filtered_bayes, geojson=URL_GEOJSON, locations="kab_kota", featureidkey="properties.kab_kota", 
                color="status_ketahanan", color_discrete_map={"Rentan": "#ef4444", "Tahan": "#fde047", "Sangat Tahan": "#22c55e"},
                map_style="basic", zoom=zoom_val, center=center_koor, opacity=0.8,
                hover_name="kab_kota", hover_data=["provinsi", "kemiskinan"] if "kemiskinan" in df_filtered_bayes.columns else ["provinsi"],
                height=530 
            )
            fig_map.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, title=""),
                margin={"r":0,"t":35,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
        rentan_awal = (pred_awal == 0).sum()
        rentan_sim = (pred_sim == 0).sum()
        tahan_awal = (pred_awal == 2).sum()
        tahan_sim = (pred_sim == 2).sum()
        total_wilayah = len(df_clean)

    if rentan_sim < rentan_awal: pesan_rentan = f"📉 Berhasil **mengentaskan {rentan_awal - rentan_sim} daerah** dari zona Rentan."
    elif rentan_sim > rentan_awal: pesan_rentan = f"⚠️ Waspada! Terdapat **{rentan_sim - rentan_awal} daerah baru** jatuh ke zona Rentan."
    else: pesan_rentan = "➖ Tidak ada perubahan jumlah wilayah pada zona Rentan (Kondisi Stagnan)."

    st.info(f"💡 **Dampak Kebijakan Nasional:**\n* {pesan_rentan}\n* Proporsi wilayah berstatus **'Sangat Tahan'** berubah dari **{(tahan_awal/total_wilayah)*100:.1f}%** menjadi **{(tahan_sim/total_wilayah)*100:.1f}%**.")

# -----------------------------------------------------------------------------
# 4B. HALAMAN BARU: SIMULASI PROVINSI (LEVEL 2)
# -----------------------------------------------------------------------------
def halaman_simulasi_provinsi():
    probs_awal_global = get_ordinal_probs_pymc(df_clean, weights, df_clean)
    pred_awal_global = np.argmax(probs_awal_global, axis=1)
    
    df_base = df_clean.copy()
    df_base["status_ketahanan"] = pd.Series(pred_awal_global).map(status_map)

    prov_terpilih = st.sidebar.selectbox("Pilih Provinsi Target:", options=sorted(df_base["provinsi"].unique()))
    nilai_awal_bansos = float(df_base[df_base["provinsi"] == prov_terpilih]["anggaran_bansos"].iloc[0])
    
    with st.sidebar.form("form_provinsi"):
        st.markdown("#### Intervensi Anggaran Bansos Pangan")
        new_bansos = st.number_input("Anggaran Bansos", value=nilai_awal_bansos, step=0.1)
        submit_prov = st.form_submit_button("Enter", use_container_width=True)

    df_sim_prov = df_base.copy()
    df_sim_prov.loc[df_sim_prov["provinsi"] == prov_terpilih, "anggaran_bansos"] = new_bansos
    
    probs_sim_prov = get_ordinal_probs_pymc(df_sim_prov, weights, df_clean)
    pred_sim_prov = np.argmax(probs_sim_prov, axis=1)
    df_sim_prov["status_baru"] = pd.Series(pred_sim_prov).map(status_map)
    
    st.markdown("<h1 style='font-size: 2.2rem; margin-bottom: 0;'>Simulasi Provinsi (Efek Multilevel)</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem;'>Fokus Peta: <b>{prov_terpilih}</b>. Mengubah Level 2 akan memengaruhi probabilitas Level 1 (Kabupaten/Kota).</p>", unsafe_allow_html=True)

    df_prov_only = df_sim_prov[df_sim_prov["provinsi"] == prov_terpilih].copy()
    
    st.markdown("#### 🎲 Rata-rata Probabilitas Wilayah (Analisis Ketidakpastian)")
    st.info(f"Batang berwarna mewakili nilai rata-rata (*Mean*), dan garis hitam adalah rentang ketidakpastian (*94% HDI*) dari model Bayesian untuk seluruh wilayah di **{prov_terpilih}**.")
    
    avg_pa_m = p_awal_m[df_prov_only.index].mean(axis=0); avg_pa_l = p_awal_l[df_prov_only.index].mean(axis=0); avg_pa_u = p_awal_u[df_prov_only.index].mean(axis=0)
    avg_ps_m = p_sim_m[df_prov_only.index].mean(axis=0); avg_ps_l = p_sim_l[df_prov_only.index].mean(axis=0); avg_ps_u = p_sim_u[df_prov_only.index].mean(axis=0)
    
    st.plotly_chart(plot_prob_stacked_bar(avg_pa_m, avg_ps_m), use_container_width=True)
    st.write("---")

    col_kiri, col_kanan = st.columns([2, 8])
    total_kab = len(df_prov_only)
    kab_naik = 0; kab_turun = 0
    
    for idx, row in df_prov_only.iterrows():
        if pred_sim_prov[idx] > pred_awal_global[idx]: kab_naik += 1
        elif pred_sim_prov[idx] < pred_awal_global[idx]: kab_turun += 1
            
    with col_kiri:
        html_tot = f"""<div class="metric-card"><div class="metric-title">Total Wilayah Admin</div><div class="metric-unit">Dalam Provinsi</div>
        <div class="metric-value">{total_kab}</div><div class="metric-delta delta-neutral">Kabupaten / Kota</div></div>"""
        st.markdown(html_tot, unsafe_allow_html=True)
        
        html_naik = f"""<div class="metric-card"><div class="metric-title">Meningkat Status</div><div class="metric-unit">Dampak Positif Bansos</div>
        <div class="metric-value">{kab_naik}</div><div class="metric-delta {'delta-positive' if kab_naik > 0 else 'delta-neutral'}">{'↑ Naik Status' if kab_naik > 0 else '→ Stagnan'}</div></div>"""
        st.markdown(html_naik, unsafe_allow_html=True)
        
        html_turun = f"""<div class="metric-card"><div class="metric-title">Menurun Status</div><div class="metric-unit">Dampak Negatif Bansos</div>
        <div class="metric-value">{kab_turun}</div><div class="metric-delta {'delta-negative' if kab_turun > 0 else 'delta-neutral'}">{'↓ Turun Status' if kab_turun > 0 else '→ Stagnan'}</div></div>"""
        st.markdown(html_turun, unsafe_allow_html=True)
        
        st.write("---")
        st.markdown(f"**Anggaran Bansos:**\n* Awal: `{nilai_awal_bansos:.2f}`\n* Baru: `{new_bansos:.2f}`")
        
    with col_kanan:
        center_koor, zoom_val = get_map_view([prov_terpilih])
        fig_map_prov = px.choropleth_map(
            df_prov_only, geojson=URL_GEOJSON, locations="kab_kota", featureidkey="properties.kab_kota", 
            color="status_baru", color_discrete_map={"Rentan": "#ef4444", "Tahan": "#fde047", "Sangat Tahan": "#22c55e"},
            map_style="basic", zoom=zoom_val, center=center_koor, opacity=0.9, 
            hover_name="kab_kota", hover_data={"status_baru": True}, height=450
        )
        fig_map_prov.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, title=""),
                                   margin={"r":0,"t":35,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_map_prov, use_container_width=True)

    st.write("---")
    df_tabel = df_prov_only[["kab_kota", "status_ketahanan", "status_baru", "kemiskinan", "stunting", "pengeluaran_pangan"]].copy()
    df_tabel.rename(columns={"status_ketahanan": "Status Awal", "status_baru": "Status Baru (Efek Bansos)"}, inplace=True)
    st.dataframe(df_tabel, use_container_width=True, hide_index=True)
    

# -----------------------------------------------------------------------------
# 4C. FUNGSI HALAMAN 3: SIMULASI LOKAL SPESIFIK (KAB/KOTA)
# -----------------------------------------------------------------------------
def halaman_simulasi_lokal():
    p_awal_m, p_awal_l, p_awal_u = predict_ordinal_probs_pymc(df_clean, weights, df_clean)
    pred_awal_global = np.argmax(p_awal_m, axis=1)
    
    df_base = df_clean.copy()
    df_base["status_ketahanan"] = pd.Series(pred_awal_global).map(status_map)

    dict_variabel_lokal = {
        "ncpr": ("NCPR", "Pangan/Kapita"), "kemiskinan": ("Kemiskinan", "% Penduduk"), 
        "pengeluaran_pangan": ("Pengeluaran", "% Belanja"), "tanpa_listrik": ("Tanpa Listrik", "% Rumah Tangga"), 
        "tanpa_air_bersih": ("Tanpa Air", "% Rumah Tangga"), "lama_sekolah_perempuan": ("Lama Sekolah", "Rata-rata Tahun"), 
        "tenaga_kesehatan": ("Nakes", "Rasio Perkapita"), "harapan_hidup": ("Harapan Hidup", "Usia (Tahun)"), 
        "stunting": ("Stunting", "% Balita")
    }

    st.sidebar.markdown("### Simulasi What-if")
    filter_status = st.sidebar.selectbox("Filter Status Awal:", options=["Semua Status", "Rentan", "Tahan", "Sangat Tahan"])
    
    if filter_status != "Semua Status": pilihan_kab = sorted(df_base[df_base["status_ketahanan"] == filter_status]["kab_kota"].unique())
    else: pilihan_kab = sorted(df_base["kab_kota"].unique())
        
    if not pilihan_kab:
        st.sidebar.warning(f"Tidak ada wilayah dengan status {filter_status}.")
        st.stop()
        
    selected_kab = st.sidebar.selectbox("Pilih Kabupaten/Kota:", options=pilihan_kab)
    prov_terpilih = df_base[df_base["kab_kota"] == selected_kab]["provinsi"].values[0]
    idx_kab = df_base[df_base["kab_kota"] == selected_kab].index[0]

    if 'kab_aktif' not in st.session_state: st.session_state.kab_aktif = None

    def reset_nilai_ke_awal():
        for col in dict_variabel_lokal.keys(): st.session_state[f"sim_{col}"] = float(df_base.loc[idx_kab, col])

    if st.session_state.kab_aktif != selected_kab:
        st.session_state.kab_aktif = selected_kab
        reset_nilai_ke_awal()

    st.sidebar.write("---")
    if st.sidebar.button("Reset Nilai", use_container_width=True): reset_nilai_ke_awal()

    with st.sidebar.form("form_lokal"):
        st.markdown("#### Ubah Nilai")
        for col, (label, unit) in dict_variabel_lokal.items():
            if f"sim_{col}" not in st.session_state: st.session_state[f"sim_{col}"] = float(df_base.loc[idx_kab, col])
            st.number_input(f"{label} ({unit})", key=f"sim_{col}", step=1.0)
        submit_simpan = st.form_submit_button("Simpan", use_container_width=True)
    
    df_sim_local = df_base.copy()
    for col in dict_variabel_lokal.keys(): df_sim_local.loc[idx_kab, col] = st.session_state[f"sim_{col}"]
        
    p_sim_m, p_sim_l, p_sim_u = predict_ordinal_probs_pymc(df_sim_local, weights, df_clean)
    pred_sim_local = np.argmax(p_sim_m, axis=1)
    
    status_awal_str = df_base.loc[idx_kab, "status_ketahanan"]
    status_baru_str = status_map[pred_sim_local[idx_kab]]
    df_sim_local["status_ketahanan"] = pd.Series(pred_sim_local).map(status_map)

    st.markdown("<h1 style='font-size: 2.2rem; margin-bottom: 0;'>Simulasi Kab/Kota (Level 1)</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem;'>Fokus Peta: <b>Provinsi {prov_terpilih}</b> (Daerah Target: <b>{selected_kab}</b>)</p>", unsafe_allow_html=True)

    # ------------------ KOTAK PROBABILITAS (HDI ERROR BARS) ------------------
    st.markdown("#### 🎲 Probabilitas Transisi Kelas (Analisis Ketidakpastian)")
    st.info("Batang berwarna mewakili probabilitas prediksi, dan garis hitam adalah **rentang keyakinan (HDI)** model terhadap prediksi tersebut.")
    
    pa_m = p_awal_m[idx_kab];
    ps_m = p_sim_m[idx_kab];
    st.plotly_chart(plot_prob_stacked_bar(pa_m, ps_m), use_container_width=True)
    st.write("---")

    col_kiri, col_kanan = st.columns([3, 7])
    
    with col_kiri:
        sub_c1, sub_c2 = st.columns(2)
        var_items = list(dict_variabel_lokal.items())
        
        for i, (col_key, (label, unit)) in enumerate(var_items):
            target_col = sub_c1 if i % 2 == 0 else sub_c2
            
            val_awal = float(df_base.loc[idx_kab, col_key])
            val_sim = st.session_state[f"sim_{col_key}"]
            delta = val_sim - val_awal
            is_inverse = col_key in ["kemiskinan", "tanpa_listrik", "tanpa_air_bersih", "stunting"]
            
            formatted_val = f"{val_sim:.2f}"
            
            # --- MENGHITUNG PERSENTASE TANPA BANSOS ---
            pct_change = (delta/val_awal)*100 if val_awal != 0 else 0
            delta_str = f"{abs(pct_change):.1f}%"
                
            if delta > 0.001: arrow = "↑"; delta_class = "delta-negative" if is_inverse else "delta-positive"
            elif delta < -0.001: arrow = "↓"; delta_class = "delta-positive" if is_inverse else "delta-negative"
            else: arrow = "→"; delta_class = "delta-neutral"; delta_str = "0.0%" 
            
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
        df_map_local = df_sim_local[df_sim_local["provinsi"] == prov_terpilih]
        center_koor, zoom_val = get_map_view([prov_terpilih])
        
        fig_map_local = px.choropleth_map(
            df_map_local, geojson=URL_GEOJSON, locations="kab_kota", featureidkey="properties.kab_kota", 
            color="status_ketahanan", color_discrete_map={"Rentan": "#ef4444", "Tahan": "#fde047", "Sangat Tahan": "#22c55e"},
            map_style="light", zoom=zoom_val, center=center_koor, opacity=0.9, 
            hover_name="kab_kota", hover_data={"status_ketahanan": True}, height=530 
        )
        fig_map_local.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, title=""),
                margin={"r":0,"t":35,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_map_local, use_container_width=True)

    idx_awal = pred_awal_global[idx_kab]
    idx_baru = pred_sim_local[idx_kab]
    
    if idx_baru > idx_awal: st.success(f"🎉 **Dampak Positif!** Variabel simulasi berhasil meningkatkan status **{selected_kab}** dari **{status_awal_str}** menjadi **{status_baru_str}**.")
    elif idx_baru < idx_awal: st.error(f"⚠️ **Waspada!** Perubahan variabel menyebabkan penurunan ketahanan pangan di **{selected_kab}** dari **{status_awal_str}** menjadi **{status_baru_str}**.")
    else: st.info(f"💡 **Stagnan:** Nilai yang diterapkan belum merubah status ketahanan pangan **{selected_kab}** (Tetap **{status_awal_str}**).")

# -----------------------------------------------------------------------------
# 4. FUNGSI HALAMAN 2: SPATIAL AUTOCORRELATION
# -----------------------------------------------------------------------------
def halaman_spasial():
    df_spasial = df_clean.copy()
    if 'ikp' not in df_spasial.columns: st.error("Variabel 'ikp' tidak ditemukan di dataset."); st.stop()
        
    y_spasial = df_spasial['ikp'].values
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
    provinsi_terpilih_spasial = st.sidebar.multiselect("Pilih Provinsi:", options=sorted(df_spasial["provinsi"].unique()), key="prov_spasial", placeholder="Semua Provinsi")
    kluster_terpilih = st.sidebar.multiselect("Pilih Kluster LISA:", options=sorted(df_spasial["cluster_label"].unique()), key="kluster_spasial", placeholder="Semua Kluster")
    st.sidebar.write("---")
    st.sidebar.info("Gunakan filter di atas untuk mengisolasi titik Hotspot/Coldspot pada provinsi tertentu di peta utama.")

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
                kesimpulan = "Clustered"; delta_kesimpulan = "Mengelompok"; warna_kesimpulan = "delta-positive"
            else:
                kesimpulan = "Dispersed"; delta_kesimpulan = "Menyebar"; warna_kesimpulan = "delta-negative"
        else:
            kesimpulan = "Random"; delta_kesimpulan = "Acak (Tidak Signifikan)"; warna_kesimpulan = "delta-neutral"
            
        html_kesimpulan = f"""
        <div class="metric-card"><div class="metric-title">Kesimpulan Pola</div><div class="metric-unit">Distribusi Spasial</div>
        <div class="metric-value">{kesimpulan}</div><div class="metric-delta {warna_kesimpulan}">{delta_kesimpulan}</div></div>
        """
        st.markdown(html_kesimpulan, unsafe_allow_html=True)

    with col_kanan:
        warna_cluster = {'HH (Hotspot)': '#16a34a', 'LL (Coldspot)': '#dc2626', 'HL (Outlier)': '#86efac', 'LH (Outlier)': '#fca5a5', 'Tidak Signifikan (ns)': '#e5e7eb'}
            center_koor, zoom_val = get_map_view(provinsi_terpilih_spasial)
            
            fig_lisa = px.choropleth_map(
                df_filtered_spasial, geojson=URL_GEOJSON, locations="kab_kota", featureidkey="properties.kab_kota", 
                color="cluster_label", color_discrete_map=warna_cluster, map_style="basic", zoom=zoom_val, 
                center=center_koor, opacity=0.9, hover_name="kab_kota", 
                hover_data={"ikp": True, "cluster_label": False}, 
                height=530 
            )
            fig_lisa.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, title=""),
                margin={"r":0,"t":35,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_lisa, use_container_width=True)

    st.write("---")
    st.subheader("📋 Raw Data Kluster Spasial")
    kolom_spasial = ["kab_kota", "provinsi", "cluster_label", "ikp"]
    st.dataframe(df_filtered_spasial[kolom_spasial], use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# 6. SETUP NAVIGATION & EKSEKUSI
# -----------------------------------------------------------------------------
page_1 = st.Page(halaman_bayesian, title="Model Bayesian", default=True)
page_3 = st.Page(halaman_simulasi_lokal, title="Simulasi Level 1 (Kabupaten/Kota)")
page_4 = st.Page(halaman_simulasi_provinsi, title="Simulasi Level 2 (Provinsi)")
page_2 = st.Page(halaman_spasial, title="Analisis Spasial")

pg = st.navigation({
    "Menu Analisis Utama": [page_1, page_4, page_3, page_2]
})

pg.run()
