import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go
import seaborn as sns





# -----------------------------
# BASIC CONFIG
# -----------------------------
st.set_page_config(
    page_title="Olimpiada analitika paneli",
    layout="wide"
)


def giveLine():
    st.write("")
    st.write("")
    st.write("")
# -----------------------------
# STYLING FUNCTION FOR PLOTLY
# -----------------------------
def beautify(fig, title: str):
    fig.update_layout(
        title=title,
        # title_x=0.5,
        title_font_size=20,
        template="simple_white",
        margin=dict(l=30, r=30, t=60, b=40),
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.07)")
    return fig



def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_sankey(df, col_left, col_mid, col_right, title,
                left_prefix="Uni", mid_prefix="Ders", right_prefix="Iş"):

    # take only needed columns and fill missing
    data = df[[col_left, col_mid, col_right]].fillna("Maglumat ýok")

    # add role prefixes to keep nodes separate (no cycles)
    left_vals  = left_prefix  + " " + data[col_left].astype(str).str.strip()
    mid_vals   = mid_prefix   + " " + data[col_mid].astype(str).str.strip()
    right_vals = right_prefix + " " + data[col_right].astype(str).str.strip()

    # all unique labels
    labels = pd.unique(pd.concat([left_vals, mid_vals, right_vals], ignore_index=True))
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    roles = [lbl.split(" ", 1)[0] if " " in lbl else "" for lbl in labels]

    # nice color palette per role
    role_colors = {
        "Uni":    "#4C78A8",   
        "Ders":   "#F58518",   
        "Iş":     "#54A24B",   
        "Status": "#E45756",   
        "Ýurt":   "#9467BD",   
    }
    node_colors = [role_colors.get(r, "#9E9E9E") for r in roles]

    source, target, value, link_colors = [], [], [], []

    # left → middle
    g1 = pd.DataFrame({"L": left_vals, "M": mid_vals})
    g1 = g1.groupby(["L", "M"]).size().reset_index(name="count")
    for _, row in g1.iterrows():
        s = label_to_idx[row["L"]]
        t = label_to_idx[row["M"]]
        source.append(s)
        target.append(t)
        value.append(row["count"])
        link_colors.append(_hex_to_rgba(node_colors[t], 0.35))

    # middle → right
    g2 = pd.DataFrame({"M": mid_vals, "R": right_vals})
    g2 = g2.groupby(["M", "R"]).size().reset_index(name="count")
    for _, row in g2.iterrows():
        s = label_to_idx[row["M"]]
        t = label_to_idx[row["R"]]
        source.append(s)
        target.append(t)
        value.append(row["count"])
        link_colors.append(_hex_to_rgba(node_colors[t], 0.35))

    # show labels without prefixes
    clean_labels = [lbl.split(" ", 1)[1] if " " in lbl else lbl for lbl in labels]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18,
            thickness=18,
            line=dict(color="rgba(0,0,0,0.2)", width=0.5),
            label=clean_labels,
            color=node_colors,
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors,
        )
    )])

    fig.update_layout(
        title_text=title,
        font_size=13,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig



def count_heatmap(df, row_col, col_col, title, z_title="Talyp sany"):
    # pivot table: count of students
    pivot = pd.crosstab(df[row_col], df[col_col])

    fig = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(x=col_col, y=row_col, color=z_title),
    )

    fig.update_layout(
        title=title,
        xaxis_title=col_col,
        yaxis_title=row_col,
        margin=dict(l=60, r=20, t=60, b=40),
        coloraxis_colorbar=dict(title=z_title),
    )
    return fig


# -----------------------------
# READ CSV FROM FILE
# -----------------------------
CSV_PATH = "data1.csv"   # your file
df = pd.read_csv(CSV_PATH)

for col in ["Okuw mekdebi", "Dersi", "Okuwdan soňky iş ýeri", "Ýagdaýy (galan / giden)", "Işleýän ýurdy"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# -----------------------------
# DATA PREP
# -----------------------------
# numeric conversions
num_cols = [
    "Okuwa giren ýyly",
    "Tamamlanan ýyly",
    "Daşary ýurt saparlarynyň sany",
    "Takmynan çykdajy möçberi (USD/manat)",
]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# helper: status as binary for ML
df["left_binary"] = np.where(df["Ýagdaýy (galan / giden)"].str.strip().str.lower() == "giden", 1, 0)

# duration of study (if years exist)
if "Okuwa giren ýyly" in df.columns and "Tamamlanan ýyly" in df.columns:
    df["Okuw dowamlylygy"] = df["Tamamlanan ýyly"] - df["Okuwa giren ýyly"]

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filtrleme")

# Olympiad level
levels = sorted(df["Olimpiadanyň derejesi (halkara / sebitleýin)"].dropna().unique().tolist())
selected_levels = st.sidebar.multiselect("Olimpiadanyň derejesi", levels, default=levels)

# Subject (future proof)
subjects = sorted(df["Dersi"].dropna().unique().tolist())
selected_subjects = st.sidebar.multiselect("Dersi", subjects, default=subjects)

# School / university
schools = sorted(df["Okuw mekdebi"].dropna().unique().tolist())
selected_schools = st.sidebar.multiselect("Okuw mekdebi", schools, default=schools)

# Status galan / giden
status_list = sorted(df["Ýagdaýy (galan / giden)"].dropna().unique().tolist())
selected_status = st.sidebar.multiselect("Ýagdaýy", status_list, default=status_list)

# Entrance year range
if df["Okuwa giren ýyly"].notna().any():
    min_year = int(df["Okuwa giren ýyly"].min())
    max_year = int(df["Okuwa giren ýyly"].max())
    year_range = st.sidebar.slider(
        "Okuwa giren ýyllar",
        min_year,
        max_year,
        (min_year, max_year),
        step=1
    )
else:
    year_range = None

# Apply filters
f = df.copy()
f = f[f["Olimpiadanyň derejesi (halkara / sebitleýin)"].isin(selected_levels)]
f = f[f["Dersi"].isin(selected_subjects)]
f = f[f["Okuw mekdebi"].isin(selected_schools)]
f = f[f["Ýagdaýy (galan / giden)"].isin(selected_status)]

# if year_range:
#     f = f[(f["Okuwa giren ýyly"] >= year_range[0]) & (f["Okuwa giren ýyly"] <= year_range[1])]

st.title("Olimpiada gatnaşan talyplar — Analitika paneli")
st.caption(f"Filtrlenen maglumatlar boýunça jemi talyp: **{len(f)}**")

# -----------------------------
# TABS
# -----------------------------
# -----------------------------
# CUSTOM TAB STYLING
# -----------------------------
st.markdown("""
<style>

    /* Wrapper spacing around the entire tab bar */
    div[data-baseweb="tabs"] {
        margin-top: 30px !important;
        margin-bottom: 25px !important;
    }

    /* Each tab button */
    div[data-baseweb="tab"] {
        font-size: 18px !important;         /* make text bigger */
        padding: 14px 26px !important;      /* more spacing inside tabs */
        margin-right: 12px !important;      /* space between tabs */
        border-radius: 8px !important;      /* soft rounded hover */
        transition: all 0.2s ease-in-out;
    }

    /* Hover effect */
    div[data-baseweb="tab"]:hover {
        background-color: #f2f2f2 !important;
        cursor: pointer;
    }

    /* Selected (active) tab underline color & thickness */
    div[data-baseweb="tab"] [aria-selected="true"] {
        border-bottom: 3px solid #e63946 !important;  /* nice red underline */
    }

    /* Make tab text slightly thicker */
    div[data-baseweb="tab"] p {
        font-weight: 500 !important;
        margin-bottom: 0px !important;
    }

</style>
""", unsafe_allow_html=True)

tab_overview, tab_olymp, tab_travel, tab_unis, tab_brain, tab_advanced, tab_table = st.tabs(
    ["Umumy syn  ", "Olimpiada profili", "Saparlar & çykdajy", "Uniwersitetler & career",
     "Giden-Galan Analitika", "Klasterleme &  Çaklama", "Data"]
)

# -----------------------------
# TAB 1: OVERVIEW
# -----------------------------
with tab_overview:
    st.subheader("Umumy görkezijiler")

    total_students = len(f)
    total_exp = f["Takmynan çykdajy möçberi (USD/manat)"].sum()
    avg_exp = f["Takmynan çykdajy möçberi (USD/manat)"].mean()
    avg_trips = f["Daşary ýurt saparlarynyň sany"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jemi talyp", total_students)
    col2.metric("Ortaça daşary ýurt sapary", f"{avg_trips:.1f}" if not np.isnan(avg_trips) else "N/A")
    col3.metric("Jemi çykdajy", f"{total_exp:,.0f}")
    col4.metric("Ortaça çykdajy", f"{avg_exp:,.0f}" if not np.isnan(avg_exp) else "N/A")
    st.write("")
    st.write("")

    # by year
    if "Okuwa giren ýyly" in f.columns:
        year_counts = f["Okuwa giren ýyly"].value_counts().sort_index().reset_index()
        year_counts.columns = ["Okuwa giren ýyly", "Talyp sany"]
        fig_year = px.bar(year_counts, x="Okuwa giren ýyly", y="Talyp sany")
        fig_year = beautify(fig_year, "Okuwa giren ýyly boýunça talyplaryň sany")
        st.plotly_chart(fig_year, use_container_width=True)

    # subject distribution (future-proof)
    subj_counts = f["Dersi"].value_counts().reset_index()
    subj_counts.columns = ["Dersi", "Talyp sany"]
    fig_sub = px.bar(subj_counts, x="Dersi", y="Talyp sany")
    fig_sub = beautify(fig_sub, "Dersler boýunça gatnaşyjylaryň sany")
    st.plotly_chart(fig_sub, use_container_width=True)

# -----------------------------
# TAB 2: OLYMPIAD PROFILE
# -----------------------------
with tab_olymp:
    st.subheader("Olimpiada gatnaşygy")

    # level distribution
    fig_lvl = px.pie(
        f,
        names="Olimpiadanyň derejesi (halkara / sebitleýin)",
        hole=0.4
    )
    fig_lvl = beautify(fig_lvl, "Olimpiadanyň derejesi boýunça paý")
    st.plotly_chart(fig_lvl, use_container_width=True)

    giveLine()
    st.markdown("### Halkara derejeli olimpiada, uniwersitet we okuwa giren ýyl boýunça")

    f_int = f[f["Olimpiadanyň derejesi (halkara / sebitleýin)"] == "Halkara"]

    fig_int_uni_year = count_heatmap(
        f_int,
        row_col="Okuw mekdebi",
        col_col="Okuwa giren ýyly",
        title="Halkara derejeli talyplar, Uniwersitet, Okuwa giren ýyly"
    )
    st.plotly_chart(fig_int_uni_year, use_container_width=True)

    giveLine()
    # students by school
    school_counts = f["Okuw mekdebi"].value_counts().reset_index()
    school_counts.columns = ["Okuw mekdebi", "Talyp sany"]
    fig_school = px.bar(school_counts, x="Okuw mekdebi", y="Talyp sany")
    fig_school = beautify(fig_school, "Okuw mekdebi boýunça gatnaşyjylaryň sany")
    st.plotly_chart(fig_school, use_container_width=True)

    # Top 10 by trips
    giveLine()
    st.markdown("### Top 10 talyp — daşary ýurt saparlarynyň sany boýunça")
    top_trips = f.sort_values("Daşary ýurt saparlarynyň sany", ascending=False).head(10)
    fig_top_trips = px.bar(
        top_trips,
        x="Talyp ady, familiýasy",
        y="Daşary ýurt saparlarynyň sany"
    )
    fig_top_trips = beautify(fig_top_trips, "Top 10: iň köp daşary ýurt sapary eden talyplar")
    st.plotly_chart(fig_top_trips, use_container_width=True)

# -----------------------------
# TAB 3: TRAVEL & EXPENSES
# -----------------------------
with tab_travel:


    # expenses by school
    exp_by_school = (
        f.groupby("Okuw mekdebi")["Takmynan çykdajy möçberi (USD/manat)"]
        .sum()
        .reset_index()
    )
    fig_exp_school = px.bar(
        exp_by_school,
        x="Okuw mekdebi",
        y="Takmynan çykdajy möçberi (USD/manat)",
    )
    fig_exp_school = beautify(fig_exp_school, "Okuw mekdebi boýunça çykdajylar")
    st.plotly_chart(fig_exp_school, use_container_width=True)

    # Top 10 by expenses
    top_exp = f.sort_values("Takmynan çykdajy möçberi (USD/manat)", ascending=False).head(10)
    fig_top_exp = px.bar(
        top_exp,
        x="Talyp ady, familiýasy",
        y="Takmynan çykdajy möçberi (USD/manat)"
    )
    fig_top_exp = beautify(fig_top_exp, "Top 10: iň köp çykdajy edilen talyplar")
    st.plotly_chart(fig_top_exp, use_container_width=True)


    giveLine()
    if "Maliýeleşdirmegiň çeşmesi (döwlet / hemaýatkär)" in f.columns:

        fig_box_sponsor = px.box(
            f,
            x="Maliýeleşdirmegiň çeşmesi (döwlet / hemaýatkär)",
            y="Takmynan çykdajy möçberi (USD/manat)",
            color="Maliýeleşdirmegiň çeşmesi (döwlet / hemaýatkär)",
            points="all"
        )
        fig_box_sponsor.update_traces(jitter=0.25, marker_size=7, opacity=0.55)
        fig_box_sponsor = beautify(fig_box_sponsor,
            "Çykdajy paýlanyşy — döwlet / hemaýatkär boýunça"
        )
        st.plotly_chart(fig_box_sponsor, use_container_width=True)
    

    giveLine()
    if "Maliýeleşdirmegiň çeşmesi (döwlet / hemaýatkär)" in f.columns:
        st.markdown("### Uniwersitet we maliýeleşdirme çeşmesi boýunça çykdajy ýylylyk kartasy")

        pivot_uni_sponsor = pd.pivot_table(
            f,
            values="Takmynan çykdajy möçberi (USD/manat)",
            index="Okuw mekdebi",
            columns="Maliýeleşdirmegiň çeşmesi (döwlet / hemaýatkär)",
            aggfunc="sum",
            fill_value=0
        )

        fig_uni_sponsor = px.imshow(
            pivot_uni_sponsor,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Maliýeleşdirmegiň çeşmesi", y="Okuw mekdebi", color="Jemi çykdajy")
        )
        fig_uni_sponsor = beautify(fig_uni_sponsor,
            ""
        )
        st.plotly_chart(fig_uni_sponsor, use_container_width=True)
    

    giveLine()
    st.markdown("### Uniwersitet we ýagdaýy boýunça çykdajy ýylylyk kartasy")

    pivot_uni_status = pd.pivot_table(
        f,
        values="Takmynan çykdajy möçberi (USD/manat)",
        index="Okuw mekdebi",
        columns="Ýagdaýy (galan / giden)",
        aggfunc="sum",
        fill_value=0
    )

    fig_uni_status_exp = px.imshow(
        pivot_uni_status,
        text_auto=True,
        color_continuous_scale="Teal",
        labels=dict(x="Ýagdaýy", y="Okuw mekdebi", color="Jemi çykdajy")
    )
    fig_uni_status_exp = beautify(fig_uni_status_exp,
        ""
    )
    st.plotly_chart(fig_uni_status_exp, use_container_width=True)


    if "Maliýeleşdirmegiň çeşmesi (döwlet / hemaýatkär)" in f.columns:
        st.markdown("### Maliýe akymy — çeşmeden uniwersitete we ýagdaýyna çenli")

        fig_sunburst = px.sunburst(
            f,
            path=[
                "Maliýeleşdirmegiň çeşmesi (döwlet / hemaýatkär)",
                "Okuw mekdebi",
                "Ýagdaýy (galan / giden)",
            ],
            values="Takmynan çykdajy möçberi (USD/manat)",
            color="Maliýeleşdirmegiň çeşmesi (döwlet / hemaýatkär)",
        )
        fig_sunburst = beautify(fig_sunburst,
            "Maliýeleşdirme - uniwersitet - galan / giden boýunça çykdajy"
        )
        st.plotly_chart(fig_sunburst, use_container_width=True)
    

    st.markdown("### Daşary ýurt saparlarynyň sany — galan / giden boýunça paýlanyş")

    fig_hist_status = px.histogram(
        f,
        x="Daşary ýurt saparlarynyň sany",
        color="Ýagdaýy (galan / giden)",
        barmode="overlay",
        nbins=10
    )
    fig_hist_status.update_traces(opacity=0.8)
    fig_hist_status = beautify(fig_hist_status,
        ""
    )
    st.plotly_chart(fig_hist_status, use_container_width=True)

    fig_bubble = px.scatter(
    f,
    x="Daşary ýurt saparlarynyň sany",
    y="Takmynan çykdajy möçberi (USD/manat)",
    size="Takmynan çykdajy möçberi (USD/manat)",
    color="Okuw mekdebi",
    hover_name="Talyp ady, familiýasy"
)
    fig_bubble = beautify(fig_bubble, "Sapar sany & çykdajy ")
    st.plotly_chart(fig_bubble, use_container_width=True)

    exp_year = f.groupby("Okuwa giren ýyly")["Takmynan çykdajy möçberi (USD/manat)"].sum().reset_index()

    fig_trend = px.line(
        exp_year,
        x="Okuwa giren ýyly",
        y="Takmynan çykdajy möçberi (USD/manat)",
        markers=True
    )
    fig_trend = beautify(fig_trend, "Ýyl boýunça çykdajylaryň üýtgemesi")
    st.plotly_chart(fig_trend, use_container_width=True)

 

    fig_heat = px.density_heatmap(
    f,
    x="Daşary ýurt saparlarynyň sany",
    y="Takmynan çykdajy möçberi (USD/manat)",
    nbinsx=10,
    nbinsy=10,
    color_continuous_scale="Blues",
)
    fig_heat = beautify(fig_heat, "Sapar sany & çykdajy — dykyzlyk kartasy")
    st.plotly_chart(fig_heat, use_container_width=True)


# -----------------------------
# TAB 4: UNIVERSITIES & CAREERS
# -----------------------------



with tab_unis:
    st.subheader("Uniwersitetler we karýera ýolugy")

    fig_career_flow = make_sankey(
    f,
    col_left="Okuw mekdebi",
    col_mid="Dersi",
    col_right="Okuwdan soňky iş ýeri",
    title="Okuwdan soňky iş ýeri boýunça gatnaşyjylar",
    left_prefix="Uni",
    mid_prefix="Ders",
    right_prefix="Iş"
)
    st.plotly_chart(fig_career_flow, use_container_width=True)

    # work country
    if "Işleýän ýurdy" in f.columns:
        work_country = f["Işleýän ýurdy"].fillna("Maglumat ýok").value_counts().reset_index()
        work_country.columns = ["Işleýän ýurdy", "Talyp sany"]
        fig_wc = px.bar(work_country, x="Işleýän ýurdy", y="Talyp sany")
        fig_wc = beautify(fig_wc, "Işleýän ýurtlar boýunça paýlanyşyk")
        st.plotly_chart(fig_wc, use_container_width=True)
    
    st.markdown("### Okuwdan soňky iş ýerleriniň paýlanyşy")

    work_counts = (
        f["Okuwdan soňky iş ýeri"]
        .fillna("Maglumat ýok")
        .value_counts()
        .reset_index()
    )
    work_counts.columns = ["Okuwdan soňky iş ýeri", "Talyp sany"]

    fig_work = px.bar(
        work_counts.head(15),
        x="Okuwdan soňky iş ýeri",
        y="Talyp sany"
    )
    fig_work = beautify(fig_work, "Top 15 okuwdan soňky iş ýerleri")
    st.plotly_chart(fig_work, use_container_width=True)


    #  drop nan
    st.markdown("### Uniwersitet, okuwdan soňky iş ýeri — ýylylyk kartasy")

# Drop rows where job is NaN
    f_clean = f[f["Okuwdan soňky iş ýeri"].str.strip() != ""]

    pivot_uni_job = pd.pivot_table(
        f_clean,
        values="Talyp ady, familiýasy",
        index="Okuw mekdebi",
        columns="Okuwdan soňky iş ýeri",
        aggfunc="count",
        fill_value=0
    )

    fig_uni_job = px.imshow(
        pivot_uni_job,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Okuwdan soňky iş ýeri", y="Okuw mekdebi", color="Talyp sany")
    )

    fig_uni_job = beautify(fig_uni_job, "")
    st.plotly_chart(fig_uni_job, use_container_width=True)


    st.markdown("### Okuwdan soňky iş ýeri — galan / giden boýunça deňeşdiriş")

    job_status = (
        f[["Okuwdan soňky iş ýeri", "Ýagdaýy (galan / giden)"]]
        .fillna({"Okuwdan soňky iş ýeri": "Maglumat ýok"})
        .value_counts()
        .reset_index(name="Talyp sany")
    )

    fig_job_status = px.bar(
        job_status,
        x="Okuwdan soňky iş ýeri",
        y="Talyp sany",
        color="Ýagdaýy (galan / giden)",
        barmode="group"
    )
    fig_job_status = beautify(fig_job_status,
        ""
    )
    st.plotly_chart(fig_job_status, use_container_width=True)

    st.markdown("### Uniwersitet, iş edýän ýurt — ýylylyk kartasy")

    pivot_uni_country = pd.pivot_table(
        f,
        values="Talyp ady, familiýasy",
        index="Okuw mekdebi",
        columns="Işleýän ýurdy",
        aggfunc="count",
        fill_value=0
    )

    fig_uni_country = px.imshow(
        pivot_uni_country,
        text_auto=True,
        color_continuous_scale="Teal",
        labels=dict(x="Işleýän ýurdy", y="Okuw mekdebi", color="Talyp sany")
    )
    fig_uni_country = beautify(fig_uni_country,
        ""
    )
    st.plotly_chart(fig_uni_country, use_container_width=True)

    st.markdown("### Karýera ýörelgesi — uniwersitetden ýurda we işe çenli")

    fig_career_sun = px.sunburst(
        f,
        path=["Okuw mekdebi", "Işleýän ýurdy", "Okuwdan soňky iş ýeri"],
        values=None,
        color="Işleýän ýurdy"
    )
    fig_career_sun = beautify(fig_career_sun,
        ""
    )
    st.plotly_chart(fig_career_sun, use_container_width=True)


# -----------------------------
# TAB 5: BRAIN DRAIN VS RETENTION
# -----------------------------
with tab_brain:
    st.subheader("Ýurtda galanlar / daşary ýurda gidenler")

    # pie status
    fig_status = px.pie(f, names="Ýagdaýy (galan / giden)", hole=0.4)
    fig_status = beautify(fig_status, "Ýagdaýy boýunça paý — galan / giden")
    st.plotly_chart(fig_status, use_container_width=True)

    fig_status_flow_2 = make_sankey(
    f,
    col_left="Okuw mekdebi",
    col_mid="Ýagdaýy (galan / giden)",
    col_right="Işleýän ýurdy",
    title="Uniwersitet , Ýagdaýy (galan/giden) , Işleýän ýurdy",
    left_prefix="Uni",
    mid_prefix="Status",
    right_prefix="Ýurt"
    )
    st.plotly_chart(fig_status_flow_2, use_container_width=True)

    st.write("")
    st.write("")


    # fig_status_flow_2 = make_sankey(
    #     f,
    #     col_left="Okuw mekdebi",
    #     col_mid="Dersi",
    #     col_right="Ýagdaýy (galan / giden)",
    #     title="Uniwersitet → Olimpiada dersi → Ýagdaýy (galan/giden)",
    #     left_prefix="Uni",
    #     mid_prefix="Ders",
    #     right_prefix="Status"
    # )

    # st.plotly_chart(fig_status_flow_2, use_container_width=True)





    # status by school
    status_by_school = (
        f.groupby(["Okuw mekdebi", "Ýagdaýy (galan / giden)"])["Talyp ady, familiýasy"]
        .count()
        .reset_index()
    )
    status_by_school.columns = ["Okuw mekdebi", "Ýagdaýy (galan / giden)", "Sany"]
    fig_stat_school = px.bar(
        status_by_school,
        x="Okuw mekdebi",
        y="Sany",
        color="Ýagdaýy (galan / giden)",
        barmode="group",
    )
    fig_stat_school = beautify(fig_stat_school, "Okuw mekdebi boýunça galan / gidenleriň sany")
    st.plotly_chart(fig_stat_school, use_container_width=True)

  
    df = f.copy()

    # tiny random offset so overlapping points separate visually
    rng = np.random.default_rng(42)
    df["Trips_plot"] = df["Daşary ýurt saparlarynyň sany"] + rng.uniform(-0.15, 0.15, len(df))
    df["Cost_plot"] = df["Takmynan çykdajy möçberi (USD/manat)"] + rng.uniform(-300, 300, len(df))

    fig_scatter = px.scatter(
        df,
        x="Trips_plot",
        y="Cost_plot",
        color="Ýagdaýy (galan / giden)",
        hover_name="Talyp ady, familiýasy",
    )

    fig_scatter.update_traces(
        marker=dict(size=15, opacity=0.75, line=dict(width=1, color="white"))
    )

    fig_scatter = beautify(fig_scatter, 
        "Saparlaryň sany & çykdajy — galan / giden boýunça (her talyp aýratyn görkezilýär)"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

# ADVANCED

    fig = px.box(
    f,
    x="Ýagdaýy (galan / giden)",
    y="Takmynan çykdajy möçberi (USD/manat)",
    color="Ýagdaýy (galan / giden)",
    title="Çykdajy mukdarynyň paýlanyşy — galan / giden boýunça"
)
    fig.update_yaxes(range=[-1000, f["Takmynan çykdajy möçberi (USD/manat)"].max() + 5000])

    st.plotly_chart(fig, use_container_width=True)


    fig = px.box(
    f,
    x="Ýagdaýy (galan / giden)",
    y="Daşary ýurt saparlarynyň sany",
    color="Ýagdaýy (galan / giden)",
    title="Daşary ýurt saparlarynyň sany — galan / giden boýunça"
)
    fig.update_yaxes(range=[0, f["Daşary ýurt saparlarynyň sany"].max() + 5])

    st.plotly_chart(fig, use_container_width=True)



    # fig = px.violin(
    # f,
    # x="Ýagdaýy (galan / giden)",
    # y="Takmynan çykdajy möçberi (USD/manat)",
    # color="Ýagdaýy (galan / giden)",
    # box=True,
    # points="all",
    # title="Daşary ýurt saparlarynyň sany — paýlanyş"
    # )
    # st.plotly_chart(fig, use_container_width=True)


    # fig = px.scatter(
    # f,
    # x="Daşary ýurt saparlarynyň sany",
    # y="Takmynan çykdajy möçberi (USD/manat)",
    # size="Daşary ýurt saparlarynyň sany",
    # color="Ýagdaýy (galan / giden)",
    # hover_name="Talyp ady, familiýasy",
    # title="3D Bubble Chart — Sapar sany, çykdajy we ýagdaýy"
    # )
    # st.plotly_chart(fig, use_container_width=True)


    fig = px.sunburst(
    f,
    path=["Okuw mekdebi","Dersi","Ýagdaýy (galan / giden)"],
    title="Uniwersitet - Dersi - Ýagdaýy"
    )
    st.plotly_chart(fig, use_container_width=True)

    giveLine()
    st.markdown("### Uniwersitet we okuwa giren ýyl boýunça giden talyplar")

    f_giden = f[f["Ýagdaýy (galan / giden)"] == "giden"]

    fig_uni_year_giden = count_heatmap(
        f_giden,
        row_col="Okuw mekdebi",
        col_col="Okuwa giren ýyly",
        title="Uniwersitet, Okuwa giren ýyly, giden talyplar"
    )
    st.plotly_chart(fig_uni_year_giden, use_container_width=True)

    giveLine()

    st.markdown("### Uniwersitet we okuwa giren ýyl boýunça galan talyplar")

    f_galan = f[f["Ýagdaýy (galan / giden)"] == "galan"]

    fig_uni_year_galan = count_heatmap(
        f_galan,
        row_col="Okuw mekdebi",
        col_col="Okuwa giren ýyly",
        title="Uniwersitet, Okuwa giren ýyly, galan talyplar"
    )
    st.plotly_chart(fig_uni_year_galan, use_container_width=True)

    giveLine()
    st.markdown("### Uniwersitet we ýagdaýy (galan / giden) boýunça ýyllyk kartasy")

    fig_uni_status = count_heatmap(
        f,
        row_col="Okuw mekdebi",
        col_col="Ýagdaýy (galan / giden)",
        title="Uniwersitet, Ýagdaýy (galan / giden)"
    )
    st.plotly_chart(fig_uni_status, use_container_width=True)

    giveLine()
    st.markdown("### Okuwa giren ýyly we ýagdaýy boýunça ýyllyk kartasy")

    fig_year_status = count_heatmap(
        f,
        row_col="Okuwa giren ýyly",
        col_col="Ýagdaýy (galan / giden)",
        title="Okuwa giren ýyly, Ýagdaýy (galan / giden)"
    )
    st.plotly_chart(fig_year_status, use_container_width=True)



# -----------------------------
# TAB 6: ADVANCED ANALYTICS
# -----------------------------
with tab_advanced:
 

    # correlation heatmap (numeric columns)
    st.markdown("Korelýasiýa matrisasy (sanly sütunlar boýunça)")
    num_for_corr = f[num_cols + ["Okuw dowamlylygy"]] if "Okuw dowamlylygy" in f.columns else f[num_cols]
    num_for_corr = num_for_corr.dropna(axis=1, how="all")
    corr = num_for_corr.corr()
    if not corr.empty:
        fig_corr = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="Blues",
            showscale=True
        )
        fig_corr.update_layout(margin=dict(l=80, r=20, t=40, b=40))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Korelýasiýa görkezmek üçin sanly maglumatlar ýeterlik däl.")

    st.markdown("---")
    st.markdown("###  Ýurtda galarmy ýa-da gidermi? (ýönekeý model)")

    # logistic regression: predict leaving
    features = ["Daşary ýurt saparlarynyň sany", "Takmynan çykdajy möçberi (USD/manat)", "Okuwa giren ýyly"]
    data_ml = f.dropna(subset=features)
    if len(data_ml["left_binary"].unique()) == 2 and len(data_ml) > 10:
        X = data_ml[features]
        y = data_ml["left_binary"]

        model = LogisticRegression()
        model.fit(X, y)

        # show coefficients


        st.markdown("Talyp profiline laýyklykda modeliň çaklamasyny görmek üçin baha giriziň:")
        c1, c2, c3 = st.columns(3)
        trips_in = c1.number_input("Daşary ýurt saparlarynyň sany", min_value=0, value=int(f["Daşary ýurt saparlarynyň sany"].median()))
        cost_in = c2.number_input("Takmynan çykdajy (USD/manat)", min_value=0, value=int(f["Takmynan çykdajy möçberi (USD/manat)"].median()))
        year_in = c3.number_input("Okuwa giren ýyly", min_value=2000, max_value=2030, value=int(f["Okuwa giren ýyly"].median()))

        X_new = pd.DataFrame([[trips_in, cost_in, year_in]], columns=features)
        prob_leave = model.predict_proba(X_new)[0, 1]

        st.success(f"Modeliň çaklamasy: bu profil bilen talybanyň daşary ýurda gitmek ähtimallygy takmynan **{prob_leave*100:.1f}%**.")
        st.caption("Bu model diňe 3 görkezijä (sapar sany, çykdajy, okuwa giren ýyly) esaslanýar. "
        "Netijeler trendleri görkezýär, ýöne her bir talyp boýunça takyk karary aňlatmaýar."
    )

    else:
        st.info("Model döretmek üçin galan/giden boýunça maglumatlar ýeterlik däl.")

    st.markdown("---")
    st.markdown("### K-means klasterleme — talyp görnüşleri ")

    # K-means clustering
    cluster_features = ["Daşary ýurt saparlarynyň sany", "Takmynan çykdajy möçberi (USD/manat)"]
    data_cluster = f.dropna(subset=cluster_features).copy()
    if len(data_cluster) >= 5:
        Xc = data_cluster[cluster_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(Xc)

        k = 3  # 3 clusters
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        data_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

      

        st.write("Her klasteriň umumy statistikalary:")
        st.dataframe(
            data_cluster.groupby("Cluster")[cluster_features].agg(["mean", "min", "max"]),
            use_container_width=True
        )
        giveLine()
    else:
        st.info("Klasterleme üçin ýeterlik sanly maglumat ýok.")
    

    cluster_features = ["Daşary ýurt saparlarynyň sany", "Takmynan çykdajy möçberi (USD/manat)"]
    data_cluster = f.dropna(subset=cluster_features).copy()

    Xc = data_cluster[cluster_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xc)

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    data_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

    # 👇 add tiny random jitter only for plotting (doesn't affect clustering)
    rng = np.random.default_rng(42)
    data_cluster["Daşary ýurt saparlarynyň sany"] = data_cluster["Daşary ýurt saparlarynyň sany"] + rng.uniform(-0.15, 0.15, len(data_cluster))
    data_cluster["Takmynan çykdajy möçberi (USD/manat)"]  = data_cluster["Takmynan çykdajy möçberi (USD/manat)"] + rng.uniform(-300, 300, len(data_cluster))

    fig_cluster = px.scatter(
        data_cluster,
        x="Daşary ýurt saparlarynyň sany",
        y="Takmynan çykdajy möçberi (USD/manat)",
        color="Cluster",
        hover_name="Talyp ady, familiýasy",
    )

    fig_cluster = beautify(fig_cluster, "Talyp klasterleri — sapar sany & çykdajy boýunça (her talyp aýratyn)")
    fig_cluster.update_traces(
        marker=dict(size=15, line=dict(width=1, color="white"), opacity=0.75)
    )
    st.plotly_chart(fig_cluster, use_container_width=True)


    cluster_features = ["Daşary ýurt saparlarynyň sany", "Takmynan çykdajy möçberi (USD/manat)"]
    data_cluster = f.dropna(subset=cluster_features).copy()

    # scale + kmeans as before
    Xc = data_cluster[cluster_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xc)

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    data_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

    # 👇 aggregate: how many students per (trips, cost, cluster)
    agg = (
        data_cluster
        .groupby(cluster_features + ["Cluster"])
        .size()
        .reset_index(name="Talyp sany")
    )

    fig_cluster = px.scatter(
        agg,
        x="Daşary ýurt saparlarynyň sany",
        y="Takmynan çykdajy möçberi (USD/manat)",
        color="Cluster",
        size="Talyp sany",             # bubble size = number of students
        hover_name="Talyp sany",
        hover_data={ "Talyp sany": True }
    )

    fig_cluster = beautify(fig_cluster, "Talyp klasterleri — sapar sany & çykdajy boýunça (talyp sany boýunça)")
    fig_cluster.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.8))
    st.plotly_chart(fig_cluster, use_container_width=True)


    

# -----------------------------
# TAB 7: RAW TABLE
# -----------------------------
with tab_table:
    giveLine()
    st.subheader("Dolulygyna data (filtrlenenen maglumatlar)")
    st.dataframe(f.drop(columns=["left_binary"], errors="ignore"), use_container_width=True)
    st.caption("Datany Excel-e eksport etmek isleseňiz: '⋮' menýudan 'Download as CSV' saýlap bilersiňiz.")
