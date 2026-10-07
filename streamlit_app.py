import io
import pandas as pd
import streamlit as st
from swing_engine import MarketData, analyze_tickers

st.set_page_config(page_title="Swing Score", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1500px; padding-top: 1.2rem;}
.small-note {color:#666;font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Swing Score")
st.caption("Detector d'oportunitats swing · Stage 2 · Fiabilitat històrica")

if "results" not in st.session_state:
    st.session_state.results = None
if "last_tickers" not in st.session_state:
    st.session_state.last_tickers = ""

col1, col2 = st.columns([4,1])
with col1:
    ticker_text = st.text_area("Tickers", value=st.session_state.last_tickers, height=90,
                               placeholder="PANW, NVDA, AMD, PLTR, VRT")
with col2:
    st.write("")
    st.write("")
    analyze = st.button("🔎 ANALITZAR", type="primary", use_container_width=True)

uploaded = st.file_uploader("📋 Importar Watchlist de Yahoo Finance (CSV)", type=["csv"])
if uploaded is not None:
    try:
        dfw = pd.read_csv(uploaded)
        ticker_col = next((c for c in dfw.columns if str(c).strip().lower() in {"symbol","ticker","tickers"}), None)
        if ticker_col is None:
            st.error("No he trobat la columna Symbol/Ticker a la Watchlist.")
        else:
            imported = list(dict.fromkeys(str(v).strip().upper() for v in dfw[ticker_col].dropna() if str(v).strip()))
            if imported:
                ticker_text = ", ".join(imported)
                st.info(f"Watchlist carregada: {len(imported)} tickers. Prem ANALITZAR.")
            else:
                st.warning("La Watchlist no conté cap ticker.")
    except Exception as e:
        st.error(f"Error llegint la Watchlist: {type(e).__name__}: {e}")

if analyze:
    tickers = list(dict.fromkeys(t.strip().upper() for t in ticker_text.replace("\n", ",").split(",") if t.strip()))
    if not tickers:
        st.warning("Introdueix almenys un ticker o carrega una Watchlist.")
    else:
        st.session_state.last_tickers = ", ".join(tickers)
        with st.spinner(f"Analitzant {len(tickers)} tickers…"):
            data = MarketData(tickers)
            st.session_state.results = analyze_tickers(tickers, data)

results = st.session_state.results
if results:
    rows=[]
    for i,r in enumerate(results,1):
        rel=r.get("stage2_reliability") or {}
        if r.get("stage2"):
            rr = "Sense dades" if rel.get("reliability") is None else f"{rel['reliability']*100:.0f}% (N={rel['signals']}) · μ20d {rel['mean20']*100:+.1f}% · med {rel['median20']*100:+.1f}%"
        else: rr="—"
        ext=r.get("price_vs_ma30w")
        if not r.get("stage2") or r.get("score",0)<0.60 or ext is None: action="⚪ WATCH"
        elif ext>0.15: action="🟡 ESPERAR PULLBACK"
        else: action="🟢 ENTRADA"
        rows.append({
            "#":i,"Ticker":r["ticker"],"Preu":r["price"],"Score":r["score"],"Etiqueta":r["label"].upper(),
            "Stage 2":"🟢 SÍ" if r.get("stage2") else "—","Acció":action,"Fiabilitat hist. 20d":rr,
            "vs MA30w": None if ext is None else f"{ext*100:+.1f}%","MA↑":"Sí" if r.get("ma30w_rising") else "No",
            "RS":None if r.get("rs") is None else f"{r['rs']*100:+.1f}%","Vol×":r.get("vol_ratio"),
            "Raw scores":f"swing={r['raw']['swing']:.3f} riser={r['raw']['riser']:.3f} down={r['raw']['down']:.3f} top={r['raw']['top']:.3f}"
        })
    out=pd.DataFrame(rows)
    st.subheader(f"Resultats — {len(out)} tickers")
    st.dataframe(out, use_container_width=True, hide_index=True)
    s2=out[out["Stage 2"]!="—"]
    st.subheader(f"🟢 STAGE 2 CANDIDATES — {len(s2)}")
    st.dataframe(s2, use_container_width=True, hide_index=True)
    st.caption("Fiabilitat hist. 20d = percentatge de senyals històrics Stage 2 NO→SÍ amb retorn positiu a 20 sessions. Classificació orientativa, no garantia.")
else:
    st.info("Introdueix tickers o importa una Watchlist Yahoo Finance i prem ANALITZAR.")
