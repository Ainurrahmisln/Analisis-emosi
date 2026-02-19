import streamlit as st

def inject_styles():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

/* ---- ROOT TOKENS ---- */
:root {
  --bg:       #0e0d0b;
  --surface:  #161412;
  --elevated: #1e1b18;
  --border:   #2e2a25;
  --accent:   #8b1a1a;
  --accent2:  #c0392b;
  --gold:     #c9a84c;
  --text:     #e8e0d0;
  --muted:    #8a8070;
  --faint:    #3a342c;

  --font-display: 'Playfair Display', Georgia, serif;
  --font-body:    'IBM Plex Sans', sans-serif;
  --font-mono:    'IBM Plex Mono', monospace;

  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 14px;
}

/* ---- GLOBAL BASE ---- */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* ---- MAIN CONTAINER ---- */
.main .block-container {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-top: 3px solid var(--accent) !important;
  border-radius: 0 0 var(--radius-lg) var(--radius-lg) !important;
  padding: 2.5rem 2.5rem 4rem !important;
  box-shadow: 0 20px 60px rgba(0,0,0,0.6) !important;
  max-width: 1400px !important;
}

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { background: var(--surface) !important; }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
  font-family: var(--font-mono) !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  font-style: normal !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
  color: var(--gold) !important;
}
section[data-testid="stSidebar"] .stDivider { border-color: var(--border) !important; }

/* ---- TYPOGRAPHY ---- */
h1, h2, h3 { font-family: var(--font-display) !important; }

h1 {
  font-size: 2.4rem !important;
  font-weight: 900 !important;
  letter-spacing: -0.02em !important;
  color: var(--text) !important;
  line-height: 1.15 !important;
  border-bottom: 2px solid var(--accent) !important;
  padding-bottom: 0.6rem !important;
  margin-bottom: 0.25rem !important;
}

h2 {
  font-size: 1.4rem !important;
  font-weight: 700 !important;
  color: var(--text) !important;
  letter-spacing: 0.01em !important;
  margin-top: 2rem !important;
}

h3 {
  font-size: 1.1rem !important;
  font-weight: 400 !important;
  font-style: italic !important;
  color: var(--gold) !important;
}

p, label, .stMarkdown, div[class*="stText"] {
  font-family: var(--font-body) !important;
  color: var(--text) !important;
  font-weight: 300 !important;
}

/* caption/subtext */
.stCaption, small {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  color: var(--muted) !important;
  letter-spacing: 0.05em !important;
}

/* ---- METRICS ---- */
div[data-testid="metric-container"] {
  background: var(--elevated) !important;
  border: 1px solid var(--border) !important;
  border-left: 3px solid var(--accent) !important;
  border-radius: var(--radius-md) !important;
  padding: 14px 18px !important;
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
  font-family: var(--font-mono) !important;
  font-size: 0.7rem !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: var(--font-display) !important;
  font-size: 2rem !important;
  font-weight: 700 !important;
  color: var(--text) !important;
}

/* ---- DATAFRAME ---- */
div[data-testid="stDataFrame"] {
  background: var(--elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}
.stDataFrame table {
  font-family: var(--font-mono) !important;
  font-size: 0.82rem !important;
}

/* ---- TABS ---- */
button[data-baseweb="tab"] {
  font-family: var(--font-mono) !important;
  font-size: 0.78rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
  color: var(--muted) !important;
  border-bottom: 2px solid transparent !important;
  padding: 0.6rem 1.2rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--text) !important;
  border-bottom: 2px solid var(--accent) !important;
}
div[data-baseweb="tab-list"] {
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
div[data-baseweb="tab-panel"] {
  padding-top: 1.5rem !important;
}

/* ---- BUTTONS ---- */
button[kind="primary"], button[data-testid="baseButton-primary"] {
  background: var(--accent) !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.6rem 1.4rem !important;
  transition: background 0.2s !important;
}
button[kind="primary"]:hover { background: var(--accent2) !important; }

button[kind="secondary"] {
  background: transparent !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}

/* ---- INPUTS ---- */
input[type="text"], input[type="number"], select, textarea {
  background: var(--elevated) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.85rem !important;
}
input:focus, select:focus, textarea:focus {
  border-color: var(--accent) !important;
  outline: none !important;
  box-shadow: 0 0 0 2px rgba(139,26,26,0.25) !important;
}

/* ---- EXPANDER ---- */
div[data-testid="stExpander"] {
  background: var(--elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}
div[data-testid="stExpander"] summary {
  font-family: var(--font-mono) !important;
  font-size: 0.8rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: var(--muted) !important;
}
div[data-testid="stExpander"] summary:hover { color: var(--text) !important; }

/* ---- ALERTS / INFO ---- */
div[data-testid="stAlert"] {
  background: var(--elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  font-family: var(--font-body) !important;
}

/* ---- CHECKBOX / TOGGLE ---- */
label[data-baseweb="checkbox"] > div {
  border-color: var(--faint) !important;
}

/* ---- DIVIDER ---- */
hr { border-color: var(--border) !important; }

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--faint); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ---- EMOTION CARDS (custom) ---- */
.emo-card {
  background: var(--elevated);
  border: 1px solid var(--border);
  border-top: 3px solid var(--accent);
  border-radius: var(--radius-md);
  padding: 16px 18px;
  text-align: left;
  transition: border-color 0.2s, transform 0.15s;
}
.emo-card:hover {
  border-color: var(--gold);
  transform: translateY(-2px);
}
.emo-card .emo-icon { font-size: 28px; line-height: 1; margin-bottom: 8px; }
.emo-card .emo-label {
  font-family: var(--font-mono);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--muted);
  margin-bottom: 4px;
}
.emo-card .emo-count {
  font-family: var(--font-display);
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--text);
  line-height: 1.1;
}
.emo-card .emo-pct {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  color: var(--gold);
  margin-top: 2px;
}

/* ---- SECTION LABEL ---- */
.section-eyebrow {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: var(--accent2);
  margin-bottom: 0.4rem;
  display: block;
}

/* ---- STAT ROW ---- */
.stat-row {
  display: flex;
  gap: 12px;
  padding: 10px 0;
  border-bottom: 1px solid var(--border);
}
.stat-row:last-child { border-bottom: none; }
.stat-key {
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: var(--muted);
  min-width: 180px;
}
.stat-val {
  font-family: var(--font-mono);
  font-size: 0.78rem;
  color: var(--text);
  font-weight: 500;
}

/* ---- TITLE CAPTION LINE ---- */
.title-caption {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--muted);
  letter-spacing: 0.06em;
  margin-top: -0.5rem;
  margin-bottom: 2rem;
  display: block;
  border-left: 2px solid var(--faint);
  padding-left: 10px;
}

.gap-sm { height: 16px; }
.gap-md { height: 28px; }

/* ---- FILE UPLOADER ---- */
/* container box */
[data-testid="stFileUploader"] section {
  background: var(--elevated) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: var(--radius-md) !important;
}
[data-testid="stFileUploader"] section:hover {
  border-color: var(--accent) !important;
}
/* "Drag and drop" text */
[data-testid="stFileUploader"] section p {
  color: var(--text) !important;
  font-family: var(--font-body) !important;
}
/* "Limit …" small text */
[data-testid="stFileUploader"] section small {
  color: var(--muted) !important;
  font-family: var(--font-mono) !important;
}
/* BROWSE FILES button — Streamlit renders this as a specific button */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzoneInstructions"] ~ div button {
  background: transparent !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
}
[data-testid="stFileUploader"] button:hover {
  border-color: var(--accent) !important;
  color: var(--text) !important;
}

/* ---- SELECT / DROPDOWN ---- */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] div[role="combobox"] {
  background: var(--elevated) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.85rem !important;
}
div[data-baseweb="select"] svg { fill: var(--muted) !important; }
/* dropdown menu panel */
ul[data-baseweb="menu"] {
  background: var(--elevated) !important;
  border: 1px solid var(--border) !important;
}
ul[data-baseweb="menu"] li {
  background: var(--elevated) !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.82rem !important;
}
ul[data-baseweb="menu"] li:hover,
ul[data-baseweb="menu"] li[aria-selected="true"] {
  background: var(--accent) !important;
  color: var(--text) !important;
}

/* ---- CHECKBOX ---- */
/* unchecked box border */
label[data-baseweb="checkbox"] [role="checkbox"] {
  border-color: var(--border) !important;
  background: var(--elevated) !important;
  border-radius: 3px !important;
}
/* checked state — replace blue fill with accent red */
label[data-baseweb="checkbox"] [role="checkbox"][aria-checked="true"],
label[data-baseweb="checkbox"] [role="checkbox"][aria-checked="mixed"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}
/* hover ring */
label[data-baseweb="checkbox"]:hover [role="checkbox"] {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(139,26,26,0.25) !important;
}
/* checkmark icon inside */
label[data-baseweb="checkbox"] [role="checkbox"] svg { color: var(--text) !important; }

/* ---- NUMBER INPUT +/− BUTTONS ---- */
button[data-testid="stNumberInputStepDown"],
button[data-testid="stNumberInputStepUp"],
/* fallback for older Streamlit builds */
div[data-testid="stNumberInput"] button {
  background: var(--elevated) !important;
  border: 1px solid var(--border) !important;
  color: var(--muted) !important;
  border-radius: var(--radius-sm) !important;
}
button[data-testid="stNumberInputStepDown"]:hover,
button[data-testid="stNumberInputStepUp"]:hover,
div[data-testid="stNumberInput"] button:hover {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: var(--text) !important;
}
/* number input text field */
div[data-testid="stNumberInput"] input {
  background: var(--elevated) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
}
div[data-testid="stNumberInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(139,26,26,0.25) !important;
}

/* ---- TEXT INPUT ---- */
div[data-testid="stTextInput"] input {
  background: var(--elevated) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
}
div[data-testid="stTextInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(139,26,26,0.25) !important;
}

/* ---- FOCUS / ACTIVE BLUE KILLER ---- */
/* wipe out any remaining Streamlit blue ring */
*:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px !important; }
button:focus   { box-shadow: none !important; }
</style>
""", unsafe_allow_html=True)
