import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import json, math, os

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MedicalGPT",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Theme State
# ─────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ─────────────────────────────────────────────
# Theme Definitions
# ─────────────────────────────────────────────
THEMES = {
    # "dark": {
    #     "--bg":      "#0a0e1a",
    #     "--surface": "#111827",
    #     "--surface2":"#1a2235",
    #     "--border":  "#2a3a55",
    #     "--accent":  "#00c9a7",
    #     "--accent2": "#3b82f6",
    #     "--warn":    "#f59e0b",
    #     "--text":    "#e2e8f0",
    #     "--muted":   "#8a9ab5",
    #     "--input-bg":"#111827",
    #     "--msg-bg":  "#111827",
    # },
    "light": {
        "--bg":      "#f0f4f8",
        "--surface": "#ffffff",
        "--surface2":"#e8f0fe",
        "--border":  "#c9d6e8",
        "--accent":  "#00a388",
        "--accent2": "#2563eb",
        "--warn":    "#d97706",
        "--text":    "#1a202c",
        "--muted":   "#4a5568",
        "--input-bg":"#ffffff",
        "--msg-bg":  "#ffffff",
    }
}

def get_css(theme):
    t = THEMES[theme]
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {{
    --bg:      {t["--bg"]};
    --surface: {t["--surface"]};
    --surface2:{t["--surface2"]};
    --border:  {t["--border"]};
    --accent:  {t["--accent"]};
    --accent2: {t["--accent2"]};
    --warn:    {t["--warn"]};
    --text:    {t["--text"]};
    --muted:   {t["--muted"]};
    --mono:    'Space Mono', monospace;
    --sans:    'DM Sans', sans-serif;
}}

html, body, [data-testid="stAppViewContainer"] {{
    background: var(--bg) !important;
    font-family: var(--sans);
    color: var(--text);
    font-size: 15px;
}}

[data-testid="stSidebar"] {{
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
    min-width: 21rem !important;
    max-width: 21rem !important;
    transform: none !important;
    visibility: visible !important;
    display: block !important;
}}
[data-testid="stSidebar"][aria-expanded="false"] {{
    margin-left: 0 !important;
    transform: none !important;
    display: block !important;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {{
    font-family: var(--mono);
    font-size: 0.95rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent) !important;
    margin-bottom: 0.6rem;
}}

.metric-card {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.metric-label {{
    font-family: var(--mono);
    font-size: 0.82rem;
    color: var(--muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
.metric-value {{
    font-family: var(--mono);
    font-size: 1.05rem;
    color: var(--accent);
    font-weight: 700;
}}
.metric-value.blue  {{ color: var(--accent2); }}
.metric-value.amber {{ color: var(--warn); }}

.info-row {{
    display: flex;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.95rem;
}}
.info-key {{
    color: var(--muted);
    font-family: var(--mono);
    font-size: 0.85rem;
}}
.info-val {{ color: var(--text); font-weight: 500; }}

[data-testid="stChatMessage"] {{
    background: {t["--msg-bg"]} !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 14px !important;
    font-size: 1rem !important;
}}

[data-testid="stChatInput"] {{
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    background: {t["--input-bg"]} !important;
    box-shadow: none !important;
    outline: none !important;
}}
[data-testid="stChatInput"]:focus-within {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,201,167,0.12) !important;
    outline: none !important;
}}
[data-testid="stChatInput"] * {{
    outline: none !important;
    box-shadow: none !important;
}}
[data-testid="stChatInputTextArea"] {{
    background: transparent !important;
    border: none !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 1rem !important;
    outline: none !important;
    box-shadow: none !important;
}}
[data-testid="stChatInputTextArea"]:focus {{
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}}

[data-testid="stSelectbox"] > div > div {{
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 0.95rem !important;
}}

hr {{ border-color: var(--border) !important; }}

.main-title {{
    font-family: var(--mono);
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}}
.main-subtitle {{
    font-family: var(--sans);
    font-size: 1rem;
    color: var(--muted);
    margin-bottom: 1.5rem;
    line-height: 1.6;
}}
.accent-dot {{ color: var(--accent); }}

.notes-box {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    font-family: var(--mono);
    font-size: 0.88rem;
    color: var(--muted);
    line-height: 1.8;
    word-break: break-all;
}}

.empty-state {{
    text-align: center;
    padding: 80px 20px;
    color: var(--muted);
}}
.empty-icon  {{ font-size: 3.5rem; margin-bottom: 14px; }}
.empty-text  {{ font-family: var(--sans); font-size: 1.05rem; }}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {{
    font-size: 0.95rem !important;
    color: var(--text) !important;
}}

[data-testid="stSidebar"] button {{
    font-size: 0.95rem !important;
    border-radius: 8px !important;
}}

.theme-badge {{
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 12px;
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--muted);
    margin-bottom: 8px;
}}
</style>
"""

# Inject CSS
st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model Definition
# ─────────────────────────────────────────────
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, pad_id, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.dropout   = nn.Dropout(dropout)
        self.d_model   = d_model
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(self.token_emb(x) * math.sqrt(self.d_model) + self.pos_emb(positions))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model; self.n_heads = n_heads; self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model); self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model); self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, causal_mask=None, pad_mask=None):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        K = self.W_k(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        V = self.W_v(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_head)
        if causal_mask is not None: scores = scores.masked_fill(causal_mask==0, float('-inf'))
        if pad_mask   is not None: scores = scores.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2)==0, float('-inf'))
        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        return self.W_o(self.dropout(attn).matmul(V).transpose(1,2).contiguous().view(B,T,self.d_model))

class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model,ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim,d_model), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn=MultiHeadAttention(d_model,n_heads,dropout); self.ffn=FeedForward(d_model,ffn_dim,dropout)
        self.norm1=nn.LayerNorm(d_model); self.norm2=nn.LayerNorm(d_model); self.dropout=nn.Dropout(dropout)
    def forward(self, x, causal_mask=None, pad_mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), causal_mask, pad_mask))
        return x + self.ffn(self.norm2(x))

class MedicalGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, ffn_dim, max_seq_len, pad_id, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.embeddings = Embeddings(vocab_size, d_model, max_seq_len, pad_id, dropout)
        self.layers  = nn.ModuleList([DecoderLayer(d_model,n_heads,ffn_dim,dropout) for _ in range(n_layers)])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.token_emb.weight
    def make_causal_mask(self, seq_len, device):
        return torch.tril(torch.ones(seq_len,seq_len,device=device)).unsqueeze(0).unsqueeze(0)
    def forward(self, input_ids, attention_mask=None):
        B,T = input_ids.shape
        x   = self.embeddings(input_ids)
        cm  = self.make_causal_mask(T, input_ids.device)
        for layer in self.layers: x = layer(x, causal_mask=cm, pad_mask=attention_mask)
        return self.lm_head(self.norm(x))

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR       = '/home/sem4/NLP/NLP'
TOKENIZER_PATH = f'{BASE_DIR}/tokenizer/medical_bpe.model'
EXP_DIR        = f'{BASE_DIR}/experiments'

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def get_experiments():
    exps = []
    if os.path.exists(EXP_DIR):
        for name in sorted(os.listdir(EXP_DIR)):
            if os.path.exists(f'{EXP_DIR}/{name}/checkpoints/best_model.pt') and \
               os.path.exists(f'{EXP_DIR}/{name}/config.json'):
                exps.append(name)
    return exps

@st.cache_resource
def load_model(exp_name):
    config_path = f'{EXP_DIR}/{exp_name}/config.json'
    model_path  = f'{EXP_DIR}/{exp_name}/checkpoints/best_model.pt'
    with open(config_path) as f:
        config = json.load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MedicalGPT(
        vocab_size=config['vocab_size'], d_model=config['d_model'],
        n_heads=config['n_heads'],       n_layers=config['n_layers'],
        ffn_dim=config['ffn_dim'],       max_seq_len=config['max_seq_len'],
        pad_id=config['pad_id'],
    )
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    return model, sp, config, device

def generate_top_p(model, sp, config, device, symptom_text,
                   max_new_tokens=150, temperature=0.7, top_p=0.9):
    """Generate using Top-P (nucleus) sampling."""
    EOS_ID, MAX_SEQ_LEN = config['eos_id'], config['max_seq_len']
    input_ids    = sp.Encode(f"<patient> {symptom_text} <doctor>", out_type=int)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_tensor.size(1) >= MAX_SEQ_LEN: break
            logits      = model(input_tensor)
            next_logits = logits[0, -1, :] / temperature

            # Top-P (nucleus) sampling
            probs_full = torch.softmax(next_logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs_full, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens once cumulative prob exceeds top_p
            sorted_idx_remove = cumulative_probs - sorted_probs > top_p
            sorted_probs[sorted_idx_remove] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()

            next_token = sorted_idx[torch.multinomial(sorted_probs, 1)].view(1,1)
            if next_token.item() == EOS_ID: break
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

    out = sp.Decode(input_tensor[0].tolist())
    if '<doctor>' in out: out = out.split('<doctor>')[-1].strip()
    return out

def metric_card(label, value, color=""):
    return f"""<div class="metric-card">
        <span class="metric-label">{label}</span>
        <span class="metric-value {color}">{value}</span>
    </div>"""

def info_row(key, val):
    return f"""<div class="info-row">
        <span class="info-key">{key}</span>
        <span class="info-val">{val}</span>
    </div>"""

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:

    # Theme toggle
    st.markdown("### 🩺 MedicalGPT")
    st.markdown(
        '<p style="font-size:0.88rem;color:var(--muted);margin-top:-8px;'
        'font-family:\'Space Mono\',monospace;">Single-Turn Medical Response Generator</p>',
        unsafe_allow_html=True
    )

    # Theme selector
    current = st.session_state.theme
    # theme_label = "🌙 Dark Mode" if current == "dark" else "☀️ Light Mode"
    # if st.button(theme_label, use_container_width=True):
    #     st.session_state.theme = "light" if current == "dark" else "dark"
    #     st.rerun()

    st.divider()

    # Experiment selector
    st.markdown("#### Select Experiment")
    experiments = get_experiments()
    if not experiments:
        st.error("No experiments found.")
        st.stop()
    selected_exp = st.selectbox("", experiments, label_visibility="collapsed")

    st.divider()

    with st.spinner("Loading weights..."):
        model, sp, config, device = load_model(selected_exp)

    total_params = sum(p.numel() for p in model.parameters())

    # Architecture
    st.markdown("#### Architecture")
    arch_html = "".join([
        info_row("d_model",    str(config['d_model'])),
        info_row("n_heads",    str(config['n_heads'])),
        info_row("n_layers",   str(config['n_layers'])),
        info_row("ffn_dim",    str(config['ffn_dim'])),
        info_row("vocab_size", str(config['vocab_size'])),
        info_row("Parameters", f"{total_params:,}"),
        info_row("Device",     str(device).upper()),
    ])
    st.markdown(arch_html, unsafe_allow_html=True)
    st.divider()

    # Eval Results
    eval_path = f"{EXP_DIR}/{selected_exp}/eval_results.json"
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            ev = json.load(f)
        st.markdown("#### Eval Results")
        ppl  = ev.get('test_perplexity', ev.get('perplexity', None))
        bleu = ev.get('bleu', None)
        r1   = ev.get('rouge1', ev.get('rouge_1', ev.get('ROUGE-1', None)))
        r2   = ev.get('rouge2', ev.get('rouge_2', ev.get('ROUGE-2', None)))
        rl   = ev.get('rougeL', ev.get('rouge_L', ev.get('ROUGE-L', None)))
        cards = ""
        if ppl  is not None: cards += metric_card("Perplexity ↓", f"{float(ppl):.2f}")
        if bleu is not None: cards += metric_card("BLEU %",       f"{float(bleu):.4f}", "blue")
        if r1   is not None: cards += metric_card("ROUGE-1 %",    f"{float(r1):.4f}",  "blue")
        if r2   is not None: cards += metric_card("ROUGE-2 %",    f"{float(r2):.4f}",  "blue")
        if rl   is not None: cards += metric_card("ROUGE-L %",    f"{float(rl):.4f}",  "blue")
        if cards:
            st.markdown(cards, unsafe_allow_html=True)
        st.divider()

    # Notes
    notes_path = f"{EXP_DIR}/{selected_exp}/notes.txt"
    if os.path.exists(notes_path):
        with open(notes_path) as f:
            notes = f.read().strip()
        st.markdown("#### Notes")
        st.markdown(f'<div class="notes-box">{notes}</div>', unsafe_allow_html=True)
        st.divider()

    # Generation Settings
    st.markdown("#### Generation")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.05,
                            help="Higher = more creative, Lower = more focused")
    top_p       = st.slider("Top-P (Nucleus)", 0.1, 1.0, 0.9, 0.05,
                            help="Cumulative probability threshold — 0.9 means sample from top 90% probability mass")
    max_tokens  = st.slider("Max Tokens",  50, 300, 150, 25)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="main-title">Medical<span class="accent-dot">GPT</span></div>
<div class="main-subtitle">
    Single-turn symptom-to-response generation &nbsp;·&nbsp; Not a conversational agent
    &nbsp;·&nbsp; Experiment: <strong>{selected_exp}</strong>
    &nbsp;·&nbsp; {total_params:,} parameters
    &nbsp;·&nbsp; Device: {str(device).upper()}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🩺</div>
        <div class="empty-text">Describe your symptoms below to get started</div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🩺"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe your symptoms..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Analyzing symptoms..."):
            response = generate_top_p(
                model, sp, config, device, prompt,
                max_tokens, temperature, top_p
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})