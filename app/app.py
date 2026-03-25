"""
app.py — Flask UI for the Model Router
Run: python app.py
Visit: http://localhost:5000
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, Response, stream_with_context

# Add project root to path so router imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from router.config_loader import load_config
from router.health_checker import check_health
from router.filter_engine import apply_filters
from router.urgency_adjuster import adjust_weights
from router.scorer import score_models

app = Flask(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def get_config():
    return load_config(CONFIG_PATH)

# ─────────────────────────────────────────────────────────────
# HTML Template
# ─────────────────────────────────────────────────────────────

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Router</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #0a0c10;
    --surface:  #111418;
    --border:   #1e2530;
    --accent:   #00e5ff;
    --accent2:  #7c3aed;
    --green:    #00ff88;
    --red:      #ff4444;
    --yellow:   #ffd600;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --code-bg:  #0d1117;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── Header ── */
  header {
    border-bottom: 1px solid var(--border);
    padding: 18px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    background: var(--surface);
  }
  .logo {
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--accent);
  }
  .logo span { color: var(--text); }
  .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    background: #00e5ff18;
    border: 1px solid #00e5ff33;
    color: var(--accent);
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 1px;
  }
  .health-bar {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  /* ── Main Layout ── */
  .main {
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 0;
    flex: 1;
    height: calc(100vh - 61px);
  }

  /* ── Left Panel ── */
  .left-panel {
    border-right: 1px solid var(--border);
    background: var(--surface);
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
  }

  /* Query form */
  textarea {
    width: 100%;
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    padding: 12px;
    resize: vertical;
    min-height: 100px;
    outline: none;
    transition: border-color 0.2s;
  }
  textarea:focus { border-color: var(--accent); }

  select, input[type="number"] {
    width: 100%;
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    padding: 10px 12px;
    outline: none;
    appearance: none;
    transition: border-color 0.2s;
    cursor: pointer;
  }
  select:focus, input:focus { border-color: var(--accent); }

  label {
    display: block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 6px;
  }

  .field { display: flex; flex-direction: column; }

  .btn-run {
    width: 100%;
    padding: 14px;
    background: var(--accent);
    color: #000;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    letter-spacing: 0.5px;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
  }
  .btn-run:hover { background: #33ecff; transform: translateY(-1px); }
  .btn-run:active { transform: translateY(0); }
  .btn-run:disabled { background: var(--border); color: var(--muted); cursor: not-allowed; transform: none; }

  /* Model cards in sidebar */
  .model-grid { display: flex; flex-direction: column; gap: 6px; }
  .model-card {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    transition: border-color 0.2s;
  }
  .model-card.winner {
    border-color: var(--green);
    background: #00ff8810;
  }
  .model-card.eliminated {
    opacity: 0.4;
    text-decoration: line-through;
  }
  .model-name { flex: 1; color: var(--text); font-weight: 600; }
  .model-score {
    color: var(--accent);
    font-size: 12px;
    font-weight: 700;
  }
  .model-badge {
    font-size: 9px;
    padding: 2px 7px;
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .badge-winner { background: #00ff8820; color: var(--green); border: 1px solid #00ff8840; }
  .badge-elim { background: #ff444420; color: var(--red); border: 1px solid #ff444440; }
  .badge-fallback { background: #ffd60020; color: var(--yellow); border: 1px solid #ffd60040; }

  /* ── Right Panel ── */
  .right-panel {
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* Steps tabs */
  .tabs {
    display: flex;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    padding: 0 24px;
    gap: 4px;
    overflow-x: auto;
  }
  .tab {
    padding: 14px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.5px;
    color: var(--muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 8px;
    user-select: none;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab .step-num {
    width: 18px; height: 18px;
    border-radius: 50%;
    background: var(--border);
    color: var(--muted);
    font-size: 10px;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.2s;
  }
  .tab.active .step-num { background: var(--accent); color: #000; }
  .tab.done .step-num { background: var(--green); color: #000; }
  .tab.done { color: var(--green); }

  /* Panel content */
  .panel-content {
    flex: 1;
    overflow-y: auto;
    padding: 28px;
  }

  .step-panel { display: none; }
  .step-panel.active { display: block; }

  /* ── Step cards ── */
  .step-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 16px;
    overflow: hidden;
  }
  .step-card-header {
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .step-card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--accent);
  }
  .step-card-body { padding: 18px; }

  /* Filter table */
  .filter-table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 12px; }
  .filter-table th {
    text-align: left;
    padding: 8px 12px;
    color: var(--muted);
    font-weight: 600;
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
  }
  .filter-table td { padding: 10px 12px; border-bottom: 1px solid #ffffff08; }
  .filter-table tr:hover td { background: #ffffff04; }
  .pass { color: var(--green); }
  .fail { color: var(--red); }

  /* Score bars */
  .score-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }
  .score-label { width: 130px; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .score-bar-wrap { flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
  .score-bar { height: 100%; border-radius: 3px; background: var(--accent); transition: width 0.8s cubic-bezier(0.4,0,0.2,1); }
  .score-bar.winner-bar { background: var(--green); }
  .score-val { width: 48px; text-align: right; color: var(--accent); font-weight: 700; }
  .winner-row .score-label { color: var(--green); font-weight: 700; }
  .winner-row .score-val { color: var(--green); }

  /* Weights display */
  .weights-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  .weight-box {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    text-align: center;
  }
  .weight-key { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--muted); margin-bottom: 6px; letter-spacing: 1px; text-transform: uppercase; }
  .weight-val { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 700; color: var(--accent); }
  .weight-change { font-family: 'JetBrains Mono', monospace; font-size: 10px; margin-top: 4px; }
  .weight-up { color: var(--green); }
  .weight-down { color: var(--red); }
  .weight-same { color: var(--muted); }

  /* Winner display */
  .winner-display {
    background: linear-gradient(135deg, #00ff8808, #00e5ff08);
    border: 1px solid #00ff8830;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin-bottom: 20px;
  }
  .winner-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; }
  .winner-name { font-family: 'Syne', sans-serif; font-size: 32px; font-weight: 800; color: var(--green); margin-bottom: 4px; }
  .winner-model-name { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--muted); margin-bottom: 12px; }
  .winner-score-big { font-family: 'JetBrains Mono', monospace; font-size: 48px; font-weight: 700; color: var(--accent); line-height: 1; }
  .winner-score-label { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--muted); margin-top: 4px; }

  /* Response panel */
  .response-box {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    line-height: 1.7;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
    min-height: 120px;
    position: relative;
  }
  .cursor {
    display: inline-block;
    width: 2px; height: 14px;
    background: var(--accent);
    animation: blink 1s infinite;
    vertical-align: middle;
    margin-left: 2px;
  }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

  /* Meta row */
  .meta-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 16px;
  }
  .meta-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 12px;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .meta-chip strong { color: var(--text); }

  /* Empty state */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--muted);
    gap: 12px;
    text-align: center;
    padding: 60px;
  }
  .empty-icon { font-size: 48px; opacity: 0.3; }
  .empty-title { font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 600; }
  .empty-sub { font-family: 'JetBrains Mono', monospace; font-size: 12px; line-height: 1.6; }

  /* Loading spinner */
  .spinner {
    display: inline-block;
    width: 14px; height: 14px;
    border: 2px solid #00000040;
    border-top-color: #000;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Tag chips */
  .tag-chip {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 2px 8px;
    background: #ffffff10;
    border-radius: 4px;
    color: var(--muted);
    margin: 2px;
  }

  /* Elimination reason */
  .elim-reason {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--red);
    opacity: 0.8;
  }

  /* Divider */
  .divider { height: 1px; background: var(--border); margin: 16px 0; }

  /* Info callout */
  .callout {
    border-left: 3px solid var(--accent);
    background: #00e5ff08;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    line-height: 1.6;
  }
  .callout strong { color: var(--text); }

  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--muted); }
</style>
</head>
<body>

<header>
  <div class="logo">Model<span>Router</span></div>
  <div class="tag">LLM RECOMMENDER</div>
  <div class="health-bar">
    <div class="dot"></div>
    <span id="health-status">Ollama connected</span>
  </div>
</header>

<div class="main">

  <!-- ── Left Panel ── -->
  <div class="left-panel">

    <div>
      <div class="section-title">Query</div>
      <div class="field">
        <textarea id="query" placeholder="Ask anything... e.g. Write a Python retry decorator"></textarea>
      </div>
    </div>

    <div>
      <div class="section-title">Hints</div>
      <div style="display:flex;flex-direction:column;gap:10px;">

        <div class="field">
          <label>USE CASE</label>
          <select id="use-case">
            <option value="">Auto-detect</option>
            <option value="chat">chat</option>
            <option value="code_generation">code_generation</option>
            <option value="reasoning">reasoning</option>
            <option value="summarization">summarization</option>
            <option value="rag_answer">rag_answer</option>
            <option value="data_extraction">data_extraction</option>
            <option value="routing_decision">routing_decision</option>
            <option value="long_context">long_context</option>
          </select>
        </div>

        <div class="field">
          <label>URGENCY</label>
          <select id="urgency">
            <option value="normal">normal</option>
            <option value="high">high — speed over quality</option>
            <option value="low">low — quality over speed</option>
          </select>
        </div>

        <div class="field">
          <label>INPUT TOKEN ESTIMATE (optional)</label>
          <input type="number" id="tokens" placeholder="e.g. 4096" min="0">
        </div>

      </div>
    </div>

    <button class="btn-run" id="run-btn">
      Run Query
    </button>

    <!-- Model result summary -->
    <div id="model-summary" style="display:none;">
      <div class="section-title">Model Ranking</div>
      <div class="model-grid" id="model-cards"></div>
    </div>

  </div>

  <!-- ── Right Panel ── -->
  <div class="right-panel">

    <div class="tabs">
      <div class="tab active" id="tab-0" >
        <div class="step-num">1</div> FILTER
      </div>
      <div class="tab" id="tab-1" >
        <div class="step-num">2</div> URGENCY
      </div>
      <div class="tab" id="tab-2" >
        <div class="step-num">3</div> SCORES
      </div>
      <div class="tab" id="tab-3" >
        <div class="step-num">4</div> WINNER
      </div>
      <div class="tab" id="tab-4" >
        <div class="step-num">5</div> RESPONSE
      </div>
    </div>

    <div class="panel-content">

      <!-- Step 1: Filter -->
      <div class="step-panel active" id="panel-0">
        <div class="empty-state" id="filter-empty">
          <div class="empty-icon">🔍</div>
          <div class="empty-title">Filter Engine</div>
          <div class="empty-sub">Run a query to see which models pass<br>health, quality, and context filters.</div>
        </div>
        <div id="filter-content" style="display:none;"></div>
      </div>

      <!-- Step 2: Urgency -->
      <div class="step-panel" id="panel-1">
        <div class="empty-state" id="urgency-empty">
          <div class="empty-icon">⚡</div>
          <div class="empty-title">Urgency Adjuster</div>
          <div class="empty-sub">See how urgency shifts the scoring weights<br>between quality, latency and cost.</div>
        </div>
        <div id="urgency-content" style="display:none;"></div>
      </div>

      <!-- Step 3: Scores -->
      <div class="step-panel" id="panel-2">
        <div class="empty-state" id="scores-empty">
          <div class="empty-icon">📊</div>
          <div class="empty-title">Scorer</div>
          <div class="empty-sub">Composite scores for all surviving models<br>with tag bonuses applied.</div>
        </div>
        <div id="scores-content" style="display:none;"></div>
      </div>

      <!-- Step 4: Winner -->
      <div class="step-panel" id="panel-3">
        <div class="empty-state" id="winner-empty">
          <div class="empty-icon">🏆</div>
          <div class="empty-title">Recommendation</div>
          <div class="empty-sub">The recommended model and why it won.</div>
        </div>
        <div id="winner-content" style="display:none;"></div>
      </div>

      <!-- Step 5: Response -->
      <div class="step-panel" id="panel-4">
        <div class="empty-state" id="response-empty">
          <div class="empty-icon">💬</div>
          <div class="empty-title">Model Response</div>
          <div class="empty-sub">The actual response from the selected model<br>will appear here after invocation.</div>
        </div>
        <div id="response-content" style="display:none;"></div>
      </div>

    </div>
  </div>
</div>

<script src="/static/router.js"></script>>
</body>
</html>'''



# JS served as static route to avoid triple-quote escaping issues
JS_CODE = r"""
let currentTab = 0;
let lastData = null;

function showTab(n) {
  document.querySelectorAll('.step-panel').forEach(function(p,i){ p.classList.toggle('active', i===n); });
  document.querySelectorAll('.tab').forEach(function(t,i){ t.classList.toggle('active', i===n); });
  currentTab = n;
}

async function runQuery() {
  const query = document.getElementById('query').value.trim();
  if (!query) { alert('Please enter a query.'); return; }

  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Running...';

  ['filter','urgency','scores','winner','response'].forEach(function(id) {
    document.getElementById(id+'-empty').style.display = '';
    document.getElementById(id+'-content').style.display = 'none';
  });
  document.querySelectorAll('.tab').forEach(function(t,i) {
    t.classList.remove('done');
    t.classList.toggle('active', i===0);
  });
  document.getElementById('model-summary').style.display = 'none';
  showTab(0);

  const hints = { urgency: document.getElementById('urgency').value };
  const uc = document.getElementById('use-case').value;
  if (uc) hints.use_case = uc;
  const tokens = document.getElementById('tokens').value;
  if (tokens) hints.input_token_estimate = parseInt(tokens);

  try {
    const resp = await fetch('/api/recommend', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ query: query, hints: hints })
    });
    const data = await resp.json();
    lastData = data;

    if (data.error) {
      alert('Error: ' + data.error);
      btn.disabled = false;
      btn.innerHTML = 'Run Query';
      return;
    }

    renderFilter(data);
    renderUrgency(data);
    renderScores(data);
    renderWinner(data);
    renderModelCards(data);

    showTab(4);
    document.getElementById('response-empty').style.display = 'none';
    document.getElementById('response-content').style.display = 'block';

    const metaRow = document.createElement('div');
    metaRow.className = 'meta-row';
    metaRow.innerHTML = ''
      + '<div class="meta-chip">USE CASE <strong>' + data.use_case + '</strong></div>'
      + '<div class="meta-chip">MODEL <strong>' + data.winner + '</strong></div>'
      + '<div class="meta-chip">SCORE <strong>' + data.winner_score.toFixed(3) + '</strong></div>'
      + '<div class="meta-chip">URGENCY <strong>' + hints.urgency + '</strong></div>';

    const box = document.createElement('div');
    box.className = 'response-box';
    box.innerHTML = '<span class="cursor"></span>';

    const rc = document.getElementById('response-content');
    rc.innerHTML = '';
    rc.appendChild(metaRow);
    rc.appendChild(box);

    await streamResponse(query, hints, data.winner, box);

    document.querySelectorAll('.tab').forEach(function(t) { t.classList.add('done'); });
    document.querySelectorAll('.tab').forEach(function(t,i) { t.classList.toggle('active', i===4); });

  } catch(e) {
    alert('Request failed: ' + e.message);
  }

  btn.disabled = false;
  btn.innerHTML = 'Run Query';
}

async function streamResponse(query, hints, winner, box) {
  try {
    const resp = await fetch('/api/invoke', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ query: query, hints: hints, model_id: winner })
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let text = '';

    while (true) {
      const result = await reader.read();
      if (result.done) break;
      const chunk = decoder.decode(result.value);
      const lines = chunk.split('\n');
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line.indexOf('data: ') === 0) {
          const payload = line.slice(6);
          if (payload === '[DONE]') break;
          try {
            const parsed = JSON.parse(payload);
            if (parsed.text) {
              text += parsed.text;
              box.textContent = text;
              const cursor = document.createElement('span');
              cursor.className = 'cursor';
              box.appendChild(cursor);
            }
            if (parsed.error) { box.textContent = 'Error: ' + parsed.error; }
          } catch(e) {}
        }
      }
    }
  } catch(e) {
    box.textContent = 'Failed to get response: ' + e.message;
  }
}

function renderFilter(data) {
  const el    = document.getElementById('filter-content');
  const empty = document.getElementById('filter-empty');

  let html = ''
    + '<div class="step-card">'
    +   '<div class="step-card-header"><div class="step-card-title">Health Check</div></div>'
    +   '<div class="step-card-body">'
    +     '<div class="callout"><strong>' + data.health_summary.online + '/' + data.health_summary.total + '</strong>'
    +     ' models online &mdash; offline models eliminated before scoring.</div>'
    +   '</div>'
    + '</div>'
    + '<div class="step-card">'
    +   '<div class="step-card-header"><div class="step-card-title">Filter Results</div></div>'
    +   '<div class="step-card-body">'
    +     '<table class="filter-table"><thead><tr>'
    +       '<th>Model</th><th>Quality</th><th>Context</th><th>Health</th><th>Status</th><th>Reason</th>'
    +     '</tr></thead><tbody>';

  for (let i = 0; i < data.all_models.length; i++) {
    const m    = data.all_models[i];
    const elim = data.eliminated[m.id];
    const pass = !elim;
    const nameColor  = pass ? 'var(--text)' : 'var(--muted)';
    const qClass     = pass ? 'pass' : 'fail';
    const hClass     = data.health[m.id] ? 'pass' : 'fail';
    const hMark      = data.health[m.id] ? '&#10003;' : '&#10007;';
    const badgeClass = pass ? 'badge-winner' : 'badge-elim';
    const badgeText  = pass ? 'PASS' : 'FAIL';
    html += '<tr>'
      + '<td style="font-weight:600;color:' + nameColor + '">' + m.id + '</td>'
      + '<td class="' + qClass + '">' + m.quality_score + '</td>'
      + '<td class="' + qClass + '">' + m.context_length.toLocaleString() + '</td>'
      + '<td class="' + hClass + '">' + hMark + '</td>'
      + '<td><span class="model-badge ' + badgeClass + '">' + badgeText + '</span></td>'
      + '<td class="elim-reason">' + (elim || '') + '</td>'
      + '</tr>';
  }

  html += '</tbody></table></div></div>';
  el.innerHTML = html;
  empty.style.display = 'none';
  el.style.display = 'block';
  document.getElementById('tab-0').classList.add('done');
}

function renderUrgency(data) {
  const el    = document.getElementById('urgency-content');
  const empty = document.getElementById('urgency-empty');
  const bw    = data.base_weights;
  const aw    = data.adjusted_weights;

  function changeBadge(b, a) {
    const diff = Math.round((a - b) * 100) / 100;
    if (Math.abs(diff) < 0.001) return '<span class="weight-change weight-same">unchanged</span>';
    if (diff > 0) return '<span class="weight-change weight-up">&#9650; +' + diff.toFixed(2) + '</span>';
    return '<span class="weight-change weight-down">&#9660; ' + diff.toFixed(2) + '</span>';
  }

  const urgencyDesc = {
    high:   'Latency boosted, quality reduced. Speed is critical.',
    normal: 'Weights unchanged — standard operation.',
    low:    'Quality boosted, latency reduced. Take time, be accurate.'
  };

  const html = ''
    + '<div class="step-card">'
    +   '<div class="step-card-header"><div class="step-card-title">Weight Adjustment &mdash; urgency: ' + data.urgency + '</div></div>'
    +   '<div class="step-card-body">'
    +     '<div class="callout" style="margin-bottom:20px;"><strong>urgency=' + data.urgency + '</strong> &mdash; '
    +     (urgencyDesc[data.urgency] || '') + '</div>'
    +     '<div style="margin-bottom:12px;font-family:JetBrains Mono,monospace;font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;">Base &rarr; Adjusted</div>'
    +     '<div class="weights-grid">'
    +       '<div class="weight-box"><div class="weight-key">Quality</div><div class="weight-val">' + aw.quality.toFixed(2) + '</div>' + changeBadge(bw.quality, aw.quality) + '</div>'
    +       '<div class="weight-box"><div class="weight-key">Latency</div><div class="weight-val">' + aw.latency.toFixed(2) + '</div>' + changeBadge(bw.latency, aw.latency) + '</div>'
    +       '<div class="weight-box"><div class="weight-key">Cost</div><div class="weight-val">'    + aw.cost.toFixed(2)    + '</div>' + changeBadge(bw.cost,    aw.cost)    + '</div>'
    +     '</div>'
    +   '</div>'
    + '</div>';

  el.innerHTML = html;
  empty.style.display = 'none';
  el.style.display = 'block';
  document.getElementById('tab-1').classList.add('done');
}

function renderScores(data) {
  const el    = document.getElementById('scores-content');
  const empty = document.getElementById('scores-empty');
  const aw    = data.adjusted_weights;

  let html = ''
    + '<div class="step-card">'
    +   '<div class="step-card-header"><div class="step-card-title">Composite Scores</div></div>'
    +   '<div class="step-card-body">'
    +     '<div class="callout" style="margin-bottom:20px;">'
    +     'Formula: <strong>(quality &times; ' + aw.quality.toFixed(2) + ') + (latency &times; ' + aw.latency.toFixed(2) + ') + (cost &times; ' + aw.cost.toFixed(2) + ') + tag_bonus</strong>'
    +     '</div>';

  const maxScore = Math.max.apply(null, data.scores.map(function(s){ return s.final_score; }));

  for (let i = 0; i < data.scores.length; i++) {
    const s        = data.scores[i];
    const isWinner = s.model_id === data.winner;
    const rowClass = isWinner ? 'score-row winner-row' : 'score-row';
    const barClass = isWinner ? 'score-bar winner-bar' : 'score-bar';
    const barWidth = (s.final_score / maxScore * 100).toFixed(1) + '%';
    const tagLine  = s.tag_bonus
      ? '<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:var(--muted);padding-left:142px;margin-top:-6px;margin-bottom:8px;">+tag bonus (' + s.matched_tags.join(', ') + ')</div>'
      : '';
    html += '<div class="' + rowClass + '">'
      + '<div class="score-label">' + s.model_id + '</div>'
      + '<div class="score-bar-wrap"><div class="' + barClass + '" style="width:' + barWidth + '"></div></div>'
      + '<div class="score-val">' + s.final_score.toFixed(4) + '</div>'
      + '</div>' + tagLine;
  }

  html += '</div></div>';
  el.innerHTML = html;
  empty.style.display = 'none';
  el.style.display = 'block';
  document.getElementById('tab-2').classList.add('done');
}

function renderWinner(data) {
  const el      = document.getElementById('winner-content');
  const empty   = document.getElementById('winner-empty');
  const runnerUp = data.scores.length > 1 ? data.scores[1] : null;

  let runnerHtml = '';
  if (runnerUp) {
    runnerHtml = '<div class="divider"></div>'
      + '<div style="font-family:JetBrains Mono,monospace;font-size:11px;color:var(--muted);">'
      + 'Runner-up: <strong style="color:var(--text)">' + runnerUp.model_id + '</strong>'
      + ' (score: ' + runnerUp.final_score.toFixed(3) + ','
      + ' gap: ' + (data.winner_score - runnerUp.final_score).toFixed(4) + ')'
      + '</div>';
  }

  let fallbackHtml = '';
  for (let i = 0; i < data.fallback_models.length; i++) {
    const m         = data.fallback_models[i];
    const isPrimary = m === data.winner;
    const badge     = isPrimary
      ? '<span class="model-badge badge-winner">PRIMARY</span>'
      : '<span class="model-badge badge-fallback">FALLBACK</span>';
    fallbackHtml += '<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--border);font-family:JetBrains Mono,monospace;font-size:12px;">'
      + '<span style="color:var(--muted);width:20px;">' + (i+1) + '.</span>'
      + '<span style="color:var(--text);flex:1;">' + m + '</span>'
      + badge + '</div>';
  }

  const html = ''
    + '<div class="winner-display">'
    +   '<div class="winner-label">Selected Model</div>'
    +   '<div class="winner-name">' + data.winner + '</div>'
    +   '<div class="winner-model-name">' + data.winner_model_name + '</div>'
    +   '<div class="winner-score-big">' + data.winner_score.toFixed(3) + '</div>'
    +   '<div class="winner-score-label">composite score</div>'
    + '</div>'
    + '<div class="step-card">'
    +   '<div class="step-card-header"><div class="step-card-title">Why this model?</div></div>'
    +   '<div class="step-card-body">'
    +     '<div class="callout">' + data.explanation.replace(/\n/g, '<br>') + '</div>'
    +     runnerHtml
    +   '</div>'
    + '</div>'
    + '<div class="step-card">'
    +   '<div class="step-card-header"><div class="step-card-title">Fallback Chain</div></div>'
    +   '<div class="step-card-body">'
    +     '<div style="font-family:JetBrains Mono,monospace;font-size:12px;color:var(--muted);margin-bottom:10px;">'
    +     'If <strong style="color:var(--text)">' + data.winner + '</strong>'
    +     ' fails &rarr; fallback group <strong style="color:var(--accent)">' + data.fallback_group + '</strong>'
    +     '</div>'
    +     fallbackHtml
    +   '</div>'
    + '</div>';

  el.innerHTML = html;
  empty.style.display = 'none';
  el.style.display = 'block';
  document.getElementById('tab-3').classList.add('done');
}

function renderModelCards(data) {
  const container = document.getElementById('model-cards');
  container.innerHTML = '';

  for (let i = 0; i < data.scores.length; i++) {
    const s        = data.scores[i];
    const isWinner = s.model_id === data.winner;
    const div      = document.createElement('div');
    div.className  = isWinner ? 'model-card winner' : 'model-card';
    const rank     = isWinner ? 'WINNER' : '#' + (i + 1);
    const badgeCls = isWinner ? 'model-badge badge-winner' : 'model-badge';
    div.innerHTML  = '<div class="model-name">' + s.model_id + '</div>'
      + '<div class="model-score">' + s.final_score.toFixed(3) + '</div>'
      + '<span class="' + badgeCls + '">' + rank + '</span>';
    container.appendChild(div);
  }

  const eliminated = Object.keys(data.eliminated);
  for (let i = 0; i < eliminated.length; i++) {
    const id  = eliminated[i];
    const div = document.createElement('div');
    div.className = 'model-card eliminated';
    div.innerHTML = '<div class="model-name">' + id + '</div>'
      + '<span class="model-badge badge-elim">OUT</span>';
    container.appendChild(div);
  }

  document.getElementById('model-summary').style.display = 'block';
}

document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('run-btn').addEventListener('click', runQuery);
  document.querySelectorAll('.tab').forEach(function(tab, i) {
    tab.addEventListener('click', function() { showTab(i); });
  });
  document.getElementById('query').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runQuery();
  });
});
"""


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────


@app.route('/.well-known/appspecific/com.chrome.devtools.json')
def devtools():
    return jsonify({}), 200

@app.route('/static/router.js')
def serve_js():
    return JS_CODE, 200, {'Content-Type': 'application/javascript'}

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """
    Runs the full pipeline up to (but not including) model invocation.
    Returns all intermediate steps for the UI to render.
    """
    try:
        body  = request.get_json()
        query = body.get('query', '')
        hints = body.get('hints', {})

        config = get_config()
        all_models = list(config.models.values())

        # Step 1 — Health check
        health_status = check_health(all_models, config.settings)
        online = sum(1 for v in health_status.values() if v)

        # Step 2 — Resolve use-case
        use_case_name = hints.get('use_case')
        auto_detected = False
        if not use_case_name:
            # Simple keyword-based auto-detection for UI demo
            q = query.lower()
            if any(w in q for w in ['write', 'code', 'function', 'debug', 'fix', 'implement', 'script']):
                use_case_name = 'code_generation'
            elif any(w in q for w in ['summarize', 'summary', 'condense', 'tldr']):
                use_case_name = 'summarization'
            elif any(w in q for w in ['analyze', 'analyse', 'explain', 'compare', 'trade-off', 'pros', 'cons']):
                use_case_name = 'reasoning'
            elif any(w in q for w in ['extract', 'parse', 'pull out', 'structured']):
                use_case_name = 'data_extraction'
            else:
                use_case_name = 'chat'
            auto_detected = True

        use_case = config.use_cases[use_case_name]

        # Step 3 — Filter
        filter_result = apply_filters(all_models, use_case, health_status, hints)

        # Step 4 — Urgency adjust
        urgency = hints.get('urgency', 'normal')
        base_weights = {
            'quality': use_case.weights.quality,
            'latency': use_case.weights.latency,
            'cost':    use_case.weights.cost,
        }
        adjusted = adjust_weights(use_case.weights, urgency)
        adjusted_weights = {'quality': adjusted.quality, 'latency': adjusted.latency, 'cost': adjusted.cost}

        # Step 5 — Score
        scores = score_models(filter_result.survivors, adjusted, use_case, config.settings)

        if not scores:
            # fallback group
            fb_group = config.groups[use_case.fallback_group]
            fb_models = [config.models[mid] for mid in fb_group.models if mid in config.models]
            scores = score_models(fb_models, adjusted, use_case, config.settings)

        winner = scores[0] if scores else None
        fallback_group_name = use_case.fallback_group
        fallback_models = config.groups[fallback_group_name].models

        # Build explanation
        if winner:
            explanation = f"✓ {winner.model_id} selected for '{use_case_name}' (score={winner.final_score:.3f})\n"
            if winner.tag_bonus_applied:
                explanation += f"  Tag bonus applied for: {winner.matched_tags}\n"
            if filter_result.eliminated:
                explanation += f"\nEliminated:\n"
                for mid, reason in list(filter_result.eliminated.items())[:4]:
                    explanation += f"  ✗ {mid}: {reason}\n"
        else:
            explanation = "No suitable model found."

        return jsonify({
            'use_case':          use_case_name,
            'auto_detected':     auto_detected,
            'urgency':           urgency,
            'winner':            winner.model_id if winner else None,
            'winner_model_name': config.models[winner.model_id].model_name if winner else '',
            'winner_score':      winner.final_score if winner else 0,
            'fallback_group':    fallback_group_name,
            'fallback_models':   fallback_models,
            'base_weights':      base_weights,
            'adjusted_weights':  adjusted_weights,
            'health':            health_status,
            'health_summary':    {'online': online, 'total': len(all_models)},
            'all_models': [
                {
                    'id':            m.id,
                    'quality_score': m.metadata.quality_score,
                    'latency_score': m.metadata.latency_score,
                    'context_length':m.metadata.context_length,
                    'tags':          m.metadata.tags,
                }
                for m in all_models
            ],
            'eliminated': filter_result.eliminated,
            'scores': [
                {
                    'model_id':    s.model_id,
                    'base_score':  s.base_score,
                    'final_score': s.final_score,
                    'tag_bonus':   s.tag_bonus_applied,
                    'matched_tags':s.matched_tags,
                }
                for s in scores
            ],
            'explanation': explanation,
        })

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})


@app.route('/api/invoke', methods=['POST'])
def api_invoke():
    """
    Invokes the winner model and streams the response back as SSE.
    """
    body     = request.get_json()
    query    = body.get('query', '')
    hints    = body.get('hints', {})

    def generate():
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import HumanMessage

            config   = get_config()
            model_id = body.get('model_id') or hints.get('use_case', 'chat')

            if model_id not in config.models:
                yield f"data: {json.dumps({'error': f'Model {model_id} not in config'})}\n\n"
                return

            model_cfg = config.models[model_id]
            llm = ChatOllama(
                model=model_cfg.model_name,
                temperature=model_cfg.parameters.get('temperature', 0.7),
                num_predict=model_cfg.parameters.get('max_tokens', 500),
            )

            full_text = ''
            for chunk in llm.stream([HumanMessage(content=query)]):
                if hasattr(chunk, 'content') and chunk.content:
                    full_text += chunk.content
                    yield f"data: {json.dumps({'text': chunk.content})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control':  'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


if __name__ == '__main__':
    print("Starting Model Router UI...")
    print("Visit: http://localhost:5001")
    app.run(debug=True, host='127.0.0.1', port=5001, threaded=True)