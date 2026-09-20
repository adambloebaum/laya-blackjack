const $ = (id) => document.getElementById(id);
const pct = (x, digits = 0) => `${(100 * x).toFixed(digits)}%`;
const signed = (x, digits = 2) => `${x >= 0 ? '+' : ''}${x.toFixed(digits)}`;
const escape = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let sessionId, data, busy = false, playing = false, timer, jobTimer;
let config = { players:3, decks:6, seed:42, samples:256 };
let checkpointAvailable = false;

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const message = typeof error.detail === 'string' ? error.detail : JSON.stringify(error.detail || response.statusText);
    throw new Error(message);
  }
  return response.json();
}
function notice(message, error = false) {
  $('notice').hidden = !message;
  $('notice').textContent = message;
  $('notice').classList.toggle('error', error);
}
function updateButtons() {
  $('step').disabled = busy || !sessionId;
  $('configure').disabled = busy;
  $('load-model').disabled = busy;
  $('autoplay').disabled = busy && !playing || !sessionId;
  $('autoplay').textContent = playing ? 'Ⅱ   Pause' : '▶   Play';
  $('play-dot').classList.toggle('playing', playing);
  $('step').textContent = busy ? 'Working…' : data?.state.phase === 'playing' ? 'Step →' : 'Deal →';
  document.querySelectorAll('.action-button').forEach(b => b.disabled = busy || playing);
}
function pause() { playing = false; clearTimeout(timer); updateButtons(); }
async function work(fn) {
  if (busy) return;
  busy = true; notice(''); updateButtons();
  try { await fn(); } catch (err) { pause(); notice(err.message, true); }
  finally { busy = false; updateButtons(); }
}
async function newSession() {
  pause();
  await work(async () => {
    const result = await api('/api/sessions', config);
    sessionId = result.id; data = result;
    localStorage.setItem('laya-session', sessionId);
    render();
  });
}
async function step(action = 'auto') {
  await work(async () => {
    data = await api(`/api/sessions/${sessionId}/step`, {action, policy:$('policy-select').value, revision:data.revision});
    render();
  });
  if (playing) timer = setTimeout(() => step(), data?.state.phase === 'playing' ? 750 : 1600);
}
function card(card) {
  if (card === '??') return '<div class="card back" aria-label="Concealed card">♠</div>';
  const rank = card.slice(0,-1), suit = card.slice(-1);
  return `<div class="card ${['♥','♦'].includes(suit) ? 'red' : ''}" aria-label="${escape(rank + ' ' + suit)}"><span>${escape(rank)}</span><span class="suit">${suit}</span><span class="big-suit" aria-hidden="true">${suit}</span></div>`;
}
function hand(h) {
  const label = h.natural ? 'BLACKJACK' : `${h.total}${h.soft ? ' soft' : ''}${h.status === 'bust' ? ' · bust' : ''}`;
  return `<div class="hand"><div class="cards">${h.cards.map(card).join('')}</div><span class="hand-score">${label}</span>${h.outcome ? `<span class="hand-result ${h.outcome}">${h.outcome.toUpperCase()} ${signed(h.profit,1)}u</span>` : `<span class="hand-result">${h.bet}u${h.status === 'stood' ? ' · stood' : h.status === 'surrendered' ? ' · surrendered' : ''}</span>`}</div>`;
}
function render() {
  const {state:s, reference:r, inference:m, history} = data;
  const isPlaying = s.phase === 'playing';
  const active = s.active[0];
  const net = $('net-return');
  net.innerHTML = `${signed(s.bankroll)} <small>units</small>`;
  net.className = s.bankroll < 0 ? 'negative-number' : 'positive-number';
  $('rounds').textContent = history.length;
  const positive = history.filter(x => x.profit > 0).length;
  $('round-note').textContent = history.length ? `${positive} positive · ${history.filter(x=>x.profit === 0).length} zero · ${history.filter(x=>x.profit < 0).length} negative` : 'Your experiment starts here';
  $('remaining').innerHTML = `${s.cards_remaining} <small>cards</small>`;
  $('shoe-note').textContent = `${s.rules.decks} decks · shuffle at ${pct(s.rules.penetration)} or reserve`;
  $('true-count').textContent = signed(s.true_count,1);
  $('count-note').textContent = `Running ${signed(s.running_count,0)} · ${s.cards_seen} exposed cards`;
  $('round-badge').textContent = `ROUND ${String(s.round).padStart(2,'0')}`;
  $('rule-summary').textContent = `${s.rules.decks}D · ${s.rules.hit_soft_17 ? 'H17' : 'S17'} · ${s.rules.double_after_split ? 'DAS' : 'NDAS'} · ${s.rules.blackjack_payout === 1.5 ? '3:2' : '6:5'}`;
  $('felt-payout').textContent = s.rules.blackjack_payout === 1.5 ? '3 TO 2' : '6 TO 5';
  $('felt-rule').textContent = `DEALER ${s.rules.hit_soft_17 ? 'HITS' : 'STANDS'} ON SOFT 17`;
  $('dealer').innerHTML = `<div class="seat-label">DEALER</div><div class="cards">${s.dealer.map(card).join('')}</div>${s.dealer_total === null ? '' : `<span class="hand-score">${s.dealer_total}${s.dealer_total>21 ? ' · bust' : ''}</span>`}`;
  const order = Array.from({length:s.players.length},(_,i)=>i).filter(i=>i!==0);
  order.splice(Math.floor(s.players.length/2),0,0);
  $('felt').classList.toggle('crowded',s.players.length >= 5);
  $('seats').innerHTML = order.map((seat,i) => {
    const x = order.length === 1 ? 50 : 11 + i * 78 / (order.length - 1);
    const y = order.length === 1 ? 64 : 51 + 12 * Math.sin(i / (order.length - 1) * Math.PI);
    return `<div class="seat ${seat===0?'hero':''} ${isPlaying&&active===seat?'active':''}" style="left:${x}%;top:${y}%"><div class="${s.players[seat].length>1?'split-hands':''}">${s.players[seat].map(hand).join('')}</div><div class="seat-label">${seat===0?'◈ PLAYER 01':`PLAYER ${String(seat+1).padStart(2,'0')}`}</div></div>`;
  }).join('');
  $('round-result').hidden = isPlaying;
  if (!isPlaying) $('round-result').textContent = `${s.phase === 'void' ? 'ROUND VOID' : 'PLAYER 01'} · ${signed(history.at(-1)?.profit || 0)} UNITS`;
  $('turn-label').textContent = isPlaying ? `${active===0 ? 'Player 01' : `Player ${String(active+1).padStart(2,'0')}`} · hand ${s.active[1]+1} to act` : 'Round complete · ready to deal';
  $('manual-actions').innerHTML = s.legal_actions.map(a=>`<button class="action-button ${r.recommendation===a?'recommended':''}" data-action="${a}">${a}</button>`).join('');
  $('manual-actions').querySelectorAll('button').forEach(b=>b.onclick=()=>step(b.dataset.action));
  $('events').innerHTML = data.events.map(e=>`<span>${escape(e)}</span>`).join('');
  renderChart(history);
  const max = Math.max(...s.unseen_counts,1);
  $('unseen-total').textContent = `${s.unseen_counts.reduce((a,b)=>a+b,0)} CARDS`;
  $('shoe-chart').innerHTML = s.unseen_counts.map((n,i)=>`<div class="shoe-bar"><b>${n}</b><i style="height:${n/max*57}px"></i><span>${i===0?'A':i+1}</span></div>`).join('');
  renderInference(r,m,s);
  renderReport(data.model.report);
  updateButtons();
}
function renderChart(history) {
  $('chart-caption').textContent = `${history.length} rounds · player 01`;
  if (!history.length) { $('chart').innerHTML='<div class="empty">Play a few rounds to trace the session.</div>'; return; }
  const values=[0,...history.map(h=>h.bankroll)], min=Math.min(-1,...values), max=Math.max(1,...values);
  const y = v => 88 - (v - min)/(max-min)*76;
  const points=values.map((v,i)=>`${i/(values.length-1)*500},${y(v)}`).join(' ');
  const color=values.at(-1)>=0?'#c8e9a6':'#df988c';
  $('chart').innerHTML=`<svg viewBox="0 0 500 100" preserveAspectRatio="none" role="img" aria-label="Session net units by round"><defs><linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${color}" stop-opacity=".13"/><stop offset="100%" stop-color="${color}" stop-opacity="0"/></linearGradient></defs><line x1="0" x2="500" y1="${y(0)}" y2="${y(0)}" stroke="#58654d" stroke-dasharray="4 4" stroke-width=".7"/><polygon points="0,100 ${points} 500,100" fill="url(#area)"/><polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>`;
}
function renderInference(r,m,s) {
  const loaded=data.model.loaded;
  $('model-name').textContent = loaded ? data.model.trained ? 'Laya · blackjack checkpoint' : 'Laya · base checkpoint' : 'Laya not loaded';
  $('model-dot').classList.toggle('inactive',!loaded);
  $('latency').textContent = m.available ? `${m.latency_ms.toFixed(0)} ms` : '— ms';
  $('load-model').textContent = loaded ? checkpointAvailable && !data.model.trained ? 'Load trained checkpoint ↗' : 'Reload checkpoint ↗' : checkpointAvailable ? 'Load trained Laya ↗' : 'Load base Laya ↗';
  $('model-message').textContent = loaded ? data.model.trained ? 'Simulator-distilled. Calibration metrics are in Experiments.' : 'Base checkpoint · not trained or calibrated for blackjack.' : 'Real model outputs appear after loading a checkpoint.';
  if (m.available) {
    const action=m.answers.action;
    $('model-actions').innerHTML=Object.entries(action.probabilities).map(([a,p])=>`<div class="model-row ${a===action.choice?'top':''}"><span>${a}</span><div class="model-track"><i style="width:${100*p}%"></i></div><b>${pct(p,1)}</b></div>`).join('');
  } else $('model-actions').innerHTML=`<div class="empty">${escape(m.reason || 'No active decision')}</div>`;
  $('sample-label').textContent=r.available ? `${r.samples} ROLLOUTS / ACTION` : 'REFERENCE';
  if (r.available) {
    $('action-values').innerHTML=Object.entries(r.actions).map(([a,v])=>`<div class="ev-row ${a===r.recommendation?'best':''}" title="95% MC interval: ${signed(v.ci95[0])} to ${signed(v.ci95[1])} units; voids ${pct(v.void,1)}"><span>${a}</span><div class="ev-track"><i class="${v.ev<0?'negative':''}" style="left:${v.ev<0?50-Math.min(Math.abs(v.ev)/4,1)*50:50}%;width:${Math.min(Math.abs(v.ev)/4,1)*50}%"></i></div><b>${signed(v.ev,3)}</b></div>`).join('');
    $('recommendation').textContent=`Reference prefers ${r.recommendation.toUpperCase()} · seat ${s.active[0]+1}`;
    $('bust-ref').textContent=pct(r.hit_bust,1);
    const outcome=r.actions[r.recommendation];
    $('outcome-bar').innerHTML=[outcome.positive,outcome.zero,outcome.negative].map(p=>`<i style="width:${p*100}%"></i>`).join('');
    $('outcomes').innerHTML=[[outcome.positive,'POSITIVE'],[outcome.zero,'ZERO'],[outcome.negative,'NEGATIVE']].map(([p,label])=>`<div><b>${pct(p,1)}</b>${label}</div>`).join('');
  } else {
    $('action-values').innerHTML='<div class="empty">Deal the next round to compare actions.</div>';
    $('recommendation').textContent='Round complete'; $('bust-ref').textContent='—'; $('outcome-bar').innerHTML=''; $('outcomes').innerHTML='';
  }
  $('bust-model').textContent=m.available ? pct(m.answers.hit_bust.noul,1) : '—';
  $('dealer-probs').innerHTML=['17','18','19','20','21','bust'].map(k=>{
    const p=r.available?r.dealer[k]:0, q=m.available?m.answers.dealer.probabilities[k]:0;
    return `<div class="prob-row"><span>${k}</span><div class="prob-track"><i style="width:${p*100}%"></i>${m.available?`<i class="model" style="width:${q*100}%"></i>`:''}</div><b>${r.available?pct(p):'—'}</b><em>${m.available?pct(q):'—'}</em></div>`;
  }).join('');
}
function renderReport(report) {
  if (!report) return;
  $('report-badge').textContent=`${report.states.train} TRAIN · ${report.states.test} TEST STATES`;
  const rows=[['Teacher agreement','teacher_agreement',true,true],['Teacher EV regret','teacher_ev_regret',false,false],['Next-hit probability Brier','hit_bust_brier',false,false],['Dealer probability Brier','dealer_brier',false,false],['Action calibration error','teacher_action_ece',false,false]];
  $('training-report').classList.remove('empty');
  $('training-report').innerHTML=`<table class="report-table"><thead><tr><th>Held-out metric</th><th>Base / raw</th><th>Trained / calibrated</th></tr></thead><tbody>${rows.map(([label,key,up,percent])=>{const a=report.baseline[key],b=report.test[key];return `<tr><td>${label}</td><td>${percent?pct(a,1):a.toFixed(4)}</td><td class="${(up?b>a:b<a)?'improved':'worse'}">${percent?pct(b,1):b.toFixed(4)}</td></tr>`;}).join('')}</tbody></table><p class="footnote">${escape(report.limitations)} ${report.states.validation} separate validation states used for temperature fitting.</p>`;
}
async function loadModel() {
  pause();
  await work(async()=>{
    notice('Loading Laya on the available device. The first load downloads model weights.');
    await api('/api/model/load',{source:checkpointAvailable?'trained':'base'});
    data=await api(`/api/sessions/${sessionId}`);render();notice('Checkpoint loaded. Model inference is live.');
  });
}
async function runJob(kind) {
  pause();
  try {
    const body=kind==='train'?{kind,states:Number($('train-states').value),epochs:Number($('train-epochs').value),samples:Number($('train-samples').value)}:{kind,rounds:Number($('benchmark-rounds').value),players:Number($('benchmark-players').value),samples:128};
    const job=await api('/api/jobs',body);
    localStorage.setItem('laya-job',job.id);pollJob(job.id);
  } catch(err) { notice(err.message,true); }
}
async function pollJob(id) {
  clearTimeout(jobTimer);
  try {
    const job=await api(`/api/jobs/${id}`);
    $('job-status').textContent=`${job.kind.toUpperCase()} · ${job.status.toUpperCase()}`;
    $('job-log').textContent=job.log;
    $('job-log').scrollTop=$('job-log').scrollHeight;
    const running=job.status==='running';
    $('train-button').disabled=running; $('benchmark-button').disabled=running;
    if (running) jobTimer=setTimeout(()=>pollJob(id),2500);
    else if (job.status==='failed') {notice(`Experiment failed: ${job.error}. See the console for details.`,true);localStorage.removeItem('laya-job');}
    else {
      if (job.kind==='train') {renderReport(job.result);checkpointAvailable=true;notice('Training complete. Load the trained checkpoint from the simulation sidebar.');if(data)renderInference(data.reference,data.inference,data.state);}
      else renderBenchmark(job.result);
      localStorage.removeItem('laya-job');
    }
  } catch(err) {notice(err.message,true);localStorage.removeItem('laya-job');$('train-button').disabled=false;$('benchmark-button').disabled=false;}
}
function renderBenchmark(report) {
  $('benchmark-results').innerHTML=`<table class="report-table"><thead><tr><th>Policy</th><th>Rounds</th><th>Mean net units</th><th>95% interval</th></tr></thead><tbody>${Object.entries(report.results).map(([name,r])=>`<tr><td>${escape(name)}</td><td>${r.rounds}</td><td>${signed(r.mean_units,3)}</td><td>${signed(r.ci95[0],3)} to ${signed(r.ci95[1],3)}</td></tr>`).join('')}</tbody></table><p class="footnote">${escape(report.sampling)}. ${report.rules.players} players; S17; 3:2; DAS; surrender.</p>`;
}
$('step').onclick=()=>step();
$('autoplay').onclick=()=>{if(playing){pause();return;}playing=true;updateButtons();step();};
$('configure').onclick=()=>{pause();$('settings').showModal();};
$('load-model').onclick=loadModel;
$('about-button').onclick=()=>{$('about').showModal();};
$('policy-select').onchange=()=>{pause();if($('policy-select').value==='laya'&&!data?.model.loaded)notice('Load Laya from the inference monitor before using the model policy.');};
document.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>$(b.dataset.close).close());
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{
  pause();document.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('active',x===b));
  $('simulation-view').hidden=b.dataset.view!=='simulation';$('experiments-view').hidden=b.dataset.view!=='experiments';
});
$('settings-form').onsubmit=async event=>{
  event.preventDefault();const values=Object.fromEntries(new FormData(event.target));config={};
  for(const [key,val] of Object.entries(values)) config[key]=['hit_soft_17','double_after_split','surrender'].includes(key)?val==='true':key==='tablemate_policy'?val:Number(val);
  $('settings').close();await newSession();
};
$('export').onclick=async()=>{
  if(!sessionId)return;
  try{const result=await api(`/api/sessions/${sessionId}/export`);const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`laya-session-${data.state.round}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(err){notice(err.message,true);}
};
$('train-button').onclick=()=>runJob('train');
$('benchmark-button').onclick=()=>runJob('benchmark');
async function init() {
  try {
    const status=await api('/api/model');checkpointAvailable=status.checkpoint_available;
    sessionId=localStorage.getItem('laya-session');
    if(sessionId){try{data=await api(`/api/sessions/${sessionId}`);render();}catch{sessionId=null;}}
    if(!sessionId)await newSession();
    if(data){for(const [key,val] of Object.entries(data.state.rules)){const field=$('settings-form').elements.namedItem(key);if(field)field.value=String(val);}}
    const job=localStorage.getItem('laya-job');if(job)pollJob(job);
  }catch(err){notice(`Could not connect to the laboratory: ${err.message}`,true);}
}
init();
