import { useState, useEffect, useRef } from "react";

// ── KaTeX loader ──────────────────────────────────────────────────────────────
const KATEX_CSS = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css";
const KATEX_JS  = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js";

function useKatex() {
  const [ready, setReady] = useState(!!window.katex);
  useEffect(() => {
    if (window.katex) { setReady(true); return; }
    if (!document.querySelector(`link[href="${KATEX_CSS}"]`)) {
      const link = document.createElement("link");
      link.rel = "stylesheet"; link.href = KATEX_CSS;
      document.head.appendChild(link);
    }
    const script = document.createElement("script");
    script.src = KATEX_JS;
    script.onload = () => setReady(true);
    document.head.appendChild(script);
  }, []);
  return ready;
}

function M({ children }) {
  const ref = useRef(null);
  const ready = useKatex();
  useEffect(() => {
    if (ready && ref.current) {
      try { window.katex.render(children, ref.current, { throwOnError: false, displayMode: false }); }
      catch(e) {}
    }
  }, [ready, children]);
  return <span ref={ref} />;
}

function D({ children, label }) {
  const ref = useRef(null);
  const ready = useKatex();
  useEffect(() => {
    if (ready && ref.current) {
      try { window.katex.render(children, ref.current, { throwOnError: false, displayMode: true }); }
      catch(e) {}
    }
  }, [ready, children]);
  return (
    <div className="math-block">
      {label && <div className="math-label">{label}</div>}
      <div ref={ref} className="math-display" />
    </div>
  );
}

function Box({ children, title }) {
  const ref = useRef(null);
  const ready = useKatex();
  useEffect(() => {
    if (ready && ref.current) {
      try { window.katex.render(children, ref.current, { throwOnError: false, displayMode: true }); }
      catch(e) {}
    }
  }, [ready, children]);
  return (
    <div className="boxed">
      {title && <div className="box-title">{title}</div>}
      <div ref={ref} className="math-display" />
    </div>
  );
}

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400;1,500&family=JetBrains+Mono:wght@400;500&family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300;1,400&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #f7f4ee; color: #1a1714; font-family: 'EB Garamond', Georgia, serif; font-size: 18px; line-height: 1.75; }
  :root {
    --ink: #1a1714; --ink-light: #4a4540; --ink-faint: #8a837a;
    --cream: #f7f4ee; --cream-dark: #ede8df; --accent: #8b2e12;
    --accent-light: #c4623e; --rule: #c8bfb0; --mono: 'JetBrains Mono', monospace;
  }
  .page { max-width: 780px; margin: 0 auto; padding: 60px 40px 120px; }

  .chapter-header { border-top: 3px solid var(--ink); padding-top: 32px; margin-bottom: 72px; }
  .chapter-label { font-family: var(--mono); font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--accent); margin-bottom: 20px; }
  .chapter-title { font-family: 'Cormorant Garamond', serif; font-size: 52px; font-weight: 300; line-height: 1.1; color: var(--ink); margin-bottom: 24px; letter-spacing: -0.01em; }
  .chapter-subtitle { font-size: 19px; font-style: italic; color: var(--ink-light); max-width: 580px; line-height: 1.6; border-left: 2px solid var(--accent); padding-left: 20px; }

  .section { margin-bottom: 64px; opacity: 0; transform: translateY(18px); transition: opacity 0.6s ease, transform 0.6s ease; }
  .section.visible { opacity: 1; transform: translateY(0); }
  .section-number { font-family: var(--mono); font-size: 11px; letter-spacing: 0.15em; color: var(--accent); text-transform: uppercase; margin-bottom: 8px; }
  h2 { font-family: 'Cormorant Garamond', serif; font-size: 32px; font-weight: 400; color: var(--ink); margin-bottom: 24px; line-height: 1.2; }
  h3 { font-family: 'EB Garamond', serif; font-size: 20px; font-weight: 600; color: var(--ink); margin: 32px 0 12px; font-style: italic; }
  p { margin-bottom: 16px; color: var(--ink-light); font-size: 18px; }
  hr.rule { border: none; border-top: 1px solid var(--rule); margin: 48px 0; }

  .math-block { background: white; border: 1px solid var(--rule); border-left: 3px solid var(--accent); border-radius: 2px; padding: 20px 28px; margin: 24px 0; overflow-x: auto; }
  .math-display { padding: 4px 0; }
  .math-label { font-family: var(--mono); font-size: 10px; color: var(--ink-faint); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 10px; }

  .boxed { background: #fdf8f2; border: 2px solid var(--ink); border-radius: 2px; padding: 24px 28px; margin: 28px 0; overflow-x: auto; }
  .boxed .box-title { font-family: var(--mono); font-size: 10px; letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent); margin-bottom: 14px; }

  code.inline { font-family: var(--mono); font-size: 14px; background: var(--cream-dark); padding: 1px 6px; border-radius: 2px; color: var(--accent); }

  .data-table { width: 100%; border-collapse: collapse; margin: 24px 0; font-size: 16px; }
  .data-table th { font-family: var(--mono); font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-faint); padding: 8px 16px; text-align: left; border-bottom: 2px solid var(--ink); }
  .data-table td { padding: 10px 16px; border-bottom: 1px solid var(--rule); color: var(--ink-light); font-variant-numeric: tabular-nums; }
  .data-table tr:last-child td { border-bottom: 2px solid var(--ink); font-weight: 500; color: var(--ink); }
  .data-table td.accent { color: var(--accent); font-weight: 600; }

  .callout { background: #fdf5f2; border: 1px solid #e8bfb0; border-radius: 2px; padding: 16px 20px; margin: 20px 0; font-size: 16px; color: var(--ink-light); }
  .callout strong { color: var(--accent); font-style: normal; }

  .widget { background: white; border: 1px solid var(--rule); border-radius: 4px; padding: 32px; margin: 32px 0; }
  .widget-title { font-family: var(--mono); font-size: 10px; letter-spacing: 0.15em; text-transform: uppercase; color: var(--ink-faint); margin-bottom: 20px; }
  .widget-controls { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 28px; align-items: flex-end; }
  .control-group { display: flex; flex-direction: column; gap: 6px; }
  .control-group label { font-family: var(--mono); font-size: 11px; color: var(--ink-faint); letter-spacing: 0.08em; }
  .control-group input[type=range] { width: 140px; accent-color: var(--accent); }
  .control-group .val { font-family: var(--mono); font-size: 13px; color: var(--accent); text-align: center; }
  .add-obs-btn { background: var(--ink); color: var(--cream); border: none; padding: 10px 20px; font-family: var(--mono); font-size: 12px; letter-spacing: 0.08em; cursor: pointer; border-radius: 2px; transition: background 0.15s; align-self: flex-end; }
  .add-obs-btn:hover { background: var(--accent); }
  .reset-btn { background: transparent; color: var(--ink-faint); border: 1px solid var(--rule); padding: 10px 16px; font-family: var(--mono); font-size: 11px; cursor: pointer; border-radius: 2px; transition: all 0.15s; align-self: flex-end; }
  .reset-btn:hover { border-color: var(--ink); color: var(--ink); }
  .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }
  @media (max-width: 640px) { .stats-grid { grid-template-columns: repeat(2, 1fr); } }
  .stat-card { background: var(--cream); border: 1px solid var(--rule); padding: 14px 16px; border-radius: 2px; }
  .stat-card .stat-label { font-family: var(--mono); font-size: 10px; color: var(--ink-faint); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; }
  .stat-card .stat-value { font-family: var(--mono); font-size: 20px; color: var(--ink); font-weight: 500; }
  .stat-card.highlight { background: #fdf5f2; border-color: var(--accent-light); }
  .stat-card.highlight .stat-value { color: var(--accent); }
  .obs-list { font-family: var(--mono); font-size: 12px; color: var(--ink-faint); margin-top: 12px; min-height: 20px; }
  .obs-tag { display: inline-block; background: var(--cream-dark); padding: 2px 8px; border-radius: 10px; margin: 2px 4px 2px 0; color: var(--ink-light); animation: fadein 0.3s ease; }
  @keyframes fadein { from { opacity:0; transform:scale(0.8); } to { opacity:1; transform:scale(1); } }
  .posterior-canvas-wrap { width: 100%; height: 200px; margin-top: 20px; position: relative; background: white; border: 1px solid var(--rule); border-radius: 2px; overflow: hidden; }
  canvas.posterior-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }

  .chapter-footer { margin-top: 80px; padding-top: 24px; border-top: 2px solid var(--ink); display: flex; justify-content: space-between; align-items: center; }
  .footer-label { font-family: var(--mono); font-size: 11px; color: var(--ink-faint); letter-spacing: 0.1em; }

  .math-display .katex-display { margin: 0; }
  .math-display .katex { font-size: 1.05em; }
`;

function gaussianPDF(x, mu, sigma) {
  return Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));
}

// Generalised Student-t: t_df(mu, scale^2)
// PDF ∝ (1 + (x-mu)^2 / (df * scale^2))^(-(df+1)/2)
// Normalisation constant: Γ((df+1)/2) / (Γ(df/2) * sqrt(df*π) * scale)
// We use log-gamma via Lanczos for accuracy
function logGamma(z) {
  // Lanczos approximation
  const g = 7;
  const c = [0.99999999999980993,676.5203681218851,-1259.1392167224028,
    771.32342877765313,-176.61502916214059,12.507343278686905,
    -0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function studentPDF(x, mu, scale, df) {
  // scale here is sqrt(beta_n / (alpha_n * nu_n))
  const z = (x - mu) / scale;
  const logNorm = logGamma((df + 1) / 2) - logGamma(df / 2) - 0.5 * Math.log(df * Math.PI) - Math.log(scale);
  const logKernel = -((df + 1) / 2) * Math.log(1 + z * z / df);
  return Math.exp(logNorm + logKernel);
}

function BayesUpdateWidget() {
  // NIG prior hyperparams: p(mu,sigma2) = N(mu | mu0, sigma2/nu0) * InvGamma(sigma2 | alpha0, beta0)
  const MU_0 = 36.2, NU_0 = 0.001, ALPHA_0 = 1, BETA_0 = 0.0625;

  const [newObs, setNewObs] = useState(38.2);
  const [observations, setObservations] = useState([38.2, 38.6, 38.1]);
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);

  const computePosterior = (obs) => {
    const n = obs.length;
    const xbar = obs.reduce((a, b) => a + b, 0) / n;
    const sumX = obs.reduce((a, b) => a + b, 0);
    const S = obs.reduce((a, x) => a + (x - xbar) ** 2, 0);

    // Standard NIG conjugate update
    const nu_n    = NU_0 + n;
    const mu_n    = (NU_0 * MU_0 + sumX) / nu_n;
    const alpha_n = ALPHA_0 + n / 2;
    const beta_n  = BETA_0 + 0.5 * S + (NU_0 * n * (xbar - MU_0) ** 2) / (2 * nu_n);

    // Marginal posterior of mu is a t-distribution; its std is:
    const sigma_mu = Math.sqrt(beta_n / (alpha_n * nu_n));
    // Posterior mean of sigma2:
    const sigma2_post = beta_n / (alpha_n - 1);

    // Lambda display values (exponential family accumulator)
    const lambda1 = NU_0 * MU_0 / BETA_0 + sumX;
    const lambda2 = -(NU_0 / (2 * BETA_0) + obs.reduce((a, x) => a + x * x, 0));

    return { mu: mu_n, sigma: sigma_mu, sigma2: sigma2_post, lambda1, lambda2, nu: nu_n, alpha_n, beta_n };
  };

  const post = observations.length > 0 ? computePosterior(observations) : null;
  const addObs = () => setObservations(prev => [...prev, newObs]);
  const removeObs = (i) => setObservations(prev => prev.filter((_, idx) => idx !== i));
  const reset = () => setObservations([38.2, 38.6, 38.1]);

  // draw receives current data as arguments to avoid stale closure issues
  const doDraw = (obs, mu, sigma, lambda1, lambda2, nu, alpha_n, beta_n) => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const W = wrap.clientWidth;
    const H = wrap.clientHeight;
    if (W === 0 || H === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const PAD_L = 10, PAD_R = 10, PAD_T = 24, PAD_B = 22;
    const plotW = W - PAD_L - PAD_R;
    const plotH = H - PAD_T - PAD_B;
    const xMin = 35.2, xMax = 40.3, range = xMax - xMin;
    const toX = v => PAD_L + ((v - xMin) / range) * plotW;
    // Student params: df=2*alpha_n, scale=sqrt(beta_n/(alpha_n*nu_n))
    const df = 2 * alpha_n;
    const scale = Math.sqrt(beta_n / (alpha_n * nu));

    // scale y so the Student peak fills ~80% of the plot height
    const peakStudent = studentPDF(mu, mu, scale, df);
    const yMax = peakStudent / 0.80;
    const toY = v => PAD_T + plotH - Math.min(v / yMax, 1.05) * plotH;

    // background
    ctx.fillStyle = "#fafaf8";
    ctx.fillRect(PAD_L, PAD_T, plotW, plotH);

    // grid verticals
    ctx.strokeStyle = "#e8e0d5"; ctx.lineWidth = 1;
    for (let t = 35.5; t <= 40.5; t += 0.5) {
      ctx.beginPath(); ctx.moveTo(toX(t), PAD_T); ctx.lineTo(toX(t), PAD_T + plotH); ctx.stroke();
    }

    // x axis
    ctx.strokeStyle = "#c8bfb0"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD_L, PAD_T + plotH); ctx.lineTo(PAD_L + plotW, PAD_T + plotH); ctx.stroke();

    // prior (wide, scaled for visibility)
    const priorSigma = 1.0;
    const priorPeak = gaussianPDF(MU_0, MU_0, priorSigma);
    const priorScale = (yMax * 0.35) / priorPeak;
    ctx.beginPath(); ctx.strokeStyle = "#b0a898"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    for (let px = 0; px <= plotW; px++) {
      const xv = xMin + (px / plotW) * range;
      const y = gaussianPDF(xv, MU_0, priorSigma) * priorScale;
      const ypx = PAD_T + plotH - (y / yMax) * plotH;
      px === 0 ? ctx.moveTo(PAD_L + px, ypx) : ctx.lineTo(PAD_L + px, ypx);
    }
    ctx.stroke(); ctx.setLineDash([]);

    // Student posterior fill
    ctx.beginPath();
    ctx.moveTo(PAD_L, PAD_T + plotH);
    for (let px = 0; px <= plotW; px++) {
      const xv = xMin + (px / plotW) * range;
      ctx.lineTo(PAD_L + px, toY(studentPDF(xv, mu, scale, df)));
    }
    ctx.lineTo(PAD_L + plotW, PAD_T + plotH);
    ctx.closePath();
    ctx.fillStyle = "rgba(139,46,18,0.10)"; ctx.fill();

    // Student posterior line (exact marginal)
    ctx.beginPath(); ctx.strokeStyle = "#8b2e12"; ctx.lineWidth = 2.5;
    for (let px = 0; px <= plotW; px++) {
      const xv = xMin + (px / plotW) * range;
      const ypx = toY(studentPDF(xv, mu, scale, df));
      px === 0 ? ctx.moveTo(PAD_L + px, ypx) : ctx.lineTo(PAD_L + px, ypx);
    }
    ctx.stroke();

    // Gaussian approximation (dashed, same mu/sigma for comparison)
    ctx.beginPath(); ctx.strokeStyle = "rgba(139,46,18,0.35)"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
    for (let px = 0; px <= plotW; px++) {
      const xv = xMin + (px / plotW) * range;
      const ypx = toY(gaussianPDF(xv, mu, sigma));
      px === 0 ? ctx.moveTo(PAD_L + px, ypx) : ctx.lineTo(PAD_L + px, ypx);
    }
    ctx.stroke(); ctx.setLineDash([]);

    // fever line 38°C
    const fx = toX(38);
    ctx.beginPath(); ctx.strokeStyle = "#c4623e"; ctx.lineWidth = 1.5; ctx.setLineDash([6, 3]);
    ctx.moveTo(fx, PAD_T); ctx.lineTo(fx, PAD_T + plotH); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#c4623e"; ctx.font = "bold 10px 'JetBrains Mono', monospace";
    ctx.fillText("38°C", fx + 4, PAD_T + 12);

    // obs ticks
    obs.forEach(xi => {
      const ox = toX(xi);
      ctx.beginPath(); ctx.strokeStyle = "rgba(139,46,18,0.5)"; ctx.lineWidth = 2;
      ctx.moveTo(ox, PAD_T + plotH - 4); ctx.lineTo(ox, PAD_T + plotH + 5); ctx.stroke();
    });

    // μ dot + label
    const muX = toX(mu);
    const muY = toY(peakStudent);
    ctx.beginPath(); ctx.arc(muX, muY, 5, 0, 2 * Math.PI);
    ctx.fillStyle = "#8b2e12"; ctx.fill();
    ctx.fillStyle = "#8b2e12"; ctx.font = "bold 11px 'JetBrains Mono', monospace";
    const labelX = muX + 8 + (muX > W * 0.75 ? -130 : 0);
    ctx.fillText(`μ = ${mu.toFixed(2)}°C`, labelX, muY + 4);

    // x labels
    ctx.font = "10px 'JetBrains Mono', monospace"; ctx.fillStyle = "#8a837a"; ctx.textAlign = "center";
    for (let t = 35.5; t <= 40; t += 0.5) {
      ctx.fillText(t.toFixed(1), toX(t), PAD_T + plotH + PAD_B - 4);
    }
    ctx.textAlign = "left";
  };

  // pass current values explicitly so no stale closure possible
  useEffect(() => {
    const id = setTimeout(() => {
      if (!post) return;
      doDraw(observations, post.mu, post.sigma, post.lambda1, post.lambda2, post.nu, post.alpha_n, post.beta_n);
    }, 30);
    return () => clearTimeout(id);
  }, [observations, post?.mu, post?.sigma]);

  // resize observer: call doDraw via a ref to always have fresh data
  const postRef = useRef(post);
  const obsRef = useRef(observations);
  useEffect(() => { postRef.current = post; obsRef.current = observations; });
  useEffect(() => {
    const ro = new ResizeObserver(() => {
      const p = postRef.current, o = obsRef.current;
      if (!p) return;
      doDraw(o, p.mu, p.sigma, p.lambda1, p.lambda2, p.nu, p.alpha_n, p.beta_n);
    });
    if (wrapRef.current) ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);

  const stats = post ? [
    { label: "n", value: observations.length, hi: false },
    { label: "λ₁", value: post.lambda1.toFixed(2), hi: false },
    { label: "λ₂", value: post.lambda2.toFixed(1), hi: false },
    { label: "ν", value: post.nu.toFixed(2), hi: false },
    { label: "μ post", value: `${post.mu.toFixed(2)}°C`, hi: true },
    { label: "df = 2α", value: (2 * post.alpha_n).toFixed(1), hi: false },
    { label: "σ prior", value: "0.25°C", hi: false },
    { label: "÷σ", value: `×${(0.25/post.sigma).toFixed(0)}`, hi: false },
  ] : [
    { label: "n", value: 0, hi: false },
    { label: "λ₁", value: "—", hi: false },
    { label: "λ₂", value: "—", hi: false },
    { label: "ν", value: "—", hi: false },
    { label: "μ post", value: `${MU_0}°C`, hi: true },
    { label: "df = 2α", value: (2 * ALPHA_0).toFixed(1), hi: false },
    { label: "σ prior", value: "0.25°C", hi: false },
    { label: "÷σ", value: "—", hi: false },
  ];

  return (
    <div className="widget">
      <div className="widget-title">Simulation interactive — mise à jour bayésienne</div>

      {/* Controls */}
      <div style={{display:"flex", gap:16, flexWrap:"wrap", alignItems:"flex-end", marginBottom:20}}>
        <div style={{display:"flex", flexDirection:"column", gap:6}}>
          <label style={{fontFamily:"var(--mono)", fontSize:11, color:"var(--ink-faint)", letterSpacing:"0.08em"}}>
            Nouvelle observation
          </label>
          <input type="range" min="35" max="41" step="0.1" value={newObs}
            onChange={e => setNewObs(parseFloat(e.target.value))}
            style={{width:160, accentColor:"var(--accent)"}} />
          <span style={{fontFamily:"var(--mono)", fontSize:13, color:"var(--accent)", textAlign:"center"}}>
            {newObs.toFixed(1)}°C
          </span>
        </div>
        <button className="add-obs-btn" onClick={addObs}>+ Ajouter</button>
        <button className="reset-btn" onClick={reset}>Réinitialiser</button>
      </div>

      {/* Stats grid */}
      <div style={{display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:10, marginBottom:16}}>
        {stats.map(s => (
          <div key={s.label} style={{
            background: s.hi ? "#fdf5f2" : "var(--cream)",
            border: `1px solid ${s.hi ? "var(--accent-light)" : "var(--rule)"}`,
            padding:"10px 12px", borderRadius:2
          }}>
            <div style={{fontFamily:"var(--mono)", fontSize:9, color:"var(--ink-faint)", letterSpacing:"0.1em", textTransform:"uppercase", marginBottom:4}}>
              {s.label}
            </div>
            <div style={{fontFamily:"var(--mono)", fontSize:16, fontWeight:500, color: s.hi ? "var(--accent)" : "var(--ink)"}}>
              {s.value}
            </div>
          </div>
        ))}
      </div>

      {/* Observations tags — click to remove */}
      <div style={{fontFamily:"var(--mono)", fontSize:12, marginBottom:12, minHeight:24}}>
        <span style={{color:"var(--ink-faint)", marginRight:6}}>observations (clic pour supprimer) :</span>
        {observations.map((x, i) => (
          <span
            key={i}
            className="obs-tag"
            onClick={() => removeObs(i)}
            style={{cursor:"pointer", userSelect:"none"}}
            title="Cliquer pour supprimer"
          >
            {x.toFixed(1)}°C ×
          </span>
        ))}
      </div>

      {/* Canvas */}
      <div ref={wrapRef} className="posterior-canvas-wrap">
        <canvas ref={canvasRef} className="posterior-canvas" />
      </div>

      <div style={{fontSize:11, color:"var(--ink-faint)", fontFamily:"var(--mono)", marginTop:8}}>
        — — prior (large) &nbsp;&nbsp; —— Student t(2α) exact &nbsp;&nbsp; - - Gaussienne approx &nbsp;&nbsp; | 38°C &nbsp;&nbsp; ↓ obs
      </div>
    </div>
  );
}

function Section({ number, title, children }) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) setVisible(true); }, { threshold: 0.08 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return (
    <div ref={ref} className={`section ${visible?"visible":""}`}>
      <div className="section-number">{number}</div>
      <h2>{title}</h2>
      {children}
    </div>
  );
}

export default function Chapter() {
  return (
    <>
      <style>{styles}</style>
      <div className="page">

        <div className="chapter-header">
          <div className="chapter-label">Inférence Bayésienne · Chapitre 4</div>
          <h1 className="chapter-title">Le Prior Conjugué<br />dans les Familles Exponentielles</h1>
          <p className="chapter-subtitle">
            De la structure abstraite d'une famille exponentielle jusqu'à l'estimation bayésienne
            de la température d'un patient fébrile — un fil conducteur unique.
          </p>
        </div>

        {/* §1 */}
        <Section number="§ 1" title="La famille exponentielle">
          <p>Une distribution appartient à la <em>famille exponentielle</em> si sa densité s'écrit sous la forme canonique :</p>
          <D label="Définition — forme canonique">{`p(x \\mid \\eta) = h(x)\\,\\exp\\!\\bigl(\\eta^\\top T(x) - A(\\eta)\\bigr)`}</D>
          <p>Les trois ingrédients : <M>{`h(x)`}</M> la mesure de base, <M>{`T(x)`}</M> la <em>statistique suffisante</em>, et <M>{`\\eta`}</M> le <em>paramètre naturel</em>. La <em>log-partition</em> <M>{`A(\\eta)`}</M> assure la normalisation :</p>
          <D>{`A(\\eta) = \\log \\int h(x)\\,\\exp\\!\\bigl(\\eta^\\top T(x)\\bigr)\\,dx`}</D>
          <p>Ses dérivées donnent les moments de <M>{`T(x)`}</M> :</p>
          <D>{`\\nabla_\\eta A(\\eta) = \\mathbb{E}_\\eta[T(x)], \\qquad \\nabla^2_\\eta A(\\eta) = \\operatorname{Var}_\\eta[T(x)]`}</D>
          <div className="callout">
            <strong>Intuition clé.</strong> Toute l'information sur <M>{`\\eta`}</M> contenue dans une observation <M>{`x`}</M> est capturée par <M>{`T(x)`}</M>. Deux observations avec le même <M>{`T(x)`}</M> sont <em>équivalentes</em> pour l'inférence.
          </div>
        </Section>

        <hr className="rule" />

        {/* §2 */}
        <Section number="§ 2" title="Vraisemblance de n observations">
          <p>Pour <M>{`n`}</M> observations i.i.d., la vraisemblance se factorise élégamment :</p>
          <D>{`p(x_1,\\dots,x_n \\mid \\eta) = \\prod_i h(x_i) \\cdot \\exp\\!\\Bigl(\\eta^\\top \\textstyle\\sum_i T(x_i) - n\\,A(\\eta)\\Bigr)`}</D>
          <p>La vraisemblance ne dépend des données qu'à travers <M>{`\\sum_i T(x_i)`}</M>. C'est le théorème de suffisance de Fisher — ce vecteur de dimension fixe résume intégralement l'information de <M>{`n`}</M> observations, quelle que soit <M>{`n`}</M>.</p>
        </Section>

        <hr className="rule" />

        {/* §3 */}
        <Section number="§ 3" title="Construction du prior conjugué joint">
          <p>On cherche un prior <M>{`p(\\eta)`}</M> tel que le posterior reste dans la <em>même famille paramétrique</em>. La forme conjuguée naturelle est :</p>
          <Box title="Théorème — Prior conjugué universel">{`p(\\eta \\mid \\lambda_0, \\nu_0) \\propto \\exp\\!\\bigl(\\eta^\\top \\lambda_0 - \\nu_0\\,A(\\eta)\\bigr), \\quad \\lambda_0 \\in \\mathbb{R}^k,\\ \\nu_0 > 0`}</Box>
          <p>Ce prior est normalisable si et seulement si <M>{`\\nu_0 > 0`}</M> et <M>{`\\lambda_0/\\nu_0`}</M> est dans l'intérieur du domaine naturel de <M>{`\\eta`}</M>.</p>

          <h3>Interprétation des hyperparamètres</h3>
          <table className="data-table">
            <thead><tr><th>Hyperparamètre</th><th>Interprétation</th><th>Analogie</th></tr></thead>
            <tbody>
              <tr><td><M>{`\\lambda_0`}</M></td><td>Statistiques suffisantes fictives a priori</td><td>Billes initiales dans une urne</td></tr>
              <tr><td><M>{`\\nu_0`}</M></td><td>Poids du prior (observations fictives)</td><td>Nombre de billes initiales</td></tr>
              <tr><td><M>{`\\lambda_0/\\nu_0`}</M></td><td>Valeur centrale a priori de <M>{`\\eta`}</M></td><td>Moyenne des billes initiales</td></tr>
            </tbody>
          </table>
          <div className="callout">
            <strong><M>{`\\lambda`}</M> comme mémoire suffisante.</strong> <M>{`\\lambda`}</M> est une <em>urne</em> qui accumule les statistiques suffisantes. <M>{`\\nu`}</M> est son poids. <M>{`\\lambda/\\nu`}</M> est sa moyenne. Le prior initialise l'urne avec <M>{`\\nu_0`}</M> billes fictives ; chaque observation verse <M>{`T(x_i)`}</M> dans l'urne et incrémente <M>{`\\nu`}</M> d'une unité.
          </div>
        </Section>

        <hr className="rule" />

        {/* §4 */}
        <Section number="§ 4" title="La mise à jour bayésienne — deux additions">
          <p>Le posterior s'obtient par le théorème de Bayes :</p>
          <D>{`p(\\eta \\mid \\mathbf{x}) \\propto \\exp\\!\\Bigl(\\eta^\\top\\textstyle\\sum_i T(x_i) - n A(\\eta)\\Bigr)\\cdot\\exp\\!\\bigl(\\eta^\\top\\lambda_0 - \\nu_0 A(\\eta)\\bigr)`}</D>
          <D>{`= \\exp\\!\\Bigl(\\eta^\\top\\underbrace{\\bigl(\\lambda_0 + \\textstyle\\sum_i T(x_i)\\bigr)}_{\\lambda_n} - \\underbrace{(\\nu_0+n)}_{\\nu_n}\\,A(\\eta)\\Bigr)`}</D>
          <Box title="Règle de mise à jour — universelle">{`\\lambda_n = \\lambda_0 + \\sum_{i=1}^n T(x_i), \\qquad \\nu_n = \\nu_0 + n`}</Box>
          <p>C'est la beauté de la conjugaison : toute la complexité bayésienne se réduit à deux additions.</p>

          <h3>Structure de shrinkage</h3>
          <p>Le ratio <M>{`\\lambda_n/\\nu_n`}</M> interpole entre prior et données :</p>
          <D>{`\\frac{\\lambda_n}{\\nu_n} = \\frac{\\nu_0}{\\nu_0+n}\\cdot\\frac{\\lambda_0}{\\nu_0} + \\frac{n}{\\nu_0+n}\\cdot\\bar{T}`}</D>
          <p>Quand <M>{`n\\to\\infty`}</M>, les données écrasent le prior (<M>{`\\lambda_n/\\nu_n \\to \\bar{T}`}</M>) et on retrouve l'estimateur fréquentiste.</p>
        </Section>

        <hr className="rule" />

        {/* §5 */}
        <Section number="§ 5" title="Application : la loi gaussienne">
          <p>On pose <M>{`X \\sim \\mathcal{N}(\\mu, \\sigma^2)`}</M>. La forme canonique donne :</p>
          <D label="Identification des composantes">{`h(x) = \\tfrac{1}{\\sqrt{2\\pi}}, \\quad T(x) = \\begin{pmatrix}x \\\\ x^2\\end{pmatrix}, \\quad \\eta = \\begin{pmatrix}\\mu/\\sigma^2 \\\\ -1/(2\\sigma^2)\\end{pmatrix}`}</D>
          <D>{`A(\\eta) = -\\frac{\\eta_1^2}{4\\eta_2} + \\frac{1}{2}\\log\\!\\left(-\\frac{\\pi}{\\eta_2}\\right)`}</D>
          <p>La bijection s'inverse :</p>
          <D>{`\\sigma^2 = -\\frac{1}{2\\eta_2}, \\qquad \\mu = -\\frac{\\eta_1}{2\\eta_2}`}</D>
          <p>En revenant aux paramètres usuels, le prior conjugué se réduit à la loi <strong>Normale-Gamma Inverse</strong> <M>{`\\mathcal{NIG}(\\mu_0, \\nu_0, \\alpha_0, \\beta_0)`}</M>.</p>

          <h3>Statistiques suffisantes gaussiennes</h3>
          <table className="data-table">
            <thead><tr><th>Composante</th><th>Statistique</th><th>Information portée</th></tr></thead>
            <tbody>
              <tr><td><M>{`T^{(1)}(x) = x`}</M></td><td>valeur brute</td><td>Localisation → <M>{`\\mu`}</M></td></tr>
              <tr><td><M>{`T^{(2)}(x) = x^2`}</M></td><td>carré</td><td>Dispersion → <M>{`\\sigma^2`}</M></td></tr>
              <tr><td><M>{`\\sum x_i,\\ \\sum x_i^2`}</M></td><td>sommes</td><td>Tout ce qu'il faut pour inférer <M>{`(\\mu, \\sigma^2)`}</M></td></tr>
            </tbody>
          </table>
        </Section>

        <hr className="rule" />

        {/* §6 */}
        <Section number="§ 6" title="Cas pratique : thermométrie bayésienne">
          <p>On mesure la température frontale d'un patient avec un thermomètre d'incertitude <M>{`\\pm 0.5^\\circ`}</M>C, et un prior épidémiologique très faible (<M>{`\\nu_0 = 0.001`}</M>).</p>

          <h3>Calibration du prior</h3>
          <p>L'incertitude <M>{`\\pm 0.5^\\circ`}</M>C correspond à environ <M>{`2\\sigma`}</M>, donc <M>{`\\sigma \\approx 0.25^\\circ\\text{C} \\Rightarrow \\sigma^2 \\approx 0.0625`}</M>. Avec <M>{`\\mu_0 = 36.2^\\circ\\text{C}`}</M> :</p>
          <D>{`\\lambda_0 = \\nu_0 \\cdot \\begin{pmatrix}\\mu_0/\\sigma_0^2 \\\\ -1/(2\\sigma_0^2)\\end{pmatrix} = \\begin{pmatrix}0.579 \\\\ -0.008\\end{pmatrix}, \\qquad \\nu_0 = 0.001`}</D>

          <h3>Trois mesures, deux additions</h3>
          <table className="data-table">
            <thead><tr><th>Mesure</th><th><M>{`x_i`}</M> (°C)</th><th><M>{`x_i^2`}</M></th></tr></thead>
            <tbody>
              <tr><td>1</td><td>38.2</td><td>1459.24</td></tr>
              <tr><td>2</td><td>38.6</td><td>1490.00</td></tr>
              <tr><td>3</td><td>38.1</td><td>1451.61</td></tr>
              <tr><td><strong>Σ</strong></td><td className="accent"><strong>114.9</strong></td><td className="accent"><strong>4400.85</strong></td></tr>
            </tbody>
          </table>
          <Box title="Mise à jour">{`\\lambda_3 = \\begin{pmatrix}0.579 + 114.9 \\\\ -0.008 - 4400.85\\end{pmatrix} = \\begin{pmatrix}115.479 \\\\ -4400.858\\end{pmatrix}, \\quad \\nu_3 = 3.001`}</Box>

          <h3>Retour vers <M>{`(\\mu, \\sigma^2)`}</M></h3>
          <D>{`\\eta_1^{\\text{post}} = \\frac{115.479}{3.001} = 38.480, \\qquad \\eta_2^{\\text{post}} = \\frac{-4400.858}{3.001} = -1466.46`}</D>
          <D>{`\\sigma^2_{\\text{post}} = -\\frac{1}{2\\,\\eta_2^{\\text{post}}} \\approx 3.41\\times 10^{-4} \\implies \\sigma_{\\text{post}} \\approx 0.018^\\circ\\text{C}`}</D>
          <D>{`\\mu_{\\text{post}} = \\eta_1^{\\text{post}}\\cdot\\sigma^2_{\\text{post}} \\approx 38.30^\\circ\\text{C}`}</D>
          <div className="callout">
            <strong>Réduction d'incertitude.</strong> Trois mesures ont réduit <M>{`\\sigma`}</M> de <M>{`0.25^\\circ`}</M>C à <M>{`0.018^\\circ`}</M>C — un facteur ×14. Le patient est clairement fébrile avec une certitude quasi-totale.
          </div>
        </Section>

        <hr className="rule" />

        {/* §7 */}
        <Section number="§ 7" title="Simulation interactive">
          <p>Ajoutez des mesures et observez la mise à jour bayésienne en temps réel. Les hyperparamètres <M>{`\\lambda`}</M> et <M>{`\\nu`}</M> s'accumulent ; le posterior (courbe rouge) se concentre autour de la vraie température.</p>
          <BayesUpdateWidget />
        </Section>

        <hr className="rule" />

        {/* §8 */}
        <Section number="§ 8" title="Synthèse — ce que la structure révèle">
          <p>La conjugaison dans les familles exponentielles n'est pas une coïncidence algébrique : c'est une conséquence directe du théorème de suffisance. La vraisemblance ne dépend des données qu'à travers <M>{`T(x)`}</M> ; le prior conjugué a <em>la même dépendance fonctionnelle</em> en <M>{`\\eta`}</M> via <M>{`A(\\eta)`}</M>.</p>
          <table className="data-table">
            <thead><tr><th>Objet</th><th>Rôle</th><th>Intuition</th></tr></thead>
            <tbody>
              <tr><td><M>{`T(x)`}</M></td><td>Statistique suffisante</td><td>Ce que <M>{`x`}</M> apprend sur <M>{`\\eta`}</M></td></tr>
              <tr><td><M>{`A(\\eta)`}</M></td><td>Log-partition</td><td>Géométrie de la famille</td></tr>
              <tr><td><M>{`\\lambda`}</M></td><td>Mémoire accumulée</td><td>Urne de statistiques suffisantes</td></tr>
              <tr><td><M>{`\\nu`}</M></td><td>Poids de la mémoire</td><td>Compteur d'observations</td></tr>
              <tr><td><M>{`\\lambda/\\nu`}</M></td><td>Estimateur courant</td><td>Moyenne de l'urne (shrinkage)</td></tr>
              <tr><td><M>{`\\lambda_n = \\lambda_0 + \\sum T(x_i)`}</M></td><td>Mise à jour bayésienne</td><td>Deux additions, c'est tout</td></tr>
            </tbody>
          </table>
          <div className="callout">
            <strong>Le résultat le plus profond.</strong> Dans la limite <M>{`\\nu_0 \\to 0`}</M> (prior non-informatif), les estimateurs bayésiens convergent vers les estimateurs du maximum de vraisemblance — le fréquentisme émerge comme cas limite du bayésien. La structure des familles exponentielles rend cette convergence transparente.
          </div>
        </Section>

        <div className="chapter-footer">
          <span className="footer-label">Inférence Bayésienne · Chapitre 4</span>
          <span className="footer-label">Prior Conjugué · Familles Exponentielles</span>
        </div>

      </div>
    </>
  );
}