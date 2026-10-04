"""theme.py - visual layer for the PSX AI dashboard (CSS, SVG icons, hero widgets). No emoji anywhere."""

LOGO = ('<svg viewBox="0 0 32 32" width="30" height="30"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#00f0ff"/><stop offset="1" stop-color="#ff2bd6"/></linearGradient></defs>'
        '<rect x="1.5" y="1.5" width="29" height="29" rx="9" fill="#0b0f1f" stroke="url(#lg)" stroke-width="1.6"/>'
        '<path d="M6 22 L12 15 L17 19 L26 8" fill="none" stroke="url(#lg)" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="26" cy="8" r="2.4" fill="#fff"/></svg>')

UP = '<svg class="ar" viewBox="0 0 10 10"><path d="M5 1 L9.5 9 H0.5Z" fill="currentColor"/></svg>'
DN = '<svg class="ar" viewBox="0 0 10 10"><path d="M5 9 L9.5 1 H0.5Z" fill="currentColor"/></svg>'


def gauge(conf: float, color: str) -> str:
    """Animated semicircle confidence gauge (pure SVG + CSS)."""
    ang = -90 + max(0.0, min(1.0, conf)) * 180
    return (
        '<svg class="gauge" viewBox="0 0 200 118"><defs><linearGradient id="gg" x1="0" x2="1">'
        '<stop offset="0" stop-color="#ff3d71"/><stop offset=".5" stop-color="#ffc645"/><stop offset="1" stop-color="#19ffa3"/></linearGradient>'
        '<filter id="gl"><feGaussianBlur stdDeviation="3.2"/></filter></defs>'
        '<path d="M14 102A86 86 0 0 1 186 102" fill="none" stroke="url(#gg)" stroke-width="12" stroke-linecap="round" filter="url(#gl)" opacity=".75"/>'
        '<path d="M14 102A86 86 0 0 1 186 102" fill="none" stroke="url(#gg)" stroke-width="10" stroke-linecap="round"/>'
        f'<g class="needle" style="--a:{ang:.1f}deg"><line x1="100" y1="102" x2="100" y2="32" stroke="#fff" stroke-width="3" stroke-linecap="round"/>'
        f'<circle cx="100" cy="102" r="8" fill="#0b0f1f" stroke="{color}" stroke-width="3"/></g></svg>')


def ticker(items: list) -> str:
    """items: [(code, price, pct)] -> infinite marquee."""
    if not items:
        return ""
    cell = "".join(
        f'<span class="tk"><b>{c}</b>{p:,.2f}<i class="{"u" if g >= 0 else "d"}">{UP if g >= 0 else DN}{abs(g):.2f}%</i></span>'
        for c, p, g in items)
    return f'<div class="tape"><div class="tape-in">{cell}{cell}</div></div>'


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&family=JetBrains+Mono:wght@500;700&display=swap');
:root{--bg:#05060f;--card:rgba(16,20,42,.55);--line:rgba(120,140,255,.16);--mut:#8a93b8;--cy:#00f0ff;--mg:#ff2bd6;--vi:#7c5cff;
--up:#19ffa3;--dn:#ff3d71;--am:#ffc645;--acc:#00f0ff}
html,body,[data-testid="stApp"]{background:var(--bg)!important;color:#eaf0ff;font-family:'Sora',-apple-system,'Segoe UI',sans-serif;-webkit-tap-highlight-color:transparent}
[data-testid="stApp"]::before{content:"";position:fixed;inset:-20%;z-index:0;pointer-events:none;filter:blur(70px);opacity:.9;
 background:radial-gradient(40% 35% at 15% 20%,rgba(0,240,255,.28),transparent 70%),radial-gradient(35% 40% at 85% 15%,rgba(255,43,214,.26),transparent 70%),
 radial-gradient(45% 40% at 70% 90%,rgba(124,92,255,.30),transparent 70%);animation:aur 18s ease-in-out infinite alternate}
[data-testid="stApp"]::after{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.55;
 background:linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px) 0 0/100% 44px,linear-gradient(90deg,rgba(255,255,255,.03) 1px,transparent 1px) 0 0/44px 100%;
 -webkit-mask-image:radial-gradient(ellipse at 50% 20%,#000,transparent 75%);mask-image:radial-gradient(ellipse at 50% 20%,#000,transparent 75%)}
@keyframes aur{to{transform:translate3d(-4%,5%,0) rotate(8deg) scale(1.15)}}
header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],footer,#MainMenu{display:none!important}
.block-container{position:relative;z-index:1;max-width:760px!important;padding:max(.6rem,env(safe-area-inset-top)) .85rem 7.5rem!important}
hr{border-color:var(--line)!important}
.tape{margin:0 -.85rem 12px;overflow:hidden;white-space:nowrap;border-block:1px solid var(--line);background:rgba(5,6,15,.6);backdrop-filter:blur(8px)}
.tape-in{display:inline-block;animation:mq 70s linear infinite}.tape:hover .tape-in{animation-play-state:paused}
@keyframes mq{to{transform:translateX(-50%)}}
.tk{display:inline-flex;gap:8px;align-items:center;margin:0 20px;padding:8px 0;font:500 .72rem 'JetBrains Mono',monospace;color:var(--mut)}
.tk b{color:#fff}.tk i{font-style:normal;display:inline-flex;gap:3px;align-items:center}.tk i.u{color:var(--up)}.tk i.d{color:var(--dn)}
.ar{width:.62em;height:.62em;display:inline-block}
/* dock nav */
[data-baseweb="tab-list"]{position:fixed;left:50%;transform:translateX(-50%);bottom:max(10px,env(safe-area-inset-bottom));width:min(94vw,720px);z-index:1000;gap:0!important;
 background:rgba(10,13,30,.72);backdrop-filter:blur(22px) saturate(1.6);border:1px solid var(--line);border-radius:24px;padding:4px;justify-content:space-around;overflow:hidden;
 box-shadow:0 18px 50px rgba(0,0,0,.6),0 0 0 1px rgba(0,240,255,.08),0 0 38px rgba(124,92,255,.25)}
[data-baseweb="tab-highlight"]{top:4px!important;bottom:4px!important;height:auto!important;border-radius:19px!important;
 background:linear-gradient(135deg,rgba(0,240,255,.22),rgba(255,43,214,.22))!important;box-shadow:inset 0 0 0 1px rgba(0,240,255,.45),0 0 22px rgba(0,240,255,.35)}
[data-baseweb="tab-border"]{display:none!important}
button[role="tab"]{flex:1 1 0;min-width:0;height:auto;padding:9px 0 8px;color:var(--mut);background:none;transition:.25s}
button[role="tab"][aria-selected="true"]{color:#fff;text-shadow:0 0 12px var(--cy)}
button[role="tab"] p{display:flex;flex-direction:column;align-items:center;gap:1px;margin:0;font-size:.56rem;font-weight:600;letter-spacing:.4px}
button[role="tab"] [data-testid="stIconMaterial"]{font-size:1.35rem!important}
/* appbar + hero */
.appbar{display:flex;align-items:center;gap:12px;margin:2px 0 12px}
.app-title{font:800 1.25rem Sora;letter-spacing:.5px;background:linear-gradient(90deg,#fff,var(--cy) 60%,var(--mg));-webkit-background-clip:text;background-clip:text;color:transparent}
.app-sub{font:500 .62rem 'JetBrains Mono';color:var(--mut);letter-spacing:1.4px;text-transform:uppercase}
.live{margin-left:auto;display:flex;align-items:center;gap:7px;font:700 .6rem 'JetBrains Mono';letter-spacing:1.5px;color:var(--up)}
.live::before{content:"";width:8px;height:8px;border-radius:50%;background:var(--up);box-shadow:0 0 0 0 var(--up);animation:pu 1.8s infinite}
@keyframes pu{70%{box-shadow:0 0 0 10px transparent}100%{box-shadow:0 0 0 0 transparent}}
.hero{position:relative;overflow:hidden;border-radius:26px;padding:20px 18px 16px;margin:10px 0 16px;border:1px solid transparent;
 background:linear-gradient(160deg,rgba(18,24,56,.9),rgba(8,10,26,.85)) padding-box,conic-gradient(from var(--ang,0deg),var(--cy),var(--vi),var(--mg),var(--cy)) border-box;
 animation:spin 7s linear infinite;box-shadow:0 0 60px rgba(124,92,255,.22),inset 0 0 60px rgba(0,240,255,.05)}
@property --ang{syntax:'<angle>';initial-value:0deg;inherits:false}
@keyframes spin{to{--ang:360deg}}
.hero::after{content:"";position:absolute;inset:0;background:linear-gradient(105deg,transparent 40%,rgba(255,255,255,.07) 50%,transparent 60%);transform:translateX(-100%);animation:sh 5s ease-in-out infinite}
@keyframes sh{60%,100%{transform:translateX(100%)}}
.hero-top{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
.hero-name{font:800 1.15rem Sora}.hero-sub{font:500 .66rem 'JetBrains Mono';color:var(--mut);margin-top:3px;letter-spacing:.8px}
.hero-price{font:800 clamp(2.4rem,11vw,3.6rem) 'JetBrains Mono';letter-spacing:-2px;margin-top:10px;line-height:1;
 background:linear-gradient(90deg,#fff,#b9c6ff 55%,var(--cy));-webkit-background-clip:text;background-clip:text;color:transparent;animation:rise .9s cubic-bezier(.2,.9,.3,1) both}
@keyframes rise{from{opacity:0;transform:translateY(16px) scale(.97);filter:blur(6px)}}
.pill{display:inline-flex;gap:6px;align-items:center;font:700 .76rem 'JetBrains Mono';padding:6px 12px;border-radius:999px;white-space:nowrap;backdrop-filter:blur(6px);border:1px solid currentColor}
.gwrap{text-align:center;margin-top:6px}.gauge{width:100%;max-width:250px}
.needle{transform-origin:100px 102px;transform:rotate(-90deg);animation:nd 1.6s cubic-bezier(.2,1.4,.3,1) .2s forwards}
@keyframes nd{to{transform:rotate(var(--a))}}
/* cards */
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.grid.one{grid-template-columns:1fr}
.grid .metric-card:last-child:nth-child(odd){grid-column:1/-1}
.metric-card,.sigcard,.news-card,.forecast-row,.srow{position:relative;background:var(--card);border:1px solid var(--line);border-radius:18px;backdrop-filter:blur(14px);
 transition:transform .3s,border-color .3s,box-shadow .3s;animation:fu .6s cubic-bezier(.2,.9,.3,1) both}
.metric-card:hover,.srow:hover,.news-card:hover{transform:translateY(-4px);border-color:rgba(0,240,255,.55);box-shadow:0 12px 34px rgba(0,240,255,.16),0 0 0 1px rgba(255,43,214,.18)}
.metric-card:nth-child(2){animation-delay:.07s}.metric-card:nth-child(3){animation-delay:.14s}.metric-card:nth-child(4){animation-delay:.21s}.metric-card:nth-child(n+5){animation-delay:.28s}
@keyframes fu{from{opacity:0;transform:translateY(18px)}}
.metric-card{padding:13px 14px;min-height:82px;height:100%;text-align:left;overflow:hidden}
.metric-card::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(var(--cy),var(--mg));opacity:.7}
.metric-label{font:600 .58rem 'JetBrains Mono';color:var(--mut);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.metric-value{font:800 1.28rem 'JetBrains Mono';line-height:1.15}
.metric-sub{font-size:.66rem;color:var(--mut);margin-top:4px;line-height:1.35}
.sigcard{padding:16px;margin-bottom:12px}.sigcard .badge{margin:6px 0 8px}
.bar{height:8px;background:rgba(255,255,255,.07);border-radius:8px;overflow:hidden;margin-top:10px}
.bar i{display:block;height:100%;border-radius:8px;box-shadow:0 0 14px currentColor;animation:gr 1.4s cubic-bezier(.2,.9,.3,1) both}
@keyframes gr{from{width:0!important}}
.badge{position:relative;overflow:hidden;display:inline-block;padding:8px 20px;border-radius:999px;font:800 .95rem Sora;letter-spacing:1px}
.badge::after{content:"";position:absolute;inset:0;background:linear-gradient(105deg,transparent 35%,rgba(255,255,255,.55) 50%,transparent 65%);transform:translateX(-120%);animation:sh 3.2s infinite}
.badge-strong-buy{background:linear-gradient(135deg,#19ffa3,#00f0ff);color:#021014;box-shadow:0 0 30px rgba(25,255,163,.55)}
.badge-buy{background:linear-gradient(135deg,#5dffc0,#19ffa3);color:#021014;box-shadow:0 0 22px rgba(25,255,163,.4)}
.badge-hold{background:linear-gradient(135deg,#ffc645,#ffe08a);color:#1a1200;box-shadow:0 0 22px rgba(255,198,69,.4)}
.badge-sell{background:linear-gradient(135deg,#ff8a50,#ff3d71);color:#fff;box-shadow:0 0 22px rgba(255,61,113,.45)}
.badge-strong-sell{background:linear-gradient(135deg,#ff3d71,#ff2bd6);color:#fff;box-shadow:0 0 30px rgba(255,43,214,.55)}
.section-header{display:flex;align-items:center;gap:10px;font:700 .72rem 'JetBrains Mono';color:#fff;text-transform:uppercase;letter-spacing:2px;margin:26px 0 12px}
.section-header::before{content:"";width:10px;height:10px;transform:rotate(45deg);background:linear-gradient(135deg,var(--cy),var(--mg));box-shadow:0 0 14px var(--cy);flex:none}
.section-header::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--cy),transparent)}
.chart-label{font:600 .64rem 'JetBrains Mono';color:var(--mut);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:4px}
.news-card{padding:13px 15px;margin-bottom:9px}.news-title{font-size:.84rem;font-weight:600;line-height:1.45}
.news-pos,.news-neg,.news-neu{font:700 .64rem 'JetBrains Mono';margin-top:6px;letter-spacing:1px;text-transform:uppercase}
.news-pos{color:var(--up)}.news-neg{color:var(--dn)}.news-neu{color:var(--mut)}
.forecast-row{padding:13px 15px;margin-bottom:9px;display:flex;justify-content:space-between;align-items:center}
.forecast-label{font-size:.74rem;color:var(--mut)}.forecast-val{font:800 1.05rem 'JetBrains Mono'}.forecast-pnl{font:700 .82rem 'JetBrains Mono'}
.status-ok,.status-warn{display:flex;gap:10px;align-items:center;border-radius:14px;padding:11px 13px;font-size:.8rem;margin-top:8px}
.status-ok{background:rgba(25,255,163,.08);border:1px solid rgba(25,255,163,.5);color:var(--up)}
.status-warn{background:rgba(255,198,69,.08);border:1px solid rgba(255,198,69,.5);color:var(--am)}
.dot{width:9px;height:9px;border-radius:50%;background:currentColor;box-shadow:0 0 12px currentColor;flex:none}
.srow{padding:12px 14px;margin-bottom:9px}.sr-top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.sr-name{color:var(--mut);font-size:.7rem;margin-left:8px}.sr-sig{font:800 .62rem 'JetBrains Mono';padding:5px 10px;border-radius:999px;white-space:nowrap}
.sr-stats{display:flex;flex-wrap:wrap;gap:4px 14px;margin-top:8px;font:500 .68rem 'JetBrains Mono';color:var(--mut)}.sr-stats b{color:#fff}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 8px}.chip{font:500 .64rem 'JetBrains Mono';color:var(--mut);background:var(--card);border:1px solid var(--line);border-radius:999px;padding:5px 11px}
[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;gap:.6rem!important}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"],[data-testid="stHorizontalBlock"]>[data-testid="column"]{flex:1 1 calc(50% - .6rem)!important;min-width:calc(50% - .6rem)!important;width:auto!important}
[data-testid="stHorizontalBlock"]:has(>:nth-child(3):last-child)>*{flex-basis:calc(33% - .6rem)!important;min-width:calc(33% - .6rem)!important}
input,textarea,[data-baseweb="select"] *{font-size:16px!important}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"]{background:rgba(16,20,42,.7)!important;border-color:var(--line)!important;border-radius:14px!important;min-height:48px;backdrop-filter:blur(10px)}
[data-baseweb="select"]>div:focus-within,[data-baseweb="input"]:focus-within{border-color:var(--cy)!important;box-shadow:0 0 0 3px rgba(0,240,255,.18),0 0 24px rgba(0,240,255,.2)!important}
[data-testid="stSelectbox"] label,[data-testid="stNumberInput"] label,[data-testid="stSlider"] label{color:#a9b3d8!important;font:600 .72rem 'JetBrains Mono'!important;letter-spacing:.8px}
[data-testid="stSlider"] [role="slider"]{background:var(--cy)!important;box-shadow:0 0 16px var(--cy)}
.stButton>button{position:relative;overflow:hidden;width:100%;min-height:50px;border:0;border-radius:16px;font:800 .85rem Sora;letter-spacing:1px;color:#04121a;
 background:linear-gradient(135deg,var(--cy),var(--vi) 55%,var(--mg));background-size:200% 200%;animation:bg 6s ease infinite;box-shadow:0 10px 34px rgba(124,92,255,.45)}
@keyframes bg{50%{background-position:100% 50%}}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 14px 44px rgba(0,240,255,.5);color:#04121a}.stButton>button:active{transform:scale(.97)}
[data-testid="stExpander"]{background:var(--card);border:1px solid var(--line)!important;border-radius:16px;backdrop-filter:blur(12px)}
[data-testid="stPlotlyChart"]{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:6px;overflow:hidden;backdrop-filter:blur(14px);box-shadow:0 10px 40px rgba(0,0,0,.35)}
[data-testid="stSpinner"] i,[data-testid="stSpinner"] svg{border-top-color:var(--cy)!important}
.foot{text-align:center;margin-top:30px;font:500 .62rem 'JetBrains Mono';letter-spacing:1.2px;color:var(--mut)}
::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-thumb{background:linear-gradient(var(--cy),var(--mg));border-radius:6px}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}.needle{transform:rotate(var(--a))}}
</style>
"""
