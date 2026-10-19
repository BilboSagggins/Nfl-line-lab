// NFL Line Lab Discord bot — Cloudflare Worker.
// - Slash commands: /games, /score, /lines, /setup  (HTTP interactions, no always-on server)
// - Every minute (cron): posts this week's games to the chosen channel as silent messages,
//   then edits them with live scores. Edits never notify anyone.

const ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard";
const API = "https://discord.com/api/v10";
const SILENT = 4096;      // SUPPRESS_NOTIFICATIONS
const EPHEMERAL = 64;
const ESPN_FIX = { LAR: "LA", WSH: "WAS" };
const fixAb = (a) => ESPN_FIX[a] || a;
const COLORS = { live: 0x2c8f50, half: 0x2c8f50, delay: 0xe3aa4c, final: 0x5b6270, pre: 0x3d84d6, off: 0xb3261e };

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (request.method === "GET") return new Response("NFL Line Lab bot is running.");
    if (request.method !== "POST" || url.pathname !== "/") return new Response("Not found", { status: 404 });
    const body = await request.text();
    const ok = await verifySignature(request, body, env.DISCORD_PUBLIC_KEY);
    if (!ok) return new Response("Bad signature", { status: 401 });
    const i = JSON.parse(body);
    if (i.type === 1) return json({ type: 1 }); // PING
    if (i.type === 2) return handleCommand(i, env, ctx);
    return json({ type: 4, data: { content: "Unsupported interaction.", flags: EPHEMERAL } });
  },
  async scheduled(event, env, ctx) {
    ctx.waitUntil(updateChannel(env, new Date(event.scheduledTime || Date.now())));
  },
};

// ---------------------------------------------------------------- helpers

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json" } });
}
const hex = (s) => new Uint8Array(s.match(/.{1,2}/g).map((b) => parseInt(b, 16)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function verifySignature(request, body, publicKey) {
  const sig = request.headers.get("x-signature-ed25519");
  const ts = request.headers.get("x-signature-timestamp");
  if (!sig || !ts || !publicKey) return false;
  try {
    const key = await crypto.subtle.importKey("raw", hex(publicKey), { name: "Ed25519" }, false, ["verify"]);
    return await crypto.subtle.verify("Ed25519", key, hex(sig), new TextEncoder().encode(ts + body));
  } catch (e) {
    return false;
  }
}

async function discord(env, method, path, payload) {
  for (let attempt = 0; attempt < 4; attempt++) {
    const r = await fetch(API + path, {
      method,
      headers: { Authorization: `Bot ${env.DISCORD_TOKEN}`, "content-type": "application/json" },
      body: payload ? JSON.stringify(payload) : undefined,
    });
    if (r.status === 429) {
      const j = await r.json().catch(() => ({}));
      await sleep(Math.ceil((j.retry_after || 1) * 1000) + 100);
      continue;
    }
    if (!r.ok) throw new Error(`Discord ${method} ${path} -> ${r.status} ${await r.text()}`);
    return r.status === 204 ? null : r.json();
  }
  throw new Error("Discord rate limit kept failing");
}

// ---------------------------------------------------------------- data

async function scoreboard(week) {
  const r = await fetch(week ? `${ESPN}?seasontype=${week.type}&week=${week.week}&dates=${week.season}` : ESPN,
    { cf: { cacheTtl: 20 } });
  if (!r.ok) throw new Error("ESPN " + r.status);
  const d = await r.json();
  return {
    season: d.season && d.season.year, type: d.season && d.season.type, week: d.week && d.week.number,
    games: (d.events || []).map(parseEvent).filter(Boolean).sort((a, b) => new Date(a.date) - new Date(b.date)),
  };
}

export function statusOf(c) {
  const st = c.status || {}, t = st.type || {}, n = t.name || "", d = t.shortDetail || t.detail || "";
  const q = st.period > 4 ? "OT" : st.period ? "Q" + st.period : "";
  if (/POSTPONED/.test(n)) return { key: "off", label: "Postponed", detail: "" };
  if (/CANCEL/.test(n)) return { key: "off", label: "Canceled", detail: "" };
  if (/DELAY|SUSPENDED/.test(n)) return { key: "delay", label: /SUSPENDED/.test(n) ? "Suspended" : "Delayed", detail: q ? `${q} ${st.displayClock || ""}`.trim() : "" };
  if (t.state === "post" || t.completed) return { key: "final", label: /OT/i.test(d) ? "Final / OT" : "Final", detail: "" };
  if (/HALFTIME/.test(n)) return { key: "half", label: "Halftime", detail: "" };
  if (t.state === "in") return { key: "live", label: "Live", detail: /END_PERIOD/.test(n) ? `End of ${q}` : `${q} ${st.displayClock || ""}`.trim() };
  return { key: "pre", label: "Upcoming", detail: "" };
}

export function parseEvent(ev) {
  const c = ev.competitions && ev.competitions[0];
  if (!c) return null;
  const side = (ha) => {
    const x = (c.competitors || []).find((y) => y.homeAway === ha) || {}, t = x.team || {};
    const rec = (x.records || []).find((r) => r.type === "total") || (x.records || [])[0];
    return { id: String(t.id || ""), ab: fixAb(t.abbreviation || ""), name: t.shortDisplayName || t.name || t.abbreviation || "",
      full: t.displayName || t.shortDisplayName || "", score: x.score != null && x.score !== "" ? Number(x.score) : null, rec: rec ? rec.summary : "" };
  };
  const v = c.venue || {}, a = v.address || {}, sit = c.situation || {};
  return {
    id: String(ev.id), date: ev.date, home: side("home"), away: side("away"), status: statusOf(c),
    venue: v.fullName || "", city: [a.city, a.state || a.country].filter(Boolean).join(", "),
    tv: (c.broadcasts || []).flatMap((b) => b.names || []).join(", "),
    poss: sit.possession ? String(sit.possession) : null, down: sit.downDistanceText || "", red: !!sit.isRedZone,
  };
}

async function siteData(env) {
  if (!env.DATA_URL) return null;
  try {
    const r = await fetch(env.DATA_URL, { cf: { cacheTtl: 600 } });
    return r.ok ? r.json() : null;
  } catch (e) {
    return null;
  }
}

function lineFor(data, g) {
  if (!data || !data.games) return null;
  return data.games.find((x) => x.away === g.away.ab && x.home === g.home.ab) || null;
}

function lineText(b, g) {
  if (!b || !b.market) return "";
  const m = b.market;
  let fav = "";
  if (m.home_line != null) fav = m.home_line === 0 ? "Pick'em" : m.home_line < 0 ? `${g.home.name} ${m.home_line}` : `${g.away.name} ${-m.home_line}`;
  return [fav && `Spread: ${fav}`, m.total != null && `Total: ${m.total}`].filter(Boolean).join(" · ");
}

// ---------------------------------------------------------------- rendering

export function gameEmbed(g, data, now = new Date()) {
  const s = g.status, live = ["live", "half", "delay"].includes(s.key), showScore = live || s.key === "final";
  const lead = (t, o) => showScore && t.score != null && o.score != null && t.score > o.score;
  const teamLine = (t, o) => {
    const name = lead(t, o) ? `**${t.name}**` : t.name;
    const ball = live && g.poss && g.poss === t.id ? "  (ball)" : "";
    return showScore ? `${name}  ${lead(t, o) ? "**" + t.score + "**" : t.score}${ball}` : `${name}${t.rec ? "  (" + t.rec + ")" : ""}`;
  };
  const head = s.key === "pre" ? "Upcoming" : `${s.label}${s.detail ? " · " + s.detail : ""}`;
  const unix = Math.floor(new Date(g.date).getTime() / 1000);
  const lines = [
    `**${head.toUpperCase()}**`,
    teamLine(g.away, g.home),
    teamLine(g.home, g.away),
  ];
  if (live && g.down) lines.push(`${g.down}${g.red ? " · red zone" : ""}`);
  lines.push("");
  lines.push(`Kickoff: <t:${unix}:F>${g.tv ? " · " + g.tv : ""}`);
  if (g.venue) lines.push(`${g.venue}${g.city ? ", " + g.city : ""}`);
  const lt = lineText(lineFor(data, g), g);
  if (lt) lines.push(lt);
  return {
    title: `${g.away.name} at ${g.home.name}`,
    description: lines.join("\n"),
    color: COLORS[s.key] || COLORS.pre,
    footer: { text: live ? "Live · updates every minute" : s.key === "final" ? "Final" : "NFL Line Lab" },
    timestamp: now.toISOString(),
  };
}

function weekLine(g) {
  const s = g.status, unix = Math.floor(new Date(g.date).getTime() / 1000);
  if (s.key === "pre") return `${g.away.name} at ${g.home.name} · <t:${unix}:f>`;
  const sc = `${g.away.name} ${g.away.score ?? ""} – ${g.home.name} ${g.home.score ?? ""}`;
  return `${sc} · ${s.label}${s.detail ? " " + s.detail : ""}`;
}

// ---------------------------------------------------------------- channel updates

export async function updateChannel(env, now = new Date(), force = false) {
  const channel = await env.STATE.get("channel");
  if (!channel) return { skipped: "no channel" };
  const sb = await scoreboard();
  if (!sb.games.length) return { skipped: "no games" };
  const key = `week:${sb.season}-${sb.type}-${sb.week}`;
  const raw = await env.STATE.get(key);
  let rec = raw ? JSON.parse(raw) : null;
  if (rec && rec.channel !== channel) rec = null; // channel changed: start fresh there
  const data = await siteData(env);
  let dirty = false, posted = 0, edited = 0;
  if (!rec) {
    const label = sb.type === 3 ? "Playoffs" : `Week ${sb.week}`;
    const header = await discord(env, "POST", `/channels/${channel}/messages`, {
      content: `**${label} games** — scores update live in the messages below.`, flags: SILENT,
    });
    rec = { channel, header: header.id, games: {} };
    dirty = true;
  }
  const minute = now.getUTCMinutes();
  for (const g of sb.games) {
    const st = rec.games[g.id];
    const live = ["live", "half", "delay"].includes(g.status.key);
    if (!st) {
      const m = await discord(env, "POST", `/channels/${channel}/messages`, { embeds: [gameEmbed(g, data, now)], flags: SILENT });
      rec.games[g.id] = { msg: m.id, state: g.status.key };
      dirty = true; posted++;
      continue;
    }
    const changed = st.state !== g.status.key;
    if (live || changed || force || minute % 30 === 0) {
      if (st.state === "final" && !changed && !force) continue; // finished games stay put
      await discord(env, "PATCH", `/channels/${channel}/messages/${st.msg}`, { embeds: [gameEmbed(g, data, now)] });
      edited++;
      if (changed) { st.state = g.status.key; dirty = true; }
    }
  }
  if (dirty) await env.STATE.put(key, JSON.stringify(rec));
  return { posted, edited, week: key };
}

// ---------------------------------------------------------------- commands

const ALIASES = {
  ARI: ["cardinals", "arizona", "cards"], ATL: ["falcons", "atlanta"], BAL: ["ravens", "baltimore"],
  BUF: ["bills", "buffalo"], CAR: ["panthers", "carolina"], CHI: ["bears", "chicago"], CIN: ["bengals", "cincinnati"],
  CLE: ["browns", "cleveland"], DAL: ["cowboys", "dallas"], DEN: ["broncos", "denver"], DET: ["lions", "detroit"],
  GB: ["packers", "green bay"], HOU: ["texans", "houston"], IND: ["colts", "indianapolis", "indy"],
  JAX: ["jaguars", "jacksonville", "jags"], KC: ["chiefs", "kansas city"], LV: ["raiders", "las vegas", "vegas"],
  LAC: ["chargers", "bolts"], LA: ["rams", "lar"], MIA: ["dolphins", "miami", "fins"], MIN: ["vikings", "minnesota"],
  NE: ["patriots", "new england", "pats"], NO: ["saints", "new orleans"], NYG: ["giants"], NYJ: ["jets"],
  PHI: ["eagles", "philadelphia", "philly"], PIT: ["steelers", "pittsburgh"], SF: ["49ers", "niners", "san francisco"],
  SEA: ["seahawks", "seattle"], TB: ["buccaneers", "bucs", "tampa bay", "tampa"], TEN: ["titans", "tennessee"],
  WAS: ["commanders", "washington", "wsh"],
};
export function teamFromText(t) {
  const q = String(t || "").trim().toLowerCase();
  if (!q) return null;
  for (const [ab, names] of Object.entries(ALIASES)) {
    if (q === ab.toLowerCase() || names.some((n) => q === n || q.includes(n))) return ab;
  }
  return null;
}
const opt = (i, name) => ((i.data.options || []).find((o) => o.name === name) || {}).value;
const reply = (data) => json({ type: 4, data });

async function handleCommand(i, env, ctx) {
  const name = i.data.name;
  try {
    if (name === "setup") {
      const channel = i.channel_id || (i.channel && i.channel.id);
      await env.STATE.put("channel", channel);
      ctx.waitUntil(updateChannel(env, new Date(), true).catch((e) => console.log("setup update failed", e.message)));
      return reply({ content: "Got it. This channel will get the live scoreboard. Posting this week's games now (silently, no pings).", flags: EPHEMERAL });
    }
    if (name === "stop") {
      await env.STATE.delete("channel");
      return reply({ content: "Live scoreboard turned off. Run /setup in a channel to turn it back on.", flags: EPHEMERAL });
    }
    const sb = await scoreboard();
    if (name === "games") {
      const label = sb.type === 3 ? "Playoff games" : `Week ${sb.week} games`;
      const order = { live: 0, half: 0, delay: 0, pre: 1, final: 2, off: 3 };
      const list = [...sb.games].sort((a, b) => order[a.status.key] - order[b.status.key] || new Date(a.date) - new Date(b.date));
      return reply({ embeds: [{ title: label, description: list.map(weekLine).join("\n") || "No games found.", color: COLORS.pre }] });
    }
    const team = teamFromText(opt(i, "team"));
    if (!team) return reply({ content: "I couldn't tell which team you meant. Try a name like `bears` or `KC`.", flags: EPHEMERAL });
    const g = sb.games.find((x) => x.home.ab === team || x.away.ab === team);
    if (!g) return reply({ content: `${team} doesn't play this week.`, flags: EPHEMERAL });
    const data = await siteData(env);
    if (name === "score") return reply({ embeds: [gameEmbed(g, data)] });
    if (name === "lines") {
      const b = lineFor(data, g);
      if (!b) return reply({ content: "No betting lines for that game yet. They're pulled Thursday morning.", flags: EPHEMERAL });
      return reply({ embeds: [linesEmbed(g, b)] });
    }
    return reply({ content: "Unknown command.", flags: EPHEMERAL });
  } catch (e) {
    return reply({ content: "Something went wrong getting that. Try again in a minute.", flags: EPHEMERAL });
  }
}

const odds = (o) => (o == null ? "–" : (o > 0 ? "+" : "") + Math.round(o));
const pct = (p) => (p == null ? "–" : Math.round(p * 100) + "%");
export function linesEmbed(g, b) {
  const m = b.market || {}, bk = b.books || {}, mo = b.model || {};
  const H = g.home.name, A = g.away.name, out = [];
  if (bk.ml && bk.ml.fair_home != null) {
    out.push(`**Who wins**`);
    out.push(`${A} ${odds(bk.ml.best_away && bk.ml.best_away.price)} (${pct(1 - bk.ml.fair_home)})`);
    out.push(`${H} ${odds(bk.ml.best_home && bk.ml.best_home.price)} (${pct(bk.ml.fair_home)})`);
  } else if (m.ml_home != null) {
    out.push(`**Who wins**`, `${A} ${odds(m.ml_away)} · ${H} ${odds(m.ml_home)}`);
  }
  if (m.home_line != null) {
    const s = bk.spread || {};
    out.push("", `**Spread**`);
    out.push(`${A} ${m.home_line > 0 ? "" : "+"}${-m.home_line}${s.best_away ? " " + odds(s.best_away.price) + " at " + s.best_away.book : ""}`);
    out.push(`${H} ${m.home_line > 0 ? "+" : ""}${m.home_line}${s.best_home ? " " + odds(s.best_home.price) + " at " + s.best_home.book : ""}`);
  }
  if (m.total != null) {
    const t = bk.total || {};
    out.push("", `**Total points: ${m.total}**`);
    if (t.best_over) out.push(`Over ${t.best_over.line} ${odds(t.best_over.price)} at ${t.best_over.book}`);
    if (t.best_under) out.push(`Under ${t.best_under.line} ${odds(t.best_under.price)} at ${t.best_under.book}`);
  }
  if (mo.home_pts != null) out.push("", `Our model: ${A} ${Math.round(mo.away_pts)}, ${H} ${Math.round(mo.home_pts)}`);
  return { title: `${A} at ${H} · betting lines`, description: out.join("\n"), color: COLORS.pre,
    footer: { text: "Percent = market chance with the sportsbooks' cut removed" } };
}
