// Agent OS — minimaler PWA-Client (Vanilla JS, Push-to-talk).
// v1: Push-to-talk + akkumulierte TTS-Wiedergabe. VAD + Streaming-Playback -> Phase 4.

const log = document.getElementById("log");
const textInput = document.getElementById("text");
const sendBtn = document.getElementById("send");
const talkBtn = document.getElementById("talk");
const voiceBtn = document.getElementById("voice");

let ws, voiceOut = true, agentEl = null;
let audioChunks = [];               // TTS-Frames des laufenden Turns
let mediaRecorder, recChunks = [];

function line(text, cls) {
  const el = document.createElement("div");
  el.className = "msg " + cls;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

// --- Token (Single-User) einmal abfragen und lokal merken ---
function getToken() {
  let t = localStorage.getItem("harness_token");
  if (!t) { t = prompt("Harness-Token:") || ""; localStorage.setItem("harness_token", t); }
  return t;
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => ws.send(JSON.stringify({ type: "auth", token: getToken() }));
  ws.onmessage = (ev) => handle(JSON.parse(ev.data));
  ws.onclose = () => line("Verbindung getrennt. Neu laden.", "sys");
}

function handle(msg) {
  switch (msg.type) {
    case "auth":
      if (!msg.ok) { localStorage.removeItem("harness_token"); line("Auth fehlgeschlagen.", "sys"); }
      else line("Verbunden.", "sys");
      break;
    case "transcript":
      line(msg.text, "user"); break;
    case "text":
      if (!agentEl) agentEl = line("", "agent");
      agentEl.textContent += msg.text;
      log.scrollTop = log.scrollHeight;
      break;
    case "audio":
      audioChunks.push(Uint8Array.from(atob(msg.data), c => c.charCodeAt(0)));
      break;
    case "done":
      agentEl = null;
      playAudio();
      break;
    case "error":
      line("Fehler: " + msg.message, "sys"); agentEl = null; break;
  }
}

function playAudio() {
  if (!voiceOut || audioChunks.length === 0) { audioChunks = []; return; }
  const blob = new Blob(audioChunks, { type: "audio/mpeg" });
  audioChunks = [];
  new Audio(URL.createObjectURL(blob)).play().catch(() => {});
}

function sendText(t) {
  if (!t.trim() || ws.readyState !== 1) return;
  line(t, "user");
  ws.send(JSON.stringify({ type: "text", text: t }));
}

sendBtn.onclick = () => { sendText(textInput.value); textInput.value = ""; };
textInput.onkeydown = (e) => { if (e.key === "Enter") sendBtn.onclick(); };

voiceBtn.onclick = () => {
  voiceOut = !voiceOut;
  voiceBtn.classList.toggle("on", voiceOut);
  ws.send(JSON.stringify({ type: voiceOut ? "voice_on" : "voice_off" }));
};

// --- Push-to-talk ---
async function startRec() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
  recChunks = [];
  mediaRecorder.ondataavailable = (e) => e.data.size && recChunks.push(e.data);
  mediaRecorder.onstop = async () => {
    stream.getTracks().forEach(t => t.stop());
    const blob = new Blob(recChunks, { type: "audio/webm" });
    const buf = new Uint8Array(await blob.arrayBuffer());
    let bin = ""; buf.forEach(b => bin += String.fromCharCode(b));
    ws.send(JSON.stringify({ type: "audio", data: btoa(bin), mime: "audio/webm" }));
  };
  mediaRecorder.start();
  talkBtn.classList.add("rec");
}
function stopRec() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  talkBtn.classList.remove("rec");
}
talkBtn.onpointerdown = (e) => { e.preventDefault(); startRec(); };
talkBtn.onpointerup = (e) => { e.preventDefault(); stopRec(); };
talkBtn.onpointercancel = stopRec;

if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
connect();
