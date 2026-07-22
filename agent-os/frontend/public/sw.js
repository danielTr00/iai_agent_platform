// Minimaler Service Worker — macht die App installierbar (PWA).
// Bewusst KEIN Caching der API/WS; nur App-Shell fuer den Prototyp.
const SHELL = ["/", "/app.js", "/manifest.webmanifest"];
self.addEventListener("install", (e) => {
  e.waitUntil(caches.open("agent-os-v1").then((c) => c.addAll(SHELL)));
  self.skipWaiting();
});
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.pathname.startsWith("/ws") || url.pathname.startsWith("/health")) return;
  e.respondWith(caches.match(e.request).then((r) => r || fetch(e.request)));
});
