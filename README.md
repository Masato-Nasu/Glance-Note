# Glance Note

Minimal mobile PWA instrument.

## Controls
- Move your face gently left/right for pitch.
- Move your face up/down for volume and brightness.
- Double blink to change voice.
- Tap **Record** once. Recording starts after 0.8s.
- Tap again to stop and download the take.

## Notes
- Use HTTPS so the front camera can start.
- Best on a phone in portrait orientation.
- The app uses MediaPipe Face Landmarker from a CDN, so the first load needs internet.
- The service worker caches the app shell for later launches.

## Deploy
Upload the whole folder to GitHub Pages, Cloudflare Pages, or any static hosting.
