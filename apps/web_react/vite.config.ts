import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev: `npm run dev` on 127.0.0.1:5173, API proxied to the FastAPI server on 127.0.0.1:8000.
// Prod: `npm run build` -> dist/, served by FastAPI at "/".
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { host: "127.0.0.1", port: 5173, proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { outDir: "dist", chunkSizeWarningLimit: 1500 },
});
