import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: true,   // Listen on 0.0.0.0 so you can access via Tailscale/LAN IP (e.g. http://<tailscale-ip>:5180)
    port: 5180,  // Use 5180 to avoid conflicts with other Vite apps
    strictPort: false,  // Use next available port if occupied
    proxy: {
      '/api': {
        // In WSL2 + Windows: if frontend runs in Windows and backend in WSL, use localhost or set VITE_API_PROXY_TARGET
        target: process.env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
        timeout: 120000, // 2 min so long-running / auto-label etc. don't socket hang up
        // Forward Range header for video streaming (browsers use it for <video>)
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq, req) => {
            const range = req.headers.range
            if (range) proxyReq.setHeader('Range', range)
          })
        },
      },
    },
  },
})

