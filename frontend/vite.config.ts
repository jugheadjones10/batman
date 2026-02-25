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
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})

