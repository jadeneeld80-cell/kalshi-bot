
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],

  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  },

  preview: {
    host: '0.0.0.0',
    allowedHosts: [
      'web-production-386f6.up.railway.app'
    ]
  }
})

