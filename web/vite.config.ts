import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return
          }

          if (id.includes('recharts')) {
            return 'charts'
          }

          if (id.includes('@supabase')) {
            return 'supabase'
          }

          if (id.includes('@tanstack')) {
            return 'query'
          }

          if (id.includes('@radix-ui')) {
            return 'radix'
          }

          if (id.includes('react-router-dom')) {
            return 'router'
          }

          if (id.includes('lucide-react')) {
            return 'icons'
          }
        },
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
