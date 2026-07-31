import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '')
  const apiTarget = env.VITE_API_TARGET || 'http://127.0.0.1:8000'
  const databaseApiTarget = env.VITE_DATABASE_API_TARGET || 'http://127.0.0.1:8001'

  return {
    plugins: [react()],
    server: {
      port: 3001,
      proxy: {
        '/api/database': {
          target: databaseApiTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/database/, '')
        },
        '/api': {
          target: apiTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, '')
        }
      }
    }
  }
})
