import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Alert,
  Container,
  Divider,
  Chip,
} from '@mui/material'
import { Science as ScienceIcon } from '@mui/icons-material'
import { useStore } from '../stores/chat-store'

// Demo users (same as Streamlit)
const DEMO_USERS: Record<string, { password: string; name: string; role: string }> = {
  Lastname_Firstname: {
    password: 'password123',
    name: 'Physics Student 1',
    role: 'student',
  },
  demo_instructor: {
    password: 'instructor123',
    name: 'Demo Instructor',
    role: 'instructor',
  },
}

export default function LoginPage() {
  const navigate = useNavigate()
  const login = useStore((state) => state.login)

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setLoading(true)

    // Simulate auth delay
    await new Promise((resolve) => setTimeout(resolve, 300))

    const user = DEMO_USERS[username]
    if (user && user.password === password) {
      login({
        username,
        name: user.name,
        role: user.role,
      })
      navigate('/')
    } else {
      setError('Invalid username or password')
    }

    setLoading(false)
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #000E2F 0%, #001a4d 50%, #0076CE 100%)',
      }}
    >
      <Container maxWidth="sm">
        <Card sx={{ p: 2 }}>
          <CardContent>
            {/* Header */}
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <ScienceIcon sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
              <Typography variant="h4" component="h1" gutterBottom>
                Physics Assistant
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Interactive Physics Tutoring System
              </Typography>
            </Box>

            {/* Login Form */}
            <Typography variant="h6" gutterBottom>
              Student Login
            </Typography>

            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            <form onSubmit={handleSubmit}>
              <TextField
                fullWidth
                label="Username"
                placeholder="Enter your student ID"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                margin="normal"
                required
                autoFocus
              />
              <TextField
                fullWidth
                label="Password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                margin="normal"
                required
              />
              <Button
                type="submit"
                fullWidth
                variant="contained"
                size="large"
                disabled={loading}
                sx={{ mt: 3, mb: 2 }}
              >
                {loading ? 'Logging in...' : 'Login'}
              </Button>
            </form>

            <Divider sx={{ my: 3 }} />

            {/* Demo Credentials */}
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                <strong>Demo Credentials:</strong>
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center', flexWrap: 'wrap' }}>
                <Chip
                  label="Username: Lastname_Firstname"
                  size="small"
                  variant="outlined"
                  onClick={() => setUsername('Lastname_Firstname')}
                  sx={{ cursor: 'pointer' }}
                />
                <Chip
                  label="Password: password123"
                  size="small"
                  variant="outlined"
                  onClick={() => setPassword('password123')}
                  sx={{ cursor: 'pointer' }}
                />
              </Box>
            </Box>
          </CardContent>
        </Card>

        {/* Footer */}
        <Typography
          variant="body2"
          sx={{ textAlign: 'center', mt: 3, color: 'rgba(255,255,255,0.7)' }}
        >
          University of Connecticut &bull; Physics Department
        </Typography>
      </Container>
    </Box>
  )
}
