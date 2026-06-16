import { Alert, Box, Paper, Typography, Chip } from '@mui/material'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { Message } from '../stores/chat-store'
import { getAgentColor, getAgentIcon } from '../themes/uconn-theme'
import PhysicsGraphPanel from './PhysicsGraphPanel'

interface ChatMessageProps {
  message: Message
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user'
  const agentColor = getAgentColor(message.agentId)
  const agentIcon = getAgentIcon(message.agentId)

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 2,
      }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: isUser ? 'row-reverse' : 'row',
          alignItems: 'flex-start',
          maxWidth: '80%',
        }}
      >
        {/* Avatar */}
        <Box
          sx={{
            width: 40,
            height: 40,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: isUser ? 'primary.main' : agentColor,
            color: 'white',
            fontSize: '1.2rem',
            flexShrink: 0,
            mx: 1,
          }}
        >
          {isUser ? '👨‍🎓' : agentIcon}
        </Box>

        {/* Message Content */}
        <Paper
          elevation={1}
          sx={{
            p: 2,
            borderRadius: 2,
            bgcolor: isUser ? 'primary.main' : 'background.paper',
            color: isUser ? 'primary.contrastText' : 'text.primary',
            borderTopRightRadius: isUser ? 0 : 16,
            borderTopLeftRadius: isUser ? 16 : 0,
          }}
        >
          {/* Markdown content with LaTeX math rendering */}
          <Box
            sx={{
              '& p': { m: 0, mb: 1 },
              '& p:last-child': { mb: 0 },
              '& code': {
                bgcolor: isUser ? 'rgba(255,255,255,0.2)' : 'grey.100',
                px: 0.5,
                borderRadius: 0.5,
                fontFamily: 'monospace',
              },
              '& pre': {
                bgcolor: isUser ? 'rgba(255,255,255,0.1)' : 'grey.100',
                p: 1,
                borderRadius: 1,
                overflow: 'auto',
              },
              '& ul, & ol': { pl: 2, mb: 1 },
              '& li': { mb: 0.5 },
              // KaTeX math styling
              '& .katex': {
                fontSize: '1.1em',
              },
              '& .katex-display': {
                margin: '0.5em 0',
                overflow: 'auto',
              },
            }}
          >
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
            >
              {message.content}
            </ReactMarkdown>
          </Box>

          {!isUser && message.graphs && message.graphs.length > 0 && (
            <PhysicsGraphPanel graphs={message.graphs} />
          )}

          {!isUser && (!message.graphs || message.graphs.length === 0) && message.graphWarnings && message.graphWarnings.length > 0 && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              {message.graphWarnings.join(' ')}
            </Alert>
          )}

          {!isUser && message.graphErrors && message.graphErrors.length > 0 && (
            <Alert severity="info" sx={{ mt: 2 }}>
              {message.graphErrors.join(' ')}
            </Alert>
          )}

          {/* Tools used indicator */}
          {message.toolsUsed && message.toolsUsed.length > 0 && (
            <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
              {message.toolsUsed.map((tool, i) => (
                <Chip
                  key={i}
                  label={tool}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: '0.7rem' }}
                />
              ))}
            </Box>
          )}

          {/* Timestamp */}
          <Typography
            variant="caption"
            sx={{
              display: 'block',
              mt: 1,
              opacity: 0.7,
              textAlign: isUser ? 'right' : 'left',
            }}
          >
            {new Date(message.timestamp).toLocaleTimeString()}
          </Typography>
        </Paper>
      </Box>
    </Box>
  )
}
