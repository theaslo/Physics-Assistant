import { useRef, useState, KeyboardEvent } from 'react'
import { Box, Button, TextField, IconButton, Paper, Typography } from '@mui/material'
import { Close as CloseIcon, Gesture as SketchIcon, Send as SendIcon } from '@mui/icons-material'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import * as katex from 'katex'
import 'katex/dist/katex.min.css'
import type { StudentDrawing } from '../stores/chat-store'
import { formatPhysicsMathMarkdown } from '../utils/physics-math-markdown'
import SketchPadDialog from './SketchPadDialog'

const LATEX_SNIPPETS = [
  {
    title: 'Variables',
    snippets: [
      { label: 'x_f', value: 'x_f ' },
      { label: 'x_0', value: 'x_0 ' },
      { label: 'v_f', value: 'v_f ' },
      { label: 'v_0', value: 'v_0 ' },
      { label: 'm', value: 'm ' },
      { label: 'a', value: 'a ' },
      { label: 't', value: 't ' },
    ],
  },
  {
    title: 'Physics',
    snippets: [
      { label: '\\Delta x', value: '\\Delta x ' },
      { label: '\\Delta t', value: '\\Delta t ' },
      { label: '\\vec{x}', value: '\\vec{x} ' },
      { label: '\\vec{v}', value: '\\vec{v} ' },
      { label: '\\theta', value: '\\theta ' },
      { label: '\\sum F', value: '\\sum F ' },
      { label: 'F_{net}', value: 'F_{net} ' },
      { label: 'f_k', value: 'f_k ' },
      { label: '\\mu_k', value: '\\mu_k ' },
    ],
  },
  {
    title: 'Operators',
    snippets: [
      { label: '=', value: ' = ' },
      { label: '+', value: ' + ' },
      { label: '-', value: ' - ' },
      { label: '*', value: ' * ' },
      { label: '/', value: ' / ' },
      { label: '()', value: '() ', cursorOffset: -2 },
      { label: 'y^x', value: '^{} ', cursorOffset: -2 },
    ],
  },
  {
    title: 'Templates',
    snippets: [
      { label: 't^2', value: 't^2 ' },
      { label: '\\frac{\\square}{\\square}', value: '\\frac{}{} ', cursorOffset: -4 },
      { label: '\\frac{1}{2}', value: '\\frac{1}{2} ' },
      { label: '\\sqrt{\\square}', value: '\\sqrt{} ', cursorOffset: -2 },
      { label: '\\sin', value: '\\sin() ', cursorOffset: -2 },
      { label: '\\cos', value: '\\cos() ', cursorOffset: -2 },
    ],
  },
]

interface ChatInputProps {
  onSend: (message: string, drawing?: StudentDrawing) => void
  disabled?: boolean
}

function RenderedMathSnippet({ expression }: { expression: string }) {
  const html = katex.renderToString(expression, {
    throwOnError: false,
    strict: 'ignore',
  })

  return (
    <Box
      component="span"
      sx={{ '& .katex': { fontSize: '1em' } }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

export default function ChatInput({ onSend, disabled = false }: ChatInputProps) {
  const inputRef = useRef<HTMLInputElement | HTMLTextAreaElement | null>(null)
  const [message, setMessage] = useState('')
  const [sketchPadOpen, setSketchPadOpen] = useState(false)
  const [attachedDrawing, setAttachedDrawing] = useState<StudentDrawing | null>(null)
  const mathPreview = formatPhysicsMathMarkdown(message)
  const showMathPreview = message.trim().length > 0 && mathPreview !== message

  const handleSend = () => {
    const trimmedMessage = message.trim()
    if ((trimmedMessage || attachedDrawing) && !disabled) {
      onSend(trimmedMessage, attachedDrawing || undefined)
      setMessage('')
      setAttachedDrawing(null)
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const insertLatexSnippet = (snippet: string, cursorOffset = 0) => {
    const input = inputRef.current
    const start = input?.selectionStart ?? message.length
    const end = input?.selectionEnd ?? message.length
    const nextMessage = `${message.slice(0, start)}${snippet}${message.slice(end)}`
    const cursorPosition = Math.max(0, start + snippet.length + cursorOffset)

    setMessage(nextMessage)
    window.requestAnimationFrame(() => {
      inputRef.current?.focus()
      inputRef.current?.setSelectionRange(cursorPosition, cursorPosition)
    })
  }

  return (
    <Paper
      elevation={3}
      sx={{
        p: 2,
        borderRadius: 0,
        borderTop: 1,
        borderColor: 'divider',
      }}
    >
      {attachedDrawing && (
        <Box
          sx={{
            mb: 1.5,
            p: 1,
            display: 'flex',
            alignItems: 'center',
            gap: 1.25,
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 2,
            bgcolor: 'grey.50',
          }}
        >
          <Box
            component="img"
            src={attachedDrawing.dataUrl}
            alt={attachedDrawing.title}
            sx={{
              width: 92,
              height: 54,
              objectFit: 'cover',
              borderRadius: 1,
              border: '1px solid',
              borderColor: 'divider',
              bgcolor: 'white',
            }}
          />
          <Box sx={{ minWidth: 0, flex: 1 }}>
            <Typography variant="body2" sx={{ fontWeight: 700 }}>
              Sketch attached
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Add a short note with the labels, axes, or equations before sending.
            </Typography>
          </Box>
          <IconButton aria-label="Remove attached sketch" size="small" onClick={() => setAttachedDrawing(null)}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      )}

      <Box
        sx={{
          mb: 1.5,
          p: 1.25,
          border: '1px solid',
          borderColor: 'primary.light',
          borderRadius: 2,
          bgcolor: 'rgba(25, 118, 210, 0.08)',
        }}
      >
        <Typography variant="body2" color="primary.dark" sx={{ display: 'block', mb: 0.75, fontWeight: 800 }}>
          Equation helper: palette
        </Typography>
        <Box sx={{ display: 'grid', gap: 1 }}>
          {LATEX_SNIPPETS.map((group) => (
            <Box
              key={group.title}
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', sm: '86px 1fr' },
                gap: 0.75,
                alignItems: 'center',
              }}
            >
              <Typography
                variant="caption"
                color="primary.dark"
                sx={{ fontWeight: 800, textTransform: 'uppercase', letterSpacing: 0.5 }}
              >
                {group.title}
              </Typography>
              <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap' }}>
                {group.snippets.map((snippet) => (
                  <Button
                    key={`${group.title}-${snippet.label}`}
                    size="small"
                    variant="outlined"
                    aria-label={`Insert ${snippet.label}`}
                    disabled={disabled}
                    onMouseDown={(event) => event.preventDefault()}
                    onClick={() => insertLatexSnippet(snippet.value, snippet.cursorOffset)}
                    sx={{
                      minWidth: 0,
                      px: 1.1,
                      py: 0.35,
                      borderRadius: 2,
                      bgcolor: 'background.paper',
                    }}
                  >
                    <RenderedMathSnippet expression={snippet.label} />
                  </Button>
                ))}
              </Box>
            </Box>
          ))}
        </Box>
      </Box>

      <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
        <Button
          variant={attachedDrawing ? 'contained' : 'outlined'}
          startIcon={<SketchIcon />}
          onClick={() => setSketchPadOpen(true)}
          disabled={disabled}
          sx={{ flexShrink: 0, borderRadius: 3, py: 1 }}
        >
          Sketch
        </Button>
        <TextField
          fullWidth
          multiline
          maxRows={4}
          placeholder={attachedDrawing ? 'Describe your sketch, labels, axes, or equations...' : 'Ask your physics question here...'}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          inputRef={inputRef}
          disabled={disabled}
          variant="outlined"
          size="small"
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: 3,
            },
          }}
        />
        <IconButton
          color="primary"
          onClick={handleSend}
          disabled={disabled || (!message.trim() && !attachedDrawing)}
          sx={{
            bgcolor: 'primary.main',
            color: 'white',
            '&:hover': { bgcolor: 'primary.dark' },
            '&.Mui-disabled': { bgcolor: 'grey.300' },
          }}
        >
          <SendIcon />
        </IconButton>
      </Box>
      {showMathPreview && (
        <Box
          sx={{
            mt: 1,
            px: 1.25,
            py: 0.9,
            border: '1px solid',
            borderColor: 'divider',
            borderRadius: 2,
            bgcolor: 'grey.50',
          }}
        >
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5, fontWeight: 700 }}>
            Equation preview
          </Typography>
          <Box
            sx={{
              '& p': { m: 0 },
              '& .katex': { fontSize: '1.05em' },
            }}
          >
            <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
              {mathPreview}
            </ReactMarkdown>
          </Box>
        </Box>
      )}
      <SketchPadDialog
        open={sketchPadOpen}
        onClose={() => setSketchPadOpen(false)}
        onAttach={(drawing) => setAttachedDrawing(drawing)}
      />
    </Paper>
  )
}
