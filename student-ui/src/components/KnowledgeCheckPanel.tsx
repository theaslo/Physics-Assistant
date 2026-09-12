import { Box, Button, Chip, Stack, Typography } from '@mui/material'
import type { PendingKnowledgeCheck } from '../stores/chat-store'

interface KnowledgeCheckPanelProps {
  check: PendingKnowledgeCheck
  onSelectOption: (optionId: string) => void
  disabled?: boolean
}

export default function KnowledgeCheckPanel({
  check,
  onSelectOption,
  disabled = false,
}: KnowledgeCheckPanelProps) {
  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        p: 3,
      }}
    >
      <Typography variant="h6" sx={{ fontWeight: 700 }}>
        Quick Knowledge Check
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
        Select one answer. If it is not correct, we will pause and review it before continuing.
      </Typography>

      <Stack direction="row" spacing={1} sx={{ mt: 1.5, flexWrap: 'wrap', gap: 1 }}>
        <Chip label={check.agentId.replace('_agent', '')} size="small" />
        <Chip label={check.conceptTag} size="small" color="primary" />
      </Stack>

      <Typography variant="subtitle1" sx={{ mt: 2.5, fontWeight: 600 }}>
        {check.question}
      </Typography>

      <Stack spacing={1.5} sx={{ mt: 2 }}>
        {check.options.map((option) => (
          <Button
            key={option.id}
            fullWidth
            variant="outlined"
            disabled={disabled}
            onClick={() => onSelectOption(option.id)}
            sx={{
              justifyContent: 'flex-start',
              textAlign: 'left',
              py: 1.5,
              px: 1.5,
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, width: '100%' }}>
              <Box component="span" sx={{ fontWeight: 700, minWidth: 20 }}>
                {option.id}.
              </Box>
              <Box component="span">{option.text}</Box>
            </Box>
          </Button>
        ))}
      </Stack>

      <Typography variant="caption" color="text.secondary" sx={{ mt: 'auto', pt: 2 }}>
        One attempt is allowed for each knowledge check.
      </Typography>
    </Box>
  )
}
