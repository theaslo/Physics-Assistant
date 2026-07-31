import { useEffect, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  LinearProgress,
  Radio,
  RadioGroup,
  Typography,
} from '@mui/material'
import {
  CheckCircle as CheckCircleIcon,
  ArrowForward as ArrowForwardIcon,
  Quiz as QuizIcon,
} from '@mui/icons-material'

export interface HitlChoice {
  id: string
  text: string
}

export interface HitlQuestion {
  id: string
  title?: string
  prompt: string
  choices: HitlChoice[]
  leg?: number
  correctChoiceId?: string
  feedback?: string
}

export interface HitlResult {
  questionId: string
  selectedChoiceId: string
  isCorrect: boolean
  attemptId?: string
  maxAttemptsReached?: boolean
}

export type HitlDialogState =
  | 'loading_question'
  | 'awaiting_answer'
  | 'submitting_answer'
  | 'showing_feedback'
  | 'completed'
  | 'error'

interface HumanInTheLoopDialogProps {
  open: boolean
  agentName: string
  question: HitlQuestion | null
  state: HitlDialogState
  feedback: string | null
  feedbackSeverity: 'success' | 'warning' | 'error'
  error: string | null
  onSubmit: (selectedChoiceId: string) => void
  onRetry: () => void
  onContinue: () => void
}

export default function HumanInTheLoopDialog({
  open,
  agentName,
  question,
  state,
  feedback,
  feedbackSeverity,
  error,
  onSubmit,
  onRetry,
  onContinue,
}: HumanInTheLoopDialogProps) {
  const [selectedChoiceId, setSelectedChoiceId] = useState('')

  const isLoading = state === 'loading_question'
  const isSubmitting = state === 'submitting_answer'
  const isBusy = isLoading || isSubmitting || state === 'showing_feedback' || state === 'completed'
  const canSubmit = Boolean(question && selectedChoiceId && !isBusy)

  useEffect(() => {
    setSelectedChoiceId('')
  }, [question?.id, open])

  const handleSubmit = () => {
    if (!question || !selectedChoiceId) return
    onSubmit(selectedChoiceId)
  }

  return (
    <Dialog
      open={open}
      disableEscapeKeyDown
      fullWidth
      maxWidth="sm"
      PaperProps={{
        sx: {
          borderRadius: 2,
        },
      }}
    >
      <DialogTitle sx={{ display: 'flex', gap: 1.25, alignItems: 'center', pb: 1 }}>
        <QuizIcon color="primary" />
        <Box>
          <Typography variant="h6" component="div">
            {question?.title || 'Reasoning Check'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {question?.leg ? `${agentName} · Step ${question.leg}` : agentName}
          </Typography>
        </Box>
      </DialogTitle>

      {(isLoading || isSubmitting) && <LinearProgress />}

      <DialogContent sx={{ pt: 3 }}>
        {isLoading && (
          <Typography variant="body2" color="text.secondary">
            Loading the next checkpoint...
          </Typography>
        )}

        {state === 'error' && (
          <Alert severity="error">
            {error || 'The checkpoint could not be loaded. Please try again.'}
          </Alert>
        )}

        {question && state !== 'error' && (
          <>
            <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 2 }}>
              {question.prompt}
            </Typography>

            <RadioGroup
              value={selectedChoiceId}
              onChange={(event) => setSelectedChoiceId(event.target.value)}
              sx={{ gap: 1 }}
            >
              {question.choices.map((choice) => (
                <FormControlLabel
                  key={choice.id}
                  value={choice.id}
                  disabled={isBusy}
                  control={<Radio />}
                  label={choice.text}
                  sx={{
                    m: 0,
                    px: 1.5,
                    py: 1,
                    border: 1,
                    borderColor: selectedChoiceId === choice.id ? 'primary.main' : 'divider',
                    borderRadius: 1,
                    bgcolor: selectedChoiceId === choice.id ? 'action.selected' : 'background.paper',
                  }}
                />
              ))}
            </RadioGroup>
          </>
        )}

        {feedback && (
          <Alert severity={feedbackSeverity} sx={{ mt: 2 }}>
            {feedback}
          </Alert>
        )}
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 2 }}>
        {state === 'error' && (
          <Button onClick={onRetry} variant="outlined">
            Retry
          </Button>
        )}
        {state === 'showing_feedback' ? (
          <Button
            onClick={onContinue}
            variant="contained"
            startIcon={<ArrowForwardIcon />}
          >
            Next question
          </Button>
        ) : (
          <Button
            onClick={handleSubmit}
            variant="contained"
            disabled={!canSubmit}
            startIcon={<CheckCircleIcon />}
          >
            {isSubmitting ? 'Checking...' : 'Check'}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  )
}
