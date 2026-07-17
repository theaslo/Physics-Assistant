import { useState } from 'react'
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
  Quiz as QuizIcon,
} from '@mui/icons-material'

export interface HitlChoice {
  id: string
  text: string
}

export interface HitlQuestion {
  id: string
  title: string
  prompt: string
  choices: HitlChoice[]
  correctChoiceId: string
  feedback: string
}

export interface HitlResult {
  questionId: string
  selectedChoiceId: string
  correctChoiceId: string
}

interface HumanInTheLoopDialogProps {
  open: boolean
  agentName: string
  questions: HitlQuestion[]
  onCancel: () => void
  onComplete: (results: HitlResult[]) => void
}

export default function HumanInTheLoopDialog({
  open,
  agentName,
  questions,
  onCancel,
  onComplete,
}: HumanInTheLoopDialogProps) {
  const [questionIndex, setQuestionIndex] = useState(0)
  const [selectedChoiceId, setSelectedChoiceId] = useState('')
  const [feedback, setFeedback] = useState<string | null>(null)
  const [results, setResults] = useState<HitlResult[]>([])

  const question = questions[questionIndex]
  const progress = questions.length > 0 ? ((questionIndex + 1) / questions.length) * 100 : 0

  const resetDialog = () => {
    setQuestionIndex(0)
    setSelectedChoiceId('')
    setFeedback(null)
    setResults([])
  }

  const handleCancel = () => {
    resetDialog()
    onCancel()
  }

  const handleSubmit = () => {
    if (!question || !selectedChoiceId) return

    const isCorrect = selectedChoiceId === question.correctChoiceId
    const nextResults = [
      ...results,
      {
        questionId: question.id,
        selectedChoiceId,
        correctChoiceId: question.correctChoiceId,
      },
    ]

    if (isCorrect) {
      resetDialog()
      onComplete(nextResults)
      return
    }

    setResults(nextResults)
    setFeedback(question.feedback)
    setSelectedChoiceId('')

    if (questionIndex < questions.length - 1) {
      window.setTimeout(() => {
        setQuestionIndex((current) => current + 1)
        setFeedback(null)
      }, 700)
    }
  }

  if (!question) return null

  return (
    <Dialog
      open={open}
      onClose={handleCancel}
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
            {question.title}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {agentName}
          </Typography>
        </Box>
      </DialogTitle>

      <LinearProgress variant="determinate" value={progress} />

      <DialogContent sx={{ pt: 3 }}>
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

        {feedback && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            {feedback}
          </Alert>
        )}
      </DialogContent>

      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={handleCancel} color="inherit">
          Cancel
        </Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          disabled={!selectedChoiceId}
          startIcon={<CheckCircleIcon />}
        >
          Check
        </Button>
      </DialogActions>
    </Dialog>
  )
}
