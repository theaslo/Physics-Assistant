import {
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  Chip,
} from '@mui/material'
import { useStore } from '../stores/chat-store'
import { getAgentColor } from '../themes/uconn-theme'

export default function AgentSelector() {
  const { agents, selectedAgent, selectAgent } = useStore()

  const handleChange = (agentId: string) => {
    if (agentId) {
      selectAgent(agentId)
    }
  }

  const selectedAgentInfo = agents.find((a) => a.agent_id === selectedAgent)

  return (
    <Box sx={{ mb: 3 }}>
      <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
        Select a Physics Agent:
      </Typography>

      <FormControl fullWidth size="small">
        <InputLabel id="agent-select-label">Choose Agent</InputLabel>
        <Select
          labelId="agent-select-label"
          value={selectedAgent || ''}
          label="Choose Agent"
          onChange={(e) => handleChange(e.target.value)}
        >
          {agents.map((agent) => (
            <MenuItem key={agent.agent_id} value={agent.agent_id}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <span>{agent.icon}</span>
                <span>{agent.name}</span>
              </Box>
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* Selected Agent Info */}
      {selectedAgentInfo && (
        <Box
          sx={{
            mt: 2,
            p: 1.5,
            borderRadius: 2,
            bgcolor: `${getAgentColor(selectedAgent!)}15`,
            borderLeft: `4px solid ${getAgentColor(selectedAgent!)}`,
          }}
        >
          <Typography variant="subtitle2" fontWeight="bold">
            {selectedAgentInfo.icon} {selectedAgentInfo.name}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {selectedAgentInfo.description}
          </Typography>
        </Box>
      )}

      {/* Quick agent chips */}
      <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
        {agents.slice(0, 3).map((agent) => (
          <Chip
            key={agent.agent_id}
            label={agent.icon}
            size="small"
            variant={selectedAgent === agent.agent_id ? 'filled' : 'outlined'}
            onClick={() => handleChange(agent.agent_id)}
            sx={{
              cursor: 'pointer',
              borderColor: getAgentColor(agent.agent_id),
              bgcolor:
                selectedAgent === agent.agent_id
                  ? getAgentColor(agent.agent_id)
                  : 'transparent',
              color:
                selectedAgent === agent.agent_id
                  ? 'white'
                  : getAgentColor(agent.agent_id),
            }}
          />
        ))}
      </Box>
    </Box>
  )
}
