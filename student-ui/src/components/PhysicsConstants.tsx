import { useState } from 'react'
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Table,
  TableBody,
  TableRow,
  TableCell,
} from '@mui/material'
import { ExpandMore as ExpandMoreIcon } from '@mui/icons-material'

const PHYSICS_CONSTANTS = [
  { name: 'g (gravity)', value: '9.81', unit: 'm/s²' },
  { name: 'c (speed of light)', value: '3.00 × 10⁸', unit: 'm/s' },
  { name: 'e (electron charge)', value: '1.60 × 10⁻¹⁹', unit: 'C' },
  { name: 'h (Planck constant)', value: '6.63 × 10⁻³⁴', unit: 'J·s' },
  { name: 'k_B (Boltzmann)', value: '1.38 × 10⁻²³', unit: 'J/K' },
  { name: 'N_A (Avogadro)', value: '6.02 × 10²³', unit: 'mol⁻¹' },
]

export default function PhysicsConstants() {
  const [expanded, setExpanded] = useState(false)

  return (
    <Accordion
      expanded={expanded}
      onChange={() => setExpanded(!expanded)}
      sx={{ mb: 2, bgcolor: 'transparent', boxShadow: 'none' }}
    >
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography variant="subtitle2">📋 Physics Constants</Typography>
      </AccordionSummary>
      <AccordionDetails sx={{ p: 0 }}>
        <Table size="small">
          <TableBody>
            {PHYSICS_CONSTANTS.map((constant) => (
              <TableRow key={constant.name}>
                <TableCell sx={{ py: 0.5, fontSize: '0.75rem', fontWeight: 500 }}>
                  {constant.name}
                </TableCell>
                <TableCell sx={{ py: 0.5, fontSize: '0.75rem' }} align="right">
                  {constant.value} {constant.unit}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </AccordionDetails>
    </Accordion>
  )
}
