import { Line } from '@react-three/drei'
import type { Vec3 } from './layout'

export interface ResidualLineProps {
  start: Vec3
  end: Vec3
}

export function ResidualLine({ start, end }: ResidualLineProps) {
  return (
    <Line
      points={[start, end]}
      color="#7b7b75"
      lineWidth={1.6}
      dashed={false}
    />
  )
}
