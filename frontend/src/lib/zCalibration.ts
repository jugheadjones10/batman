import type { BoundingBox, Detection, ZCalibration } from '@/types'

export function measurementSizePxForBox(
  cal: ZCalibration | null | undefined,
  box: BoundingBox,
  vw: number,
  vh: number,
  className?: string,
): number {
  const baseSize = Math.max(box.width * vw, box.height * vh)
  if (
    cal?.measurement_source !== 'round_feature_equivalent_length' ||
    className !== cal.reference_class
  ) {
    return baseSize
  }

  const ratio = cal.equivalent_size_ratio
  if (ratio == null || ratio <= 0) return baseSize
  return baseSize * ratio
}

export function computeZForBox(
  cal: ZCalibration | null | undefined,
  box: BoundingBox,
  vw: number,
  vh: number,
  className?: string,
): number | null {
  if (!cal || !cal.model) return null
  const s = measurementSizePxForBox(cal, box, vw, vh, className)
  if (s <= 0) return null
  if (cal.model.type === 'k_over_s' && cal.model.k != null) return cal.model.k / s
  if (cal.model.type === 'linear_inv' && cal.model.m != null && cal.model.c != null) {
    return cal.model.m / s + cal.model.c
  }
  return null
}

export function computeZForDetection(
  cal: ZCalibration | null | undefined,
  det: Detection,
  vw: number,
  vh: number,
): number | null {
  return computeZForBox(cal, det.box, vw, vh, det.class_name)
}
