import { useEffect, type RefObject } from 'react'

/**
 * Mirror play / pause / seek / playback-rate / time drift from a master
 * <video> element to a mirror <video> element. Both videos must point at the
 * same source for the sync to look right; the hook just wires up the event
 * listeners and re-attaches when `resetKey` changes (typically the video
 * element's React `key` — src change, cache-buster bump, inference id switch).
 *
 * A re-entrancy guard prevents feedback when we programmatically set
 * `currentTime` / `playbackRate` on the mirror (which would otherwise fire
 * `seeked` / `ratechange` and loop back).
 */
export function useSyncedVideoMirror(
  masterRef: RefObject<HTMLVideoElement>,
  mirrorRef: RefObject<HTMLVideoElement>,
  enabled: boolean,
  resetKey?: string | number,
): void {
  useEffect(() => {
    if (!enabled) return
    const master = masterRef.current
    const mirror = mirrorRef.current
    if (!master || !mirror) return

    let syncing = false
    const guard = (fn: () => void) => {
      if (syncing) return
      syncing = true
      try {
        fn()
      } finally {
        queueMicrotask(() => {
          syncing = false
        })
      }
    }

    const onPlay = () => guard(() => { void mirror.play().catch(() => {}) })
    const onPause = () => guard(() => { mirror.pause() })
    const onSeeked = () => guard(() => { mirror.currentTime = master.currentTime })
    const onRate = () => guard(() => { mirror.playbackRate = master.playbackRate })
    const onTime = () => {
      if (Math.abs(mirror.currentTime - master.currentTime) > 0.05) {
        guard(() => { mirror.currentTime = master.currentTime })
      }
    }

    master.addEventListener('play', onPlay)
    master.addEventListener('pause', onPause)
    master.addEventListener('seeked', onSeeked)
    master.addEventListener('ratechange', onRate)
    master.addEventListener('timeupdate', onTime)

    if (!master.paused) void mirror.play().catch(() => {})
    mirror.currentTime = master.currentTime
    mirror.playbackRate = master.playbackRate

    return () => {
      master.removeEventListener('play', onPlay)
      master.removeEventListener('pause', onPause)
      master.removeEventListener('seeked', onSeeked)
      master.removeEventListener('ratechange', onRate)
      master.removeEventListener('timeupdate', onTime)
    }
  }, [enabled, resetKey, masterRef, mirrorRef])
}
