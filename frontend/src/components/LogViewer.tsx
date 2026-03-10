import { useEffect, useRef, useState, useCallback } from 'react'
import { Terminal, ArrowDown } from 'lucide-react'
import { Button } from '@/components/ui/Button'

interface LogViewerProps {
  url: string | null
  maxLines?: number
}

type ConnectionState = 'idle' | 'connecting' | 'streaming' | 'done' | 'error'
type StreamType = 'stdout' | 'stderr' | 'system'

interface LogLine {
  text: string
  stream: StreamType
}

export default function LogViewer({ url, maxLines = 5000 }: LogViewerProps) {
  const [lines, setLines] = useState<LogLine[]>([])
  const [connectionState, setConnectionState] = useState<ConnectionState>('idle')
  const [autoscrollOut, setAutoscrollOut] = useState(true)
  const [autoscrollErr, setAutoscrollErr] = useState(true)
  const outRef = useRef<HTMLDivElement>(null)
  const errRef = useRef<HTMLDivElement>(null)
  const eventSourceRef = useRef<EventSource | null>(null)

  const scrollToBottom = useCallback((el: HTMLDivElement | null) => {
    if (el) el.scrollTop = el.scrollHeight
  }, [])

  useEffect(() => {
    if (!url) {
      setConnectionState('idle')
      return
    }

    setLines([])
    setConnectionState('connecting')

    const es = new EventSource(url)
    eventSourceRef.current = es

    es.onopen = () => {
      setConnectionState('streaming')
    }

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        if (data.type === 'done') {
          setConnectionState('done')
          es.close()
          return
        }

        if (data.type === 'error') {
          setLines((prev) => [
            ...prev.slice(-maxLines + 1),
            { text: `[error] ${data.message}`, stream: 'stderr' as StreamType },
          ])
          setConnectionState('error')
          es.close()
          return
        }

        if (data.type === 'log' && data.line !== undefined) {
          const line = data.line as string
          let stream: StreamType =
            data.stream === 'stdout' || data.stream === 'stderr' || data.stream === 'system'
              ? data.stream
              : line.startsWith('[system]')
                ? 'system'
                : line.startsWith('[stderr]')
                  ? 'stderr'
                  : 'stdout'
          setLines((prev) => {
            const next = [...prev.slice(-maxLines + 1), { text: line, stream }]
            return next
          })
        }
      } catch {
        // Ignore malformed SSE messages
      }
    }

    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) {
        setConnectionState('done')
      } else {
        setConnectionState('error')
      }
    }

    return () => {
      es.close()
      eventSourceRef.current = null
    }
  }, [url, maxLines])

  // Double-rAF ensures the DOM has been laid out with the new content before scrolling
  useEffect(() => {
    const id = requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (autoscrollOut) scrollToBottom(outRef.current)
        if (autoscrollErr) scrollToBottom(errRef.current)
      })
    })
    return () => cancelAnimationFrame(id)
  }, [lines, scrollToBottom, autoscrollOut, autoscrollErr])

  const outLines = lines.filter((l) => l.stream === 'stdout' || l.stream === 'system')
  const errLines = lines.filter((l) => l.stream === 'stderr')

  const handleScrollOut = () => {
    if (!outRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = outRef.current
    const atBottom = scrollHeight - scrollTop - clientHeight < 40
    setAutoscrollOut(atBottom)
  }

  const handleScrollErr = () => {
    if (!errRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = errRef.current
    const atBottom = scrollHeight - scrollTop - clientHeight < 40
    setAutoscrollErr(atBottom)
  }

  const statusColors: Record<ConnectionState, string> = {
    idle: 'text-muted-foreground',
    connecting: 'text-yellow-500',
    streaming: 'text-green-500',
    done: 'text-muted-foreground',
    error: 'text-destructive',
  }

  const statusLabels: Record<ConnectionState, string> = {
    idle: 'No logs',
    connecting: 'Connecting...',
    streaming: 'Live',
    done: 'Finished',
    error: 'Error',
  }

  return (
    <div className="border border-border rounded-lg overflow-hidden bg-[#0d1117] flex flex-col h-[42rem]">
      <div className="flex items-center justify-between px-3 py-1.5 bg-muted/50 border-b border-border shrink-0">
        <div className="flex items-center gap-2">
          <Terminal className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-medium">Job Logs</span>
          <span className={`text-xs ${statusColors[connectionState]}`}>
            {connectionState === 'streaming' && (
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-green-500 mr-1 animate-pulse" />
            )}
            {statusLabels[connectionState]}
          </span>
        </div>
        <span className="text-xs text-muted-foreground">
          .out {outLines.length} · .err {errLines.length}
        </span>
      </div>

      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Left: stdout (.out) */}
        <div className="flex flex-col flex-1 min-w-0 border-r border-border overflow-hidden">
          <div className="shrink-0 px-2 py-1 bg-muted/30 border-b border-border flex items-center justify-between">
            <span className="text-xs font-medium text-muted-foreground">.out (stdout)</span>
            {!autoscrollOut && connectionState === 'streaming' && (
              <Button
                variant="ghost"
                size="sm"
                className="h-5 px-1.5 text-xs gap-0.5"
                onClick={() => {
                  setAutoscrollOut(true)
                  scrollToBottom(outRef.current)
                }}
              >
                <ArrowDown className="h-3 w-3" />
                Follow
              </Button>
            )}
          </div>
          <div
            ref={outRef}
            onScroll={handleScrollOut}
            className="flex-1 overflow-y-auto font-mono text-xs leading-5 p-3 select-text min-h-0"
          >
            {outLines.length === 0 && connectionState === 'idle' && (
              <p className="text-muted-foreground/50 text-center py-8">Select a running job to view logs</p>
            )}
            {outLines.length === 0 && connectionState === 'connecting' && (
              <p className="text-yellow-500/70 text-center py-8">Connecting to log stream...</p>
            )}
            {outLines.map((line, i) => (
              <div
                key={`out-${i}`}
                className={
                  line.stream === 'system' ? 'text-blue-400/90' : 'text-gray-300'
                }
              >
                {line.text}
              </div>
            ))}
          </div>
        </div>

        {/* Right: stderr (.err) */}
        <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
          <div className="shrink-0 px-2 py-1 bg-muted/30 border-b border-border flex items-center justify-between">
            <span className="text-xs font-medium text-muted-foreground">.err (stderr)</span>
            {!autoscrollErr && connectionState === 'streaming' && (
              <Button
                variant="ghost"
                size="sm"
                className="h-5 px-1.5 text-xs gap-0.5"
                onClick={() => {
                  setAutoscrollErr(true)
                  scrollToBottom(errRef.current)
                }}
              >
                <ArrowDown className="h-3 w-3" />
                Follow
              </Button>
            )}
          </div>
          <div
            ref={errRef}
            onScroll={handleScrollErr}
            className="flex-1 overflow-y-auto font-mono text-xs leading-5 p-3 select-text min-h-0 text-red-400/90"
          >
            {errLines.length === 0 && connectionState !== 'idle' && connectionState !== 'connecting' && (
              <p className="text-muted-foreground/50 text-center py-8">No stderr output yet</p>
            )}
            {errLines.map((line, i) => (
              <div key={`err-${i}`}>{line.text}</div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
