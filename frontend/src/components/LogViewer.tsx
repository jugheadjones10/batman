import { useEffect, useRef, useState, useCallback } from 'react'
import { Terminal, ArrowDown } from 'lucide-react'
import { Button } from '@/components/ui/Button'

interface LogViewerProps {
  url: string | null
  maxLines?: number
}

type ConnectionState = 'idle' | 'connecting' | 'streaming' | 'done' | 'error'

export default function LogViewer({ url, maxLines = 5000 }: LogViewerProps) {
  const [lines, setLines] = useState<{ text: string; isStderr: boolean; isSystem: boolean }[]>([])
  const [connectionState, setConnectionState] = useState<ConnectionState>('idle')
  const [autoscroll, setAutoscroll] = useState(true)
  const containerRef = useRef<HTMLDivElement>(null)
  const eventSourceRef = useRef<EventSource | null>(null)

  const scrollToBottom = useCallback(() => {
    if (containerRef.current && autoscroll) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [autoscroll])

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
            { text: `[error] ${data.message}`, isStderr: true, isSystem: true },
          ])
          setConnectionState('error')
          es.close()
          return
        }

        if (data.type === 'log' && data.line !== undefined) {
          const line = data.line as string
          const isStderr = line.startsWith('[stderr]')
          const isSystem = line.startsWith('[system]')

          setLines((prev) => [
            ...prev.slice(-maxLines + 1),
            { text: line, isStderr, isSystem },
          ])
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

  useEffect(() => {
    scrollToBottom()
  }, [lines, scrollToBottom])

  const handleScroll = () => {
    if (!containerRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current
    const atBottom = scrollHeight - scrollTop - clientHeight < 40
    setAutoscroll(atBottom)
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
    <div className="border border-border rounded-lg overflow-hidden bg-[#0d1117]">
      <div className="flex items-center justify-between px-3 py-1.5 bg-muted/50 border-b border-border">
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
        <div className="flex items-center gap-1">
          {!autoscroll && connectionState === 'streaming' && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-xs gap-1"
              onClick={() => {
                setAutoscroll(true)
                scrollToBottom()
              }}
            >
              <ArrowDown className="h-3 w-3" />
              Follow
            </Button>
          )}
          <span className="text-xs text-muted-foreground">{lines.length} lines</span>
        </div>
      </div>

      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="h-80 overflow-y-auto font-mono text-xs leading-5 p-3 select-text"
      >
        {lines.length === 0 && connectionState === 'idle' && (
          <p className="text-muted-foreground/50 text-center py-8">
            Select a running job to view logs
          </p>
        )}
        {lines.length === 0 && connectionState === 'connecting' && (
          <p className="text-yellow-500/70 text-center py-8">Connecting to log stream...</p>
        )}
        {lines.map((line, i) => (
          <div
            key={i}
            className={
              line.isStderr
                ? 'text-red-400/90'
                : line.isSystem
                  ? 'text-blue-400/80'
                  : 'text-gray-300'
            }
          >
            {line.text}
          </div>
        ))}
      </div>
    </div>
  )
}
