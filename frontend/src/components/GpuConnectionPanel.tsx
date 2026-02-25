import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Loader2, Server, Wifi, WifiOff, LogOut } from 'lucide-react'
import { api } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { useToast } from '@/components/ui/Toaster'

export default function GpuConnectionPanel() {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  const [password, setPassword] = useState('')

  const { data: gpuStatus } = useQuery({
    queryKey: ['gpu-status'],
    queryFn: () => api.gpu.getStatus(),
    refetchInterval: 10000,
  })

  const connectMutation = useMutation({
    mutationFn: (pw: string) => api.gpu.connect(pw),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['gpu-status'] })
      toast({ title: 'Connected to GPU cluster', type: 'success' })
      setPassword('')
    },
    onError: (error: Error) => {
      toast({ title: 'Connection failed', description: error.message, type: 'error' })
    },
  })

  const disconnectMutation = useMutation({
    mutationFn: () => api.gpu.disconnect(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['gpu-status'] })
      toast({ title: 'Disconnected', type: 'success' })
    },
  })

  const connected = gpuStatus?.connected ?? false

  if (connected) {
    return (
      <div className="flex items-center gap-3 px-4 py-2 bg-green-500/10 border border-green-500/20 rounded-lg">
        <Wifi className="h-4 w-4 text-green-500" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-green-600 dark:text-green-400">
            GPU Cluster Connected
          </p>
          <p className="text-xs text-muted-foreground truncate">
            {gpuStatus?.user}@{gpuStatus?.host}
          </p>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => disconnectMutation.mutate()}
          disabled={disconnectMutation.isPending}
          className="text-muted-foreground hover:text-destructive"
        >
          <LogOut className="h-4 w-4" />
        </Button>
      </div>
    )
  }

  return (
    <div className="px-4 py-3 bg-muted/50 border border-border rounded-lg">
      <div className="flex items-center gap-2 mb-2">
        <WifiOff className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium">GPU Cluster</span>
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault()
          if (password) connectMutation.mutate(password)
        }}
        className="flex gap-2"
      >
        <Input
          type="password"
          placeholder="SSH password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="h-8 text-sm"
        />
        <Button
          type="submit"
          size="sm"
          disabled={!password || connectMutation.isPending}
          className="gap-1.5 whitespace-nowrap"
        >
          {connectMutation.isPending ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Server className="h-3.5 w-3.5" />
          )}
          Connect
        </Button>
      </form>
    </div>
  )
}
