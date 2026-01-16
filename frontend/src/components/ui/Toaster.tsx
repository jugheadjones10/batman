import * as React from 'react'
import * as Toast from '@radix-ui/react-toast'
import { X } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface ToastData {
  id: string
  title: string
  description?: string
  type?: 'default' | 'success' | 'error'
}

const ToastContext = React.createContext<{
  toast: (data: Omit<ToastData, 'id'>) => void
}>({ toast: () => {} })

export function useToast() {
  return React.useContext(ToastContext)
}

export function Toaster({ children }: { children?: React.ReactNode }) {
  const [toasts, setToasts] = React.useState<ToastData[]>([])

  const toast = React.useCallback((data: Omit<ToastData, 'id'>) => {
    const id = Math.random().toString(36).slice(2)
    setToasts((prev) => [...prev, { ...data, id }])
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id))
    }, 5000)
  }, [])

  const dismiss = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <Toast.Provider swipeDirection="right">
        <AnimatePresence>
          {toasts.map((t) => (
            <Toast.Root key={t.id} asChild forceMount>
              <motion.div
                initial={{ opacity: 0, x: 100 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 100 }}
                className={`
                  fixed bottom-4 right-4 z-50 min-w-[300px] max-w-md rounded-lg border p-4 shadow-lg
                  ${t.type === 'error' ? 'border-destructive bg-destructive/10' : ''}
                  ${t.type === 'success' ? 'border-green-500 bg-green-500/10' : ''}
                  ${!t.type || t.type === 'default' ? 'border-border bg-secondary' : ''}
                `}
              >
                <div className="flex items-start gap-3">
                  <div className="flex-1">
                    <Toast.Title className="font-semibold">{t.title}</Toast.Title>
                    {t.description && (
                      <Toast.Description className="mt-1 text-sm text-muted-foreground">
                        {t.description}
                      </Toast.Description>
                    )}
                  </div>
                  <Toast.Close asChild>
                    <button
                      onClick={() => dismiss(t.id)}
                      className="rounded p-1 hover:bg-muted"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </Toast.Close>
                </div>
              </motion.div>
            </Toast.Root>
          ))}
        </AnimatePresence>
        <Toast.Viewport />
      </Toast.Provider>
    </ToastContext.Provider>
  )
}

export { ToastContext }

