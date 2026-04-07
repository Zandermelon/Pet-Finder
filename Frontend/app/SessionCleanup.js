'use client'
import { useEffect } from 'react'

const HEARTBEAT_MS = 5 * 60 * 1000 // 5 minutes

export default function SessionCleanup() {
  useEffect(() => {
    function cleanup() {
      const sid = localStorage.getItem('session_id')
      if (!sid) return
      // keepalive ensures the request completes even as the page unloads
      fetch(`http://localhost:8000/api/session/${sid}`, {
        method: 'DELETE',
        keepalive: true
      })
      localStorage.removeItem('session_id')
    }

    function sendHeartbeat() {
      const sid = localStorage.getItem('session_id')
      if (!sid) return
      fetch(`http://localhost:8000/api/session/${sid}/heartbeat`, { method: 'POST' })
        .catch(() => {})
    }

    sendHeartbeat()
    const interval = setInterval(sendHeartbeat, HEARTBEAT_MS)
    window.addEventListener('beforeunload', cleanup)

    return () => {
      clearInterval(interval)
      window.removeEventListener('beforeunload', cleanup)
    }
  }, [])

  return null
}
