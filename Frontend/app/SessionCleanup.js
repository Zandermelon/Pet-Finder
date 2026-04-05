'use client'
import { useEffect } from 'react'

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

    window.addEventListener('beforeunload', cleanup)
    return () => window.removeEventListener('beforeunload', cleanup)
  }, [])

  return null
}
