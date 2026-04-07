'use client'
import { useState, useEffect } from 'react'
import Link from 'next/link'

export default function Dashboard() {
  const [frames, setFrames] = useState([])
  const [loading, setLoading] = useState(true)
  const [sessionId, setSessionId] = useState(null)
  const [pets, setPets] = useState([])

  useEffect(() => {
    const sid = localStorage.getItem('session_id')
    setSessionId(sid)

    if (sid) {
      fetch(`http://localhost:8000/api/pets/${sid}`)
        .then(r => r.json())
        .then(data => setPets(data.pets || []))
        .catch(() => setPets([]))

      fetch(`http://localhost:8000/api/spotted-frames/${sid}`)
        .then(r => r.json())
        .then(data => {
          setFrames(data.frames)
          setLoading(false)
        })
        .catch(() => setLoading(false))
    } else {
      setLoading(false)
    }
  }, [])

  async function clearSpotted() {
    if (!sessionId) return
    if (!confirm('Clear all spotted frames?')) return

    await fetch(`http://localhost:8000/api/clear-spotted/${sessionId}`, {
      method: 'DELETE'
    })
    setFrames([])
  }

  return (
    <main className="main">

      {/* Nav */}
      <nav className="nav">
        <Link href="/" className="nav-logo">🐾 PetFinder</Link>
        <div className="nav-links">
          <Link href="/register">Register Pet</Link>
          <Link href="/scan" className="nav-cta">Scan Footage</Link>
        </div>
      </nav>

      <div className="page-container" style={{ maxWidth: '1100px' }}>
        <div className="page-header">
          <h1 className="page-title">Dashboard</h1>
          <p className="page-sub">All spotted frames from your scans</p>
        </div>

        {/* No session */}
        {!sessionId && !loading && (
          <div className="empty-state">
            <div className="empty-icon">🐾</div>
            <h2>No pet registered yet</h2>
            <p>Register your pet first to start scanning for them</p>
            <Link href="/register" className="btn-primary" style={{ marginTop: '1.5rem', display: 'inline-block' }}>
              Register Your Pet →
            </Link>
          </div>
        )}

        {/* Status cards */}
        {sessionId && (
          <div className="dashboard-stats">
            <div className="dashboard-stat-card">
              <div className="dashboard-stat-icon">🧠</div>
              <div>
                <div className="dashboard-stat-value">
                  {pets.length > 0 ? pets.length : 'None'}
                </div>
                <div className="dashboard-stat-label">Registered pet{pets.length !== 1 ? 's' : ''}</div>
              </div>
            </div>
            <div className="dashboard-stat-card">
              <div className="dashboard-stat-icon">📸</div>
              <div>
                <div className="dashboard-stat-value">{frames.length}</div>
                <div className="dashboard-stat-label">Total sightings</div>
              </div>
            </div>
            <div className="dashboard-stat-card">
              <div className="dashboard-stat-icon">🔍</div>
              <div>
                <div className="dashboard-stat-value">
                  {pets.length > 0 ? 'Ready' : 'Setup needed'}
                </div>
                <div className="dashboard-stat-label">Scan status</div>
              </div>
            </div>
          </div>
        )}

        {/* Registered pets */}
        {sessionId && pets.length > 0 && (
          <div className="spotted-section">
            <div className="spotted-header">
              <h2 className="section-title" style={{ marginBottom: 0 }}>Registered pets</h2>
              <Link href="/register" className="btn-secondary" style={{ fontSize: '0.85rem', padding: '0.5rem 1rem' }}>
                + Add pet
              </Link>
            </div>
            <div className="pet-tabs" style={{ marginTop: '1rem' }}>
              {pets.map(pet => (
                <div key={pet.slug} className="pet-tab active" style={{ cursor: 'default' }}>
                  {pet.name}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Quick actions */}
        {sessionId && (
          <div className="quick-actions">
            <Link href="/scan" className="action-card">
              <div className="action-icon">🎥</div>
              <div>
                <div className="action-title">Scan video</div>
                <div className="action-sub">Upload footage to search</div>
              </div>
              <span className="action-arrow">→</span>
            </Link>
            <Link href="/scan" className="action-card">
              <div className="action-icon">📷</div>
              <div>
                <div className="action-title">Live webcam</div>
                <div className="action-sub">Watch in real time</div>
              </div>
              <span className="action-arrow">→</span>
            </Link>
            <Link href="/register" className="action-card">
              <div className="action-icon">➕</div>
              <div>
                <div className="action-title">Add another pet</div>
                <div className="action-sub">Register a new profile</div>
              </div>
              <span className="action-arrow">→</span>
            </Link>
          </div>
        )}

        {/* Spotted frames */}
        {sessionId && (
          <div className="spotted-section">
            <div className="spotted-header">
              <h2 className="section-title" style={{ marginBottom: 0 }}>
                Spotted frames
              </h2>
              {frames.length > 0 && (
                <button
                  className="btn-secondary"
                  style={{ fontSize: '0.85rem', padding: '0.5rem 1rem' }}
                  onClick={clearSpotted}
                >
                  Clear all
                </button>
              )}
            </div>

            {loading && (
              <div className="processing-box">
                <div className="spinner"/>
                <h2>Loading...</h2>
              </div>
            )}

            {!loading && frames.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon">📭</div>
                <h2>No sightings yet</h2>
                <p>Start scanning footage to find your pet</p>
                <Link
                  href="/scan"
                  className="btn-primary"
                  style={{ marginTop: '1.5rem', display: 'inline-block' }}
                >
                  Start Scanning →
                </Link>
              </div>
            )}

            {!loading && frames.length > 0 && (
              <div className="sightings-grid">
                {frames.map((frame, i) => (
                  <div key={i} className="sighting-card">
                    <img
                      src={`http://localhost:8000/sessions/${sessionId}/spotted/${frame.filename}`}
                      alt={`Sighting ${i + 1}`}
                    />
                    <div className="sighting-label">
                      {frame.filename.replace('spotted_', '').replace('.jpg', '')}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  )
}