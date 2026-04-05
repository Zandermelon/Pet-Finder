'use client'
import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'

export default function Scan() {
  const [mode, setMode] = useState('video') // video or webcam
  const [file, setFile] = useState(null)
  const [step, setStep] = useState('upload')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [sightings, setSightings] = useState([])
  const [profileExists, setProfileExists] = useState(null)
  const [webcamActive, setWebcamActive] = useState(false)
  const [checkCount, setCheckCount] = useState(0)
  const [streak, setStreak] = useState(0)

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const intervalRef = useRef(null)

  useEffect(() => {
    fetch('http://localhost:8000/api/profile-status')
      .then(r => r.json())
      .then(data => setProfileExists(data.profile_exists))
      .catch(() => setProfileExists(false))
  }, [])

  // Cleanup webcam on unmount
  useEffect(() => {
    return () => stopWebcam()
  }, [])

  function handleFileChange(e) {
    setFile(e.target.files[0])
    setError('')
  }

  function handleDrop(e) {
    e.preventDefault()
    setFile(e.dataTransfer.files[0])
    setError('')
  }

  async function handleScanVideo() {
    if (!file) {
      setError('Please select a video file first')
      return
    }
    try {
      setStep('scanning')
      setMessage('Scanning video for your pet...')

      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch('http://localhost:8000/api/scan-video', {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail)

      setSightings(data.spotted_urls)
      setStep('done')
      setMessage(data.message)

    } catch (err) {
      setError(err.message)
      setStep('upload')
    }
  }

  async function startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      streamRef.current = stream
      setWebcamActive(true)
      setStep('webcam')
      setCheckCount(0)
      setSightings([])

      // Wait for video element to mount after step change
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }

        intervalRef.current = setInterval(async () => {
          await checkFrame()
        }, 3000)
      }, 100)

    } catch (err) {
      setError('Could not access webcam: ' + err.message)
    }
  }

  function stopWebcam() {
    if (intervalRef.current) clearInterval(intervalRef.current)
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }
    setWebcamActive(false)
    setStep('upload')
  }

  async function checkFrame() {
    if (!videoRef.current || !canvasRef.current) return

    // Draw current frame to canvas
    const canvas = canvasRef.current
    const video = videoRef.current
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)

    // Convert canvas to blob and send to backend
    canvas.toBlob(async (blob) => {
      if (!blob) return

      setCheckCount(c => c + 1)

      const formData = new FormData()
      formData.append('file', blob, 'frame.jpg')

      try {
        const res = await fetch('http://localhost:8000/api/match-image', {
          method: 'POST',
          body: formData
        })
        const data = await res.json()

        if (data.found) {
          setStreak(s => s + 1)
          // Add the frame to sightings
          const url = URL.createObjectURL(blob)
          setSightings(prev => [url, ...prev])
        } else {
          setStreak(0)
        }
      } catch (err) {
        console.error('Frame check failed:', err)
      }
    }, 'image/jpeg', 0.8)
  }

  return (
    <main className="main">

      {/* Nav */}
      <nav className="nav">
        <Link href="/" className="nav-logo">🐾 PetFinder</Link>
        <div className="nav-links">
          <Link href="/register">Register Pet</Link>
          <Link href="/dashboard">Dashboard</Link>
        </div>
      </nav>

      <div className="page-container">
        <div className="page-header">
          <h1 className="page-title">Scan footage</h1>
          <p className="page-sub">Upload video footage or use your webcam to find your pet</p>
        </div>

        {/* No profile warning */}
        {profileExists === false && (
          <div className="warning-box">
            ⚠️ No pet profile found.{' '}
            <Link href="/register" style={{ color: 'var(--accent)' }}>
              Register your pet first →
            </Link>
          </div>
        )}

        {/* Mode switcher */}
        {step === 'upload' && (
          <div className="mode-switcher">
            <button
              className={mode === 'video' ? 'mode-btn active' : 'mode-btn'}
              onClick={() => setMode('video')}
            >
              🎥 Upload Video
            </button>
            <button
              className={mode === 'webcam' ? 'mode-btn active' : 'mode-btn'}
              onClick={() => setMode('webcam')}
            >
              📷 Live Webcam
            </button>
          </div>
        )}

        {/* Video upload mode */}
        {step === 'upload' && mode === 'video' && (
          <div className="upload-section">
            <div
              className="dropzone"
              onDrop={handleDrop}
              onDragOver={e => e.preventDefault()}
              onClick={() => document.getElementById('video-input').click()}
            >
              <div className="dropzone-icon">🎥</div>
              <p className="dropzone-title">Drop video here or click to browse</p>
              <p className="dropzone-sub">Supports MP4, MOV, AVI — any security camera footage</p>
              {file && <div className="file-count">📹 {file.name}</div>}
              <input
                id="video-input"
                type="file"
                accept="video/*"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
            </div>

            {error && <div className="error-box">{error}</div>}

            <div className="tips-box">
              <h3>How scanning works</h3>
              <ul>
                <li>🔍 AI checks a frame every few seconds</li>
                <li>🐱 YOLO finds any cats or dogs in each frame</li>
                <li>🧠 DINO compares them to your pet's profile</li>
                <li>📸 Matching frames are saved as screenshots</li>
              </ul>
            </div>

            <button
              className="btn-primary btn-large btn-full"
              onClick={handleScanVideo}
              disabled={!file || profileExists === false}
            >
              Start Scanning →
            </button>
          </div>
        )}

        {/* Webcam mode */}
        {step === 'upload' && mode === 'webcam' && (
          <div className="upload-section">
            <div className="tips-box">
              <h3>Live webcam scanning</h3>
              <ul>
                <li>📷 Your webcam feed is checked every 3 seconds</li>
                <li>🔒 Video never leaves your browser — only frames are sent</li>
                <li>🚨 Matching frames appear below in real time</li>
                <li>⏹️ Click Stop to end the scan</li>
              </ul>
            </div>

            {error && <div className="error-box">{error}</div>}

            <button
              className="btn-primary btn-large btn-full"
              onClick={startWebcam}
              disabled={profileExists === false}
            >
              Start Webcam Scan →
            </button>
          </div>
        )}

        {/* Webcam live view */}
        {step === 'webcam' && (
          <div className="webcam-section">
            <div className="webcam-container">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="webcam-feed"
              />
              <div className="webcam-overlay">
                <div className="webcam-status">
                  <span className="status-dot"/>
                  Scanning live
                </div>
              </div>
            </div>

            {/* Hidden canvas for frame capture */}
            <canvas ref={canvasRef} style={{ display: 'none' }}/>

            <div className="webcam-stats">
              <div className="webcam-stat">
                <span className="stat-num">{checkCount}</span>
                <span className="stat-label">Frames checked</span>
              </div>
              <div className="stat-divider"/>
              <div className="webcam-stat">
                <span className="stat-num">{sightings.length}</span>
                <span className="stat-label">Sightings found</span>
              </div>
              <div className="stat-divider"/>
              <div className="webcam-stat">
                <span className="stat-num">{streak}</span>
                <span className="stat-label">Current streak</span>
              </div>
            </div>

            {sightings.length > 0 && (
              <div className="sightings-grid" style={{ marginTop: '1.5rem' }}>
                {sightings.map((url, i) => (
                  <div key={i} className="sighting-card">
                    <img src={url} alt={`Sighting ${i + 1}`}/>
                    <div className="sighting-label">Sighting {sightings.length - i}</div>
                  </div>
                ))}
              </div>
            )}

            <button
              className="btn-secondary btn-large btn-full"
              onClick={stopWebcam}
              style={{ marginTop: '1.5rem' }}
            >
              ⏹️ Stop Scanning
            </button>
          </div>
        )}

        {/* Video scanning */}
        {step === 'scanning' && (
          <div className="processing-box">
            <div className="spinner"/>
            <h2>Scanning your footage...</h2>
            <p className="processing-sub">
              This may take a few minutes depending on video length.
              The AI is checking every few seconds of footage for your pet.
            </p>
          </div>
        )}

        {/* Done */}
        {step === 'done' && (
          <div>
            <div className="done-box">
              <div className="done-icon">
                {sightings.length > 0 ? '🎉' : '😔'}
              </div>
              <h2>
                {sightings.length > 0
                  ? `Your pet was spotted ${sightings.length} time${sightings.length > 1 ? 's' : ''}!`
                  : 'No sightings found'
                }
              </h2>
              <p className="done-sub">
                {sightings.length > 0
                  ? 'Here are the frames where your pet was detected.'
                  : 'Your pet was not detected in this footage. Try different footage or adjust the threshold.'
                }
              </p>
            </div>

            {sightings.length > 0 && (
              <div className="sightings-grid">
                {sightings.map((url, i) => (
                  <div key={i} className="sighting-card">
                    <img src={`http://localhost:8000${url}`} alt={`Sighting ${i + 1}`}/>
                    <div className="sighting-label">Sighting {i + 1}</div>
                  </div>
                ))}
              </div>
            )}

            <div className="done-buttons" style={{ marginTop: '2rem' }}>
              <button
                className="btn-primary"
                onClick={() => { setStep('upload'); setFile(null); setSightings([]) }}
              >
                Scan another video
              </button>
              <Link href="/dashboard" className="btn-secondary">
                View dashboard
              </Link>
            </div>
          </div>
        )}
      </div>
    </main>
  )
}