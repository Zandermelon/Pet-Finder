'use client'
import { useState } from 'react'
import Link from 'next/link'

export default function Register() {
  const [files, setFiles] = useState([])
  const [videoFiles, setVideoFiles] = useState([])
  const [petName, setPetName] = useState('')
  const [step, setStep] = useState('upload')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  function handleFileChange(e) {
    setFiles(Array.from(e.target.files))
    setError('')
  }

  function handleVideoChange(e) {
    setVideoFiles(prev => {
      const existing = new Set(prev.map(f => f.name))
      const added = Array.from(e.target.files).filter(f => !existing.has(f.name))
      return [...prev, ...added]
    })
    setError('')
  }

  function handlePhotoDrop(e) {
    e.preventDefault()
    const dropped = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'))
    if (dropped.length) { setFiles(dropped); setError('') }
  }

  function handleVideoDrop(e) {
    e.preventDefault()
    const vids = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('video/'))
    if (vids.length) {
      setVideoFiles(prev => {
        const existing = new Set(prev.map(f => f.name))
        return [...prev, ...vids.filter(f => !existing.has(f.name))]
      })
      setError('')
    }
  }

  async function handleSubmit() {
    if (!petName.trim()) {
      setError("Please enter your pet's name")
      return
    }
    if (files.length === 0 && videoFiles.length === 0) {
      setError('Please upload at least some photos or a video')
      return
    }
    if (files.length > 0 && files.length < 5) {
      setError('Please upload at least 5 photos, or add a video too')
      return
    }

    try {
      let sid = localStorage.getItem('session_id')

      // Upload photos first (if any)
      if (files.length > 0) {
        setStep('uploading')
        setMessage('Uploading photos...')

        const formData = new FormData()
        files.forEach(f => formData.append('files', f))

        const sidParam = sid ? `?session_id=${encodeURIComponent(sid)}` : ''
        const uploadRes = await fetch(`http://localhost:8000/api/upload-photos${sidParam}`, {
          method: 'POST',
          body: formData
        })
        const uploadData = await uploadRes.json()
        if (!uploadRes.ok) throw new Error(uploadData.detail)
        sid = uploadData.session_id
      }

      // Upload each video in sequence, reusing the same session
      for (let i = 0; i < videoFiles.length; i++) {
        setStep('uploading')
        setMessage(
          videoFiles.length > 1
            ? `Extracting frames from video ${i + 1} of ${videoFiles.length}...`
            : 'Uploading video and extracting frames...'
        )

        const formData = new FormData()
        formData.append('file', videoFiles[i])

        const sidParam = sid ? `?session_id=${encodeURIComponent(sid)}` : ''
        const uploadRes = await fetch(`http://localhost:8000/api/upload-profile-video${sidParam}`, {
          method: 'POST',
          body: formData
        })
        const uploadData = await uploadRes.json()
        if (!uploadRes.ok) throw new Error(uploadData.detail)
        sid = uploadData.session_id
      }

      localStorage.setItem('session_id', sid)

      // Crop
      setStep('cropping')
      setMessage('Finding your pet in each photo...')

      const cropRes = await fetch(`http://localhost:8000/api/crop-photos/${sid}`, { method: 'POST' })
      const cropData = await cropRes.json()
      if (!cropRes.ok) throw new Error(cropData.detail)

      // Build profile
      setStep('building')
      setMessage(`Building ${petName}'s unique profile...`)

      const profileRes = await fetch(
        `http://localhost:8000/api/build-profile/${sid}?pet_name=${encodeURIComponent(petName.trim())}`,
        { method: 'POST' }
      )
      const profileData = await profileRes.json()
      if (!profileRes.ok) throw new Error(profileData.detail)

      setStep('done')
      setMessage(profileData.message)

    } catch (err) {
      setError(err.message)
      setStep('upload')
    }
  }

  const hasContent = files.length > 0 || videoFiles.length > 0

  return (
    <main className="main">

      {/* Nav */}
      <nav className="nav">
        <Link href="/" className="nav-logo">🐾 PetFinder</Link>
        <div className="nav-links">
          <Link href="/scan" className="nav-cta">Start Scanning</Link>
        </div>
      </nav>

      <div className="page-container">
        <div className="page-header">
          <h1 className="page-title">Register your pet</h1>
          <p className="page-sub">Upload photos and/or videos — the more the better</p>
        </div>

        {/* Progress steps */}
        <div className="progress-steps">
          {['upload', 'uploading', 'cropping', 'building', 'done'].map((s, i) => (
            <div key={s} className={`progress-step ${step === s ? 'active' : ''} ${
              ['upload', 'uploading', 'cropping', 'building', 'done'].indexOf(step) > i ? 'complete' : ''
            }`}>
              <div className="progress-dot">{
                ['upload', 'uploading', 'cropping', 'building', 'done'].indexOf(step) > i ? '✓' : i + 1
              }</div>
              <span>{['Upload', 'Uploading', 'Detecting', 'Building', 'Done'][i]}</span>
            </div>
          ))}
        </div>

        {/* Upload area */}
        {step === 'upload' && (
          <div className="upload-section">

            {/* Pet name input */}
            <div className="input-group" style={{ marginBottom: '1.5rem' }}>
              <label className="input-label">Pet's name</label>
              <input
                type="text"
                className="text-input"
                placeholder="e.g. Fluffy, Max, Bella..."
                value={petName}
                onChange={e => { setPetName(e.target.value); setError('') }}
                maxLength={50}
              />
            </div>

            {/* Photos dropzone */}
            <div
              className="dropzone"
              onDrop={handlePhotoDrop}
              onDragOver={e => e.preventDefault()}
              onClick={() => document.getElementById('file-input').click()}
            >
              <div className="dropzone-icon">📸</div>
              <p className="dropzone-title">Drop photos here or click to browse</p>
              <p className="dropzone-sub">20+ photos recommended — different angles, lighting, distances</p>
              {files.length > 0 && (
                <div className="file-count">{files.length} photo{files.length !== 1 ? 's' : ''} selected</div>
              )}
              <input
                id="file-input"
                type="file"
                multiple
                accept="image/*"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
            </div>

            {files.length > 0 && (
              <div className="preview-grid">
                {files.slice(0, 8).map((f, i) => (
                  <div key={i} className="preview-item">
                    <img src={URL.createObjectURL(f)} alt={`preview ${i}`} />
                  </div>
                ))}
                {files.length > 8 && (
                  <div className="preview-more">+{files.length - 8} more</div>
                )}
              </div>
            )}

            {/* Videos dropzone */}
            <div
              className="dropzone"
              style={{ marginTop: '1rem' }}
              onDrop={handleVideoDrop}
              onDragOver={e => e.preventDefault()}
              onClick={() => document.getElementById('video-input').click()}
            >
              <div className="dropzone-icon">🎥</div>
              <p className="dropzone-title">Drop videos here or click to browse</p>
              <p className="dropzone-sub">Optional — supports MP4, MOV, AVI, MKV. Multiple videos allowed.</p>
              {videoFiles.length > 0 && (
                <div className="file-count">{videoFiles.length} video{videoFiles.length !== 1 ? 's' : ''} selected</div>
              )}
              <input
                id="video-input"
                type="file"
                accept="video/*"
                multiple
                onChange={handleVideoChange}
                style={{ display: 'none' }}
              />
            </div>

            {videoFiles.length > 0 && (
              <div style={{ marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                {videoFiles.map((f, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    <span>🎥 {f.name} ({(f.size / 1024 / 1024).toFixed(1)} MB)</span>
                    <button
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: '1rem', padding: '0 0.25rem' }}
                      onClick={() => setVideoFiles(prev => prev.filter((_, j) => j !== i))}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}

            {error && <div className="error-box">{error}</div>}

            <div className="tips-box">
              <h3>Tips for best results</h3>
              <ul>
                <li>📷 Use at least 20 photos or a 30–60 second video</li>
                <li>🔄 Capture different angles — front, side, back</li>
                <li>💡 Include different lighting conditions</li>
                <li>📏 Mix close-up and wide shots</li>
                <li>🚫 Avoid blurry or very dark images</li>
              </ul>
            </div>

            <button
              className="btn-primary btn-large btn-full"
              onClick={handleSubmit}
              disabled={!hasContent}
            >
              Register My Pet →
            </button>
          </div>
        )}

        {/* Processing states */}
        {['uploading', 'cropping', 'building'].includes(step) && (
          <div className="processing-box">
            <div className="spinner"/>
            <h2>{message}</h2>
            <p className="processing-sub">
              {step === 'uploading' && videoFiles.length > 0 && files.length > 0 && 'Uploading photos then extracting video frames...'}
              {step === 'uploading' && videoFiles.length > 0 && files.length === 0 && 'Extracting 1 frame per second from each video...'}
              {step === 'uploading' && videoFiles.length === 0 && `Uploading ${files.length} photos...`}
              {step === 'cropping' && 'YOLO AI is finding your pet in each photo and cropping them out...'}
              {step === 'building' && 'DINO AI is building a unique fingerprint for your pet...'}
            </p>
          </div>
        )}

        {/* Done */}
        {step === 'done' && (
          <div className="done-box">
            <div className="done-icon">🎉</div>
            <h2>{petName} is registered!</h2>
            <p>{message}</p>
            <p className="done-sub">
              {petName}'s unique profile has been saved. You can register more pets or start scanning.
            </p>
            <div className="done-buttons">
              <Link href="/scan" className="btn-primary btn-large">
                Start Scanning →
              </Link>
              <button
                className="btn-secondary"
                onClick={() => { setStep('upload'); setFiles([]); setVideoFiles([]); setMessage(''); setPetName('') }}
              >
                Register another pet
              </button>
            </div>
          </div>
        )}
      </div>
    </main>
  )
}
