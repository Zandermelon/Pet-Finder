'use client'
import { useState } from 'react'
import Link from 'next/link'

export default function Register() {
  const [files, setFiles] = useState([])
  const [step, setStep] = useState('upload') // upload → cropping → building → done
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [photoCount, setPhotoCount] = useState(0)

  function handleFileChange(e) {
    setFiles(Array.from(e.target.files))
    setError('')
  }

  function handleDrop(e) {
    e.preventDefault()
    setFiles(Array.from(e.dataTransfer.files))
    setError('')
  }

  const [sessionId, setSessionId] = useState(null)

async function handleSubmit() {
    if (files.length < 10) {
      setError('Please upload at least 10 photos for better accuracy')
      return
    }

    try {
      // Step 1 — upload photos
      setStep('uploading')
      setMessage('Uploading photos...')

      const formData = new FormData()
      files.forEach(f => formData.append('files', f))

      const uploadRes = await fetch('http://localhost:8000/api/upload-photos', {
        method: 'POST',
        body: formData
      })
      const uploadData = await uploadRes.json()
      if (!uploadRes.ok) throw new Error(uploadData.detail)

      // Save session ID
      const sid = uploadData.session_id
      setSessionId(sid)
      localStorage.setItem('session_id', sid)  // save for other pages too
      setPhotoCount(uploadData.count)

      // Step 2 — crop
      setStep('cropping')
      setMessage('Finding your pet in each photo...')

      const cropRes = await fetch(`http://localhost:8000/api/crop-photos/${sid}`, {
        method: 'POST'
      })
      const cropData = await cropRes.json()
      if (!cropRes.ok) throw new Error(cropData.detail)

      // Step 3 — build profile
      setStep('building')
      setMessage("Building your pet's unique profile...")

      const profileRes = await fetch(`http://localhost:8000/api/build-profile/${sid}`, {
        method: 'POST'
      })
      const profileData = await profileRes.json()
      if (!profileRes.ok) throw new Error(profileData.detail)

      setStep('done')
      setMessage(profileData.message)

    } catch (err) {
      setError(err.message)
      setStep('upload')
    }
  }

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
          <p className="page-sub">Upload photos of your pet so our AI can learn to recognize them</p>
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
            <div
              className="dropzone"
              onDrop={handleDrop}
              onDragOver={e => e.preventDefault()}
              onClick={() => document.getElementById('file-input').click()}
            >
              <div className="dropzone-icon">📸</div>
              <p className="dropzone-title">Drop photos here or click to browse</p>
              <p className="dropzone-sub">Upload 20+ photos for best results — different angles, lighting, distances</p>
              {files.length > 0 && (
                <div className="file-count">{files.length} photos selected</div>
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

            {/* Photo previews */}
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

            {error && <div className="error-box">{error}</div>}

            <div className="tips-box">
              <h3>Tips for best results</h3>
              <ul>
                <li>📷 Use at least 20 photos</li>
                <li>🔄 Different angles — front, side, back</li>
                <li>💡 Different lighting conditions</li>
                <li>📏 Different distances — close up and far away</li>
                <li>🚫 Avoid blurry or very dark photos</li>
              </ul>
            </div>

            <button
              className="btn-primary btn-large btn-full"
              onClick={handleSubmit}
              disabled={files.length === 0}
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
              {step === 'uploading' && `Uploading ${files.length} photos...`}
              {step === 'cropping' && 'YOLO AI is finding your pet in each photo and cropping them out...'}
              {step === 'building' && 'DINO AI is building a unique fingerprint for your pet...'}
            </p>
          </div>
        )}

        {/* Done */}
        {step === 'done' && (
          <div className="done-box">
            <div className="done-icon">🎉</div>
            <h2>Your pet is registered!</h2>
            <p>{message}</p>
            <p className="done-sub">
              Your pet's unique profile has been saved. You can now scan video footage
              or connect a camera to start watching for them.
            </p>
            <div className="done-buttons">
              <Link href="/scan" className="btn-primary btn-large">
                Start Scanning →
              </Link>
              <button
                className="btn-secondary"
                onClick={() => { setStep('upload'); setFiles([]); setMessage('') }}
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