'use client'
import { useState } from 'react'
import Link from 'next/link'

export default function Register() {
  const [uploadMode, setUploadMode] = useState('photos') // 'photos' | 'video'
  const [files, setFiles] = useState([])
  const [videoFile, setVideoFile] = useState(null)
  const [petName, setPetName] = useState('')
  const [step, setStep] = useState('upload')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  function handleFileChange(e) {
    setFiles(Array.from(e.target.files))
    setError('')
  }

  function handleVideoChange(e) {
    setVideoFile(e.target.files[0] || null)
    setError('')
  }

  function handleDrop(e) {
    e.preventDefault()
    const dropped = Array.from(e.dataTransfer.files)
    if (uploadMode === 'video') {
      const vid = dropped.find(f => f.type.startsWith('video/'))
      if (vid) { setVideoFile(vid); setError('') }
    } else {
      setFiles(dropped.filter(f => f.type.startsWith('image/')))
      setError('')
    }
  }

  function switchMode(mode) {
    setUploadMode(mode)
    setFiles([])
    setVideoFile(null)
    setError('')
  }

  async function handleSubmit() {
    if (!petName.trim()) {
      setError("Please enter your pet's name")
      return
    }

    try {
      let sid = localStorage.getItem('session_id')

      if (uploadMode === 'video') {
        if (!videoFile) {
          setError('Please select a video file')
          return
        }

        setStep('uploading')
        setMessage('Uploading video and extracting frames...')

        const formData = new FormData()
        formData.append('file', videoFile)

        const uploadRes = await fetch('http://localhost:8000/api/upload-profile-video', {
          method: 'POST',
          body: formData
        })
        const uploadData = await uploadRes.json()
        if (!uploadRes.ok) throw new Error(uploadData.detail)

        sid = uploadData.session_id

      } else {
        if (files.length < 10) {
          setError('Please upload at least 10 photos for better accuracy')
          return
        }

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

        sid = uploadData.session_id
      }

      localStorage.setItem('session_id', sid)

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
          <p className="page-sub">Upload photos or a video of your pet so our AI can learn to recognize them</p>
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

            {/* Mode toggle */}
            <div className="mode-toggle">
              <button
                className={`mode-btn ${uploadMode === 'photos' ? 'active' : ''}`}
                onClick={() => switchMode('photos')}
              >
                📸 Photos
              </button>
              <button
                className={`mode-btn ${uploadMode === 'video' ? 'active' : ''}`}
                onClick={() => switchMode('video')}
              >
                🎥 Video
              </button>
            </div>

            {uploadMode === 'photos' ? (
              <>
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
              </>
            ) : (
              <div
                className="dropzone"
                onDrop={handleDrop}
                onDragOver={e => e.preventDefault()}
                onClick={() => document.getElementById('video-input').click()}
              >
                <div className="dropzone-icon">🎥</div>
                <p className="dropzone-title">Drop a video here or click to browse</p>
                <p className="dropzone-sub">Supports MP4, MOV, AVI, MKV — frames will be extracted automatically</p>
                {videoFile && (
                  <div className="file-count">{videoFile.name} ({(videoFile.size / 1024 / 1024).toFixed(1)} MB)</div>
                )}
                <input
                  id="video-input"
                  type="file"
                  accept="video/*"
                  onChange={handleVideoChange}
                  style={{ display: 'none' }}
                />
              </div>
            )}

            {error && <div className="error-box">{error}</div>}

            <div className="tips-box">
              {uploadMode === 'photos' ? (
                <>
                  <h3>Tips for best results</h3>
                  <ul>
                    <li>📷 Use at least 20 photos</li>
                    <li>🔄 Different angles — front, side, back</li>
                    <li>💡 Different lighting conditions</li>
                    <li>📏 Different distances — close up and far away</li>
                    <li>🚫 Avoid blurry or very dark photos</li>
                  </ul>
                </>
              ) : (
                <>
                  <h3>Tips for best results</h3>
                  <ul>
                    <li>🎥 A 30–60 second clip works great</li>
                    <li>🔄 Walk around your pet to capture different angles</li>
                    <li>💡 Film in good lighting</li>
                    <li>📏 Include close-up and wide shots</li>
                    <li>🚫 Avoid shaky or very fast movement</li>
                  </ul>
                </>
              )}
            </div>

            <button
              className="btn-primary btn-large btn-full"
              onClick={handleSubmit}
              disabled={uploadMode === 'photos' ? files.length === 0 : !videoFile}
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
              {step === 'uploading' && uploadMode === 'video' && 'Extracting 1 frame per second from your video...'}
              {step === 'uploading' && uploadMode === 'photos' && `Uploading ${files.length} photos...`}
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
                onClick={() => { setStep('upload'); setFiles([]); setVideoFile(null); setMessage(''); setPetName('') }}
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
