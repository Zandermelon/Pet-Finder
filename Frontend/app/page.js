import Link from 'next/link'

export default function Home() {
  return (
    <main className="main">

      {/* Nav */}
      <nav className="nav">
        <div className="nav-logo">🐾 PetFinder</div>
        <div className="nav-links">
          <Link href="/register">Register Pet</Link>
          <Link href="/scan" className="nav-cta">Start Scanning</Link>
        </div>
      </nav>

      {/* Hero */}
      <section className="hero">
        <div className="hero-badge">AI-Powered Pet Detection</div>
        <h1 className="hero-title">
          Find your lost pet<br />
          <span className="hero-accent">before it's too late</span>
        </h1>
        <p className="hero-sub">
          Upload photos of your pet. Connect any security camera.
          Our AI watches 24/7 and alerts you the moment your pet is spotted.
        </p>
        <div className="hero-buttons">
          <Link href="/register" className="btn-primary">Register Your Pet</Link>
          <Link href="/scan" className="btn-secondary">Scan Footage</Link>
        </div>
        <div className="hero-stats">
          <div className="stat">
            <span className="stat-num">24/7</span>
            <span className="stat-label">Continuous monitoring</span>
          </div>
          <div className="stat-divider"/>
          <div className="stat">
            <span className="stat-num">AI</span>
            <span className="stat-label">Individual recognition</span>
          </div>
          <div className="stat-divider"/>
          <div className="stat">
            <span className="stat-num">Fast</span>
            <span className="stat-label">Instant alerts</span>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="how">
        <h2 className="section-title">How it works</h2>
        <div className="steps">
          <div className="step">
            <div className="step-num">01</div>
            <div className="step-icon">📸</div>
            <h3>Upload photos</h3>
            <p>Upload 20+ photos of your pet from different angles and lighting conditions</p>
          </div>
          <div className="step-arrow">→</div>
          <div className="step">
            <div className="step-num">02</div>
            <div className="step-icon">🧠</div>
            <h3>AI builds profile</h3>
            <p>Our AI creates a unique fingerprint for your specific pet — not just their breed</p>
          </div>
          <div className="step-arrow">→</div>
          <div className="step">
            <div className="step-num">03</div>
            <div className="step-icon">📹</div>
            <h3>Scan footage</h3>
            <p>Upload video or connect a live camera feed — the AI watches every frame</p>
          </div>
          <div className="step-arrow">→</div>
          <div className="step">
            <div className="step-num">04</div>
            <div className="step-icon">🚨</div>
            <h3>Get alerted</h3>
            <p>The moment your pet is spotted you get an alert with the exact timestamp and screenshot</p>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="features">
        <h2 className="section-title">Why PetFinder?</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🎯</div>
            <h3>Individual recognition</h3>
            <p>Unlike other apps that just detect "a cat", we recognize YOUR specific cat — even among dozens of similar looking ones</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📹</div>
            <h3>Any camera works</h3>
            <p>Works with any security camera footage, webcam, or uploaded video. No special hardware needed</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3>Real-time scanning</h3>
            <p>Checks every few seconds so you know the moment your pet appears — not hours later</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🔒</div>
            <h3>Your data stays yours</h3>
            <p>Photos and footage are processed locally. We never store or share your pet's images</p>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="cta-section">
        <h2>Don't wait until your pet is missing</h2>
        <p>Set up takes less than 5 minutes. Register your pet today.</p>
        <Link href="/register" className="btn-primary btn-large">
          Get Started — It's Free
        </Link>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-logo">🐾 PetFinder</div>
        <p>Built with AI to bring pets home</p>
      </footer>

    </main>
  )
}