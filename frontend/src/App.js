import React, { useRef, useEffect, useState, useCallback } from 'react';
import './App.css';

const WS_URL = 'ws://127.0.0.1:8000/ws';



function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const intervalRef = useRef(null);
  const lastSpaceTime = useRef(0);
  const sentenceRef = useRef('');

  const [isConnected, setIsConnected] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [letter, setLetter] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [top3, setTop3] = useState([]);
  const [word, setWord] = useState('');
  const [sentence, setSentence] = useState('');
  const [handDetected, setHandDetected] = useState(false);
  const [stableFrames, setStableFrames] = useState(0);
  const [requiredFrames, setRequiredFrames] = useState(8);
  const [history, setHistory] = useState([]);
  const [spaceMsg, setSpaceMsg] = useState('');

  const connectWS = useCallback(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'prediction') {
        setHandDetected(data.hand_detected);
        setStableFrames(data.stable_frames || 0);
        setRequiredFrames(data.required_frames || 8);
        if (data.hand_detected) {
          setLetter(data.letter);
          setConfidence(data.confidence);
          setTop3(data.top3 || []);
        }
        setWord(data.word || '');
        setSentence(data.sentence || '');
        sentenceRef.current = data.sentence || '';
        if (data.letter && data.confidence > 0.85) {
          setHistory(prev => [{
            letter: data.letter,
            confidence: data.confidence,
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
          }, ...prev].slice(0, 15));
        }
      }
      if (data.type === 'word_done') {
        setWord('');
        setSentence(data.sentence || '');
        sentenceRef.current = data.sentence || '';
      }
    };
    ws.onclose = () => setIsConnected(false);
    ws.onerror = () => setIsConnected(false);
  }, []);

  const sendFrame = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    if (!videoRef.current || !canvasRef.current) return;
    if (!videoRef.current.videoWidth) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = 640;
    canvas.height = 480;
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(videoRef.current, -640, 0, 640, 480);
    ctx.restore();
    const base64 = canvas.toDataURL('image/jpeg', 0.7);
    wsRef.current.send(JSON.stringify({ type: 'frame', data: base64 }));
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      connectWS();
      setIsRunning(true);
      intervalRef.current = setInterval(sendFrame, 100);
    } catch {
      alert('Camera access denied! Please allow camera permissions.');
    }
  };

  const stopCamera = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (wsRef.current) wsRef.current.close();
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      videoRef.current.srcObject = null;
    }
    setIsRunning(false);
    setIsConnected(false);
    setHandDetected(false);
    setLetter(null);
    setTop3([]);
    setStableFrames(0);
  }, []);

  const clearAll = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'clear' }));
    }
    setWord('');
    setSentence('');
    sentenceRef.current = '';
    setHistory([]);
    setLetter(null);
    setTop3([]);
  };

  const speak = (text) => {
    if (!text) return;
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 0.9;
      u.pitch = 1;
      u.volume = 1;
      window.speechSynthesis.speak(u);
    }
  };

  useEffect(() => {
    const handleKey = (e) => {
      if (e.code !== 'Space') return;
      e.preventDefault();
      e.stopPropagation();
      if (!isRunning) return;
      const now = Date.now();
      const gap = now - lastSpaceTime.current;
      if (gap < 600) {
        lastSpaceTime.current = 0;
        const cur = sentenceRef.current;
        if (cur) {
          speak(cur);
          setSpaceMsg('🔊 Speaking...');
          setTimeout(() => setSpaceMsg(''), 2000);
        }
      } else {
        lastSpaceTime.current = now;
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'clear_word' }));
          setSpaceMsg('✅ Word added!');
          setTimeout(() => setSpaceMsg(''), 1500);
        }
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isRunning]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const stabilityPct = Math.min(100, Math.round((stableFrames / requiredFrames) * 100));

  return (
    <div className="app">

      {/* ── HEADER ── */}
      <header className="header">
       <div className="logo">
  <svg width="36" height="36" viewBox="0 0 44 44" style={{flexShrink:0}}>
    <rect x="0" y="0" width="44" height="44" rx="11" fill="#2d2416"/>
    <rect x="9" y="12" width="6" height="20" rx="3" fill="white"/>
    <rect x="18" y="8" width="6" height="24" rx="3" fill="white"/>
    <rect x="27" y="12" width="6" height="20" rx="3" fill="white"/>
    <rect x="6" y="24" width="10" height="8" rx="4" fill="#7c5cbf"/>
    <ellipse cx="22" cy="38" rx="14" ry="5" fill="none" stroke="#7c5cbf" strokeWidth="1.5"/>
    <ellipse cx="22" cy="38" rx="4" ry="4" fill="#7c5cbf"/>
  </svg>
  <span className="logo-text">Sign<span className="accent">Lens</span></span>
</div>
        <div className="header-right">
          {spaceMsg && <div className="space-msg">{spaceMsg}</div>}
          <div className="status-pill">
            <div className={`status-dot ${isConnected ? 'active' : ''}`} />
            {isConnected ? 'Live' : 'Offline'}
          </div>
        </div>
      </header>

      {/* ── SCROLLABLE PAGE ── */}
      <div className="page">

        {/* ── TRANSLATOR SECTION ── */}
        <section className="translator-section">
          <div className="translator-grid">

            {/* CAMERA */}
            <div className="cam-col">
              <div className="cam-wrap">
                <video ref={videoRef} className="video" muted playsInline />
                <canvas ref={canvasRef} className="canvas-hidden" />
                {!isRunning && (
                  <div className="cam-overlay">
                    <div className="cam-overlay-icon">📷</div>
                    <p>Click Start Camera to begin</p>
                  </div>
                )}
                {isRunning && (
                  <div className={`hand-badge ${handDetected ? 'detected' : ''}`}>
                    <div className="hand-badge-dot" />
                    {handDetected ? 'Hand Detected' : 'Looking for hand...'}
                  </div>
                )}
                {isRunning && letter && handDetected && (
                  <div className="letter-overlay active">
                    <div className="lo-letter">{letter}</div>
                    <div className="lo-conf">{Math.round(confidence * 100)}%</div>
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="controls-row">
                {!isRunning ? (
                  <button type="button" className="btn btn-primary" onClick={startCamera}>▶ Start Camera</button>
                ) : (
                  <button type="button" className="btn btn-stop" onClick={stopCamera}>■ Stop</button>
                )}
                <button type="button" className="btn btn-ghost" onClick={clearAll}>🗑 Clear</button>
              </div>

              {/* Space hint */}
              <div className="space-hint">
                <span className="kbd">Space</span> = finish word &nbsp;|&nbsp;
                <span className="kbd">Space Space</span> = speak sentence
              </div>
            </div>

            {/* OUTPUT */}
            <div className="output-col">

              {/* Stability + Top3 */}
              <div className="stats-row">
                <div className="stat-card">
                  <div className="stat-label">Stability</div>
                  <div className="stat-val">{stabilityPct}%</div>
                  <div className="stat-bar-wrap">
                    <div className="stat-bar" style={{ width: `${stabilityPct}%` }} />
                  </div>
                </div>
                <div className="stat-card flex1">
                  <div className="stat-label">Top Predictions</div>
                  {top3.length > 0 ? top3.map((p, i) => (
                    <div key={i} className="pred-row">
                      <span className="pred-letter">{p.letter}</span>
                      <div className="pred-bar-wrap">
                        <div className="pred-bar" style={{ width: `${Math.round(p.confidence * 100)}%` }} />
                      </div>
                      <span className="pred-pct">{Math.round(p.confidence * 100)}%</span>
                    </div>
                  )) : <div className="placeholder-text">Show your hand...</div>}
                </div>
              </div>

              {/* Word */}
              <div className="output-card">
                <div className="out-label">Current Word</div>
                <div className="word-display">
                  {word || <span className="placeholder-text">Signing...</span>}
                </div>
              </div>

              {/* Sentence */}
              <div className="output-card">
                <div className="out-label">Full Sentence</div>
                <div className="sentence-display">
                  {sentence || <span className="placeholder-text">Your sentence appears here...</span>}
                </div>
                {sentence && (
                  <button type="button" className="speak-btn" onClick={() => speak(sentence)}>
                    🔊 Speak Sentence
                  </button>
                )}
              </div>

              {/* History */}
              <div className="output-card">
                <div className="out-label">Detection History</div>
                <div className="history-list">
                  {history.length === 0
                    ? <div className="placeholder-text">No detections yet...</div>
                    : history.map((h, i) => (
                      <div key={i} className="history-item">
                        <span className="hi-letter">{h.letter}</span>
                        <span className="hi-conf">{Math.round(h.confidence * 100)}%</span>
                        <span className="hi-time">{h.time}</span>
                      </div>
                    ))
                  }
                </div>
              </div>

            </div>
          </div>
        </section>

        {/* ── DIVIDER ── */}
        <div className="section-divider">
          <div className="divider-line" />
          <div className="divider-text">ASL Alphabet Reference Guide</div>
          <div className="divider-line" />
        </div>

        {/* ── HOW TO USE ── */}
        <section className="guide-section">
          <div className="guide-title">How to use SignLens</div>
          <div className="guide-subtitle">Follow these steps to translate your signs into text</div>
          <div className="steps-grid">
            {[
              { num: '1', icon: '📷', title: 'Start Camera', desc: 'Click the Start Camera button and allow camera permissions when prompted.' },
              { num: '2', icon: '✋', title: 'Show your hand', desc: 'Hold your hand clearly in front of the camera with good lighting.' },
              { num: '3', icon: '⏱', title: 'Hold still', desc: 'Keep your sign steady until the Stability bar reaches 100%.' },
              { num: '4', icon: '🔤', title: 'Letters build words', desc: 'Each confirmed letter gets added to your current word automatically.' },
              { num: '5', icon: '⌨️', title: 'Press Space', desc: 'Press Spacebar to finish a word and move it to the sentence.' },
              { num: '6', icon: '🔊', title: 'Double Space to speak', desc: 'Press Spacebar twice quickly to hear your full sentence spoken aloud.' },
            ].map((s) => (
              <div key={s.num} className="step-card">
                <div className="step-num">{s.num}</div>
                <div className="step-icon">{s.icon}</div>
                <div className="step-title">{s.title}</div>
                <div className="step-desc">{s.desc}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── ASL ALPHABET ── */}
<section className="alphabet-section">
  <div className="guide-title">ASL Alphabet Reference</div>
  <div className="guide-subtitle">Use this chart to learn the correct hand position for each letter</div>
  <div className="asl-image-wrap">
    <img
      src="/asl-alphabet.png"
      alt="ASL Alphabet Reference Chart"
      className="asl-image"
    />
    <div className="asl-image-credit">Source: wplipart.com — public domain images</div>
  </div>
</section>

        {/* ── TIPS ── */}
        <section className="tips-section">
          <div className="guide-title">Tips for best accuracy</div>
          <div className="guide-subtitle">Follow these tips to get the most accurate results</div>
          <div className="tips-grid">
            {[
              { icon: '💡', title: 'Good lighting', desc: 'Make sure your hand is well lit. Natural light or a lamp in front of you works best. Avoid dark rooms.' },
              { icon: '🎯', title: 'Center your hand', desc: 'Keep your hand in the middle of the camera frame. Avoid the edges where detection is weaker.' },
              { icon: '✋', title: 'Hold perfectly still', desc: 'Wait for the stability bar to reach 100% before moving to the next letter. Fast movements cause errors.' },
              { icon: '🔄', title: 'Change between letters', desc: 'After each letter, briefly move your hand away or close your fist before making the next sign.' },
              { icon: '📏', title: 'Right distance', desc: 'Keep your hand about 30–50cm from the camera. Too close or too far reduces accuracy.' },
              { icon: '🌟', title: 'Plain background', desc: 'A plain light-colored wall behind your hand helps the model detect landmarks more accurately.' },
            ].map((t, i) => (
              <div key={i} className="tip-card">
                <div className="tip-icon">{t.icon}</div>
                <div className="tip-title">{t.title}</div>
                <div className="tip-desc">{t.desc}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── FOOTER ── */}
        <footer className="footer">
          <div className="footer-logo">🤟 SignLens</div>
          <div className="footer-text">Built with MediaPipe · TensorFlow · FastAPI · React</div>
          <div className="footer-text">96.47% accuracy · 66,272 training samples · 26 ASL letters</div>
        </footer>

      </div>
    </div>
  );
}

export default App;