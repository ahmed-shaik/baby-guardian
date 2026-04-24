import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Camera, Link2, Shield, Radio, LoaderCircle } from 'lucide-react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import CameraFeed from './components/CameraFeed';
import AlertBanner from './components/AlertBanner';
import StatsBar from './components/StatsBar';
import DetectedObjects from './components/DetectedObjects';
import PoseAnalysis from './components/PoseAnalysis';
import AlertHistory from './components/AlertHistory';
import Timeline from './components/Timeline';
import usePipelineSocket from './hooks/usePipelineSocket';

const DEFAULT_SOURCE = '0';
const SOURCE_STORAGE_KEY = 'baby-guardian-stream-source';

function getApiBase() {
  if (window.location.port === '5173') {
    return `${window.location.protocol}//${window.location.hostname}:8765`;
  }
  return '';
}

async function fetchJson(path, options) {
  const response = await fetch(`${getApiBase()}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return response.json();
}

function SourceSetupScreen({ initialSource, onStart }) {
  const [source, setSource] = useState(initialSource || DEFAULT_SOURCE);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setSource(initialSource || DEFAULT_SOURCE);
  }, [initialSource]);

  const examples = [
    '0 for webcam',
    'rtsp://user:pass@camera-ip:554/stream',
    'http://192.168.1.5:8080/video',
    'Cloudflare tunnel or any public live-stream URL',
  ];

  const handleSubmit = async (event) => {
    event.preventDefault();
    const trimmedSource = source.trim();

    if (!trimmedSource) {
      setError('Enter a webcam number or live stream link first.');
      return;
    }

    setSubmitting(true);
    setError('');

    try {
      const result = await fetchJson('/api/source', {
        method: 'POST',
        body: JSON.stringify({ source: trimmedSource }),
      });

      if (!result.ok) {
        throw new Error(result.error || 'Could not start the selected stream.');
      }

      localStorage.setItem(SOURCE_STORAGE_KEY, trimmedSource);
      onStart(trimmedSource);
    } catch (err) {
      setError(err.message || 'Could not start the selected stream.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[var(--color-bg)] noise-overlay overflow-hidden">
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute -top-32 left-[-10%] h-80 w-80 rounded-full bg-primary-500/10 blur-[110px]" />
        <div className="absolute bottom-[-10%] right-[-5%] h-96 w-96 rounded-full bg-safe-500/10 blur-[120px]" />
      </div>

      <main className="relative min-h-screen flex items-center justify-center px-5 py-8 sm:px-6">
        <div className="w-full max-w-6xl grid gap-5 lg:grid-cols-[1.15fr_0.85fr]">
          <motion.section
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45 }}
            className="glass-card border-gradient rounded-[32px] p-6 sm:p-8 lg:p-10"
          >
            <div className="inline-flex items-center gap-2 rounded-full border border-primary-500/15 bg-primary-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.24em] text-primary-300">
              <Radio size={12} />
              Live Source Setup
            </div>

            <div className="mt-6 max-w-2xl">
              <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-white">
                Connect your stream before opening the dashboard
              </h1>
              <p className="mt-3 text-sm sm:text-[15px] leading-7 text-slate-400">
                Paste any live source the backend can open: IP camera link, RTSP stream, Cloudflare tunnel URL,
                public live stream, or a webcam index like <span className="text-primary-300 font-semibold">0</span>.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="mt-8 space-y-4">
              <label className="block">
                <span className="mb-2.5 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
                  <Link2 size={13} className="text-primary-400" />
                  Stream Link
                </span>
                <div className="glass-subtle rounded-2xl border border-white/[0.07] px-4 py-3 focus-within:border-primary-500/40 focus-within:bg-primary-500/5 transition-colors">
                  <input
                    value={source}
                    onChange={(event) => setSource(event.target.value)}
                    placeholder="Paste webcam number or stream URL..."
                    className="w-full bg-transparent text-sm text-slate-100 outline-none placeholder:text-slate-600"
                    autoComplete="off"
                    spellCheck="false"
                  />
                </div>
              </label>

              {error && (
                <div className="rounded-2xl border border-danger-500/20 bg-danger-500/8 px-4 py-3 text-sm text-danger-300">
                  {error}
                </div>
              )}

              <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                <button
                  type="submit"
                  disabled={submitting}
                  className="inline-flex items-center justify-center gap-2 rounded-2xl bg-primary-500 px-5 py-3 text-sm font-semibold text-white shadow-[0_16px_50px_rgba(59,130,246,0.28)] transition hover:bg-primary-400 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {submitting ? <LoaderCircle size={16} className="animate-spin" /> : <Camera size={16} />}
                  {submitting ? 'Starting stream...' : 'Open Dashboard'}
                </button>
                <p className="text-xs text-slate-500">
                  The stream is started on the backend first, then the live dashboard opens.
                </p>
              </div>
            </form>

            <div className="mt-8 grid gap-3 sm:grid-cols-2">
              {examples.map((example) => (
                <div key={example} className="glass-subtle rounded-2xl border border-white/[0.05] px-4 py-3 text-sm text-slate-400">
                  {example}
                </div>
              ))}
            </div>
          </motion.section>

          <motion.aside
            initial={{ opacity: 0, y: 22 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.05 }}
            className="glass-card rounded-[32px] border border-white/[0.04] p-6 sm:p-8"
          >
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-safe-500/10 text-safe-400">
                <Shield size={22} />
              </div>
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-slate-500">Baby Guardian</p>
                <h2 className="text-lg font-semibold text-white">Startup checklist</h2>
              </div>
            </div>

            <div className="mt-6 space-y-3">
              <div className="rounded-2xl border border-white/[0.05] bg-white/[0.02] p-4">
                <p className="text-sm font-semibold text-white">Use a direct live stream URL</p>
                <p className="mt-1.5 text-sm leading-6 text-slate-400">
                  RTSP, MJPEG, IP Webcam, DroidCam, webcam index, Cloudflare tunnel, or any URL OpenCV can read live.
                </p>
              </div>
              <div className="rounded-2xl border border-white/[0.05] bg-white/[0.02] p-4">
                <p className="text-sm font-semibold text-white">Keep the existing AI flow</p>
                <p className="mt-1.5 text-sm leading-6 text-slate-400">
                  Your dashboard still uses the same backend pipeline, websocket feed, detections, and alert panels.
                </p>
              </div>
              <div className="rounded-2xl border border-white/[0.05] bg-white/[0.02] p-4">
                <p className="text-sm font-semibold text-white">If a link fails</p>
                <p className="mt-1.5 text-sm leading-6 text-slate-400">
                  The page shows the backend error instead of silently loading a broken React screen.
                </p>
              </div>
            </div>
          </motion.aside>
        </div>
      </main>
    </div>
  );
}

function DashboardView() {
  const { connected, analysis, frameUrl } = usePipelineSocket();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [alertHistory, setAlertHistory] = useState([]);
  const [timeline, setTimeline] = useState([]);
  const [activeAlert, setActiveAlert] = useState(null);
  const alertIdCounter = useRef(1);
  const prevStatus = useRef('safe');
  const startTime = useRef(Date.now());

  const fps = analysis?.fps || 0;
  const status = analysis?.status || 'safe';
  const persons = analysis?.persons || [];
  const detections = analysis?.detections || [];

  const pose = persons.length > 0 ? {
    confidence: persons[0].pose?.person_confidence || 0,
    visibleLandmarks: persons[0].pose?.keypoints?.filter((kp) => kp.confidence > 0.5).length || 0,
    totalLandmarks: 33,
    riskScore: persons[0].risk?.score || 0,
    riskLabel: persons[0].risk?.label === 'dangerous' ? 'danger' : (persons[0].risk?.label === 'uncertain' ? 'warning' : 'safe'),
    activeRules: persons[0].risk?.reasons || [],
  } : {
    confidence: 0,
    visibleLandmarks: 0,
    totalLandmarks: 33,
    riskScore: 0,
    riskLabel: 'safe',
    activeRules: connected ? ['Waiting for person detection...'] : ['Pipeline not connected'],
  };

  const mappedDetections = detections.map((d, i) => ({
    id: d.track_id || i + 1,
    className: d.class_name,
    confidence: d.confidence,
    trackId: d.track_id || i + 1,
    riskLevel: d.risk_level || 'safe',
  }));

  useEffect(() => {
    if (!analysis) return;

    if (status !== prevStatus.current) {
      const timeStr = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
      const id = alertIdCounter.current++;

      let severity = 'safe';
      let message = 'Baby is in a safe position';
      let detail = 'All risk signals cleared';

      if (status === 'danger' && persons.length > 0) {
        severity = 'danger';
        const reasons = persons[0].risk?.reasons || [];
        message = reasons[0] || 'Dangerous posture detected';
        detail = reasons.slice(1).join('; ') || 'Immediate attention needed';
      } else if (status === 'warning' && persons.length > 0) {
        severity = 'warning';
        const reasons = persons[0].risk?.reasons || [];
        message = reasons[0] || 'Potential risk detected';
        detail = reasons.slice(1).join('; ') || 'Monitor closely';
      }

      setAlertHistory((prev) => [{
        id, severity, message, detail, time: timeStr, timestamp: Date.now(),
      }, ...prev].slice(0, 20));

      const timelineType = status === 'danger' ? 'danger' : status === 'warning' ? 'warning' : 'safe';
      setTimeline((prev) => [{
        id, time: timeStr, event: message, type: timelineType,
      }, ...prev].slice(0, 15));

      if (status === 'danger' || status === 'warning') {
        setActiveAlert({ id, severity, message, detail });
      } else {
        setActiveAlert(null);
      }

      prevStatus.current = status;
    }
  }, [analysis, persons, status]);

  useEffect(() => {
    if (connected && timeline.length === 0) {
      const timeStr = new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
      setTimeline([{ id: 0, time: timeStr, event: 'Pipeline connected', type: 'info' }]);
    }
  }, [connected, timeline.length]);

  const uptimeMs = Date.now() - startTime.current;
  const uptimeMin = Math.floor(uptimeMs / 60000);
  const uptimeStr = uptimeMin >= 60
    ? `${Math.floor(uptimeMin / 60)}h ${uptimeMin % 60}m`
    : `${uptimeMin}m`;

  const stats = {
    uptime: connected ? uptimeStr : '--',
    fps,
    totalAlerts: alertHistory.length,
    dangerAlerts: alertHistory.filter((alert) => alert.severity === 'danger').length,
  };

  return (
    <div className="min-h-screen bg-[var(--color-bg)] noise-overlay">
      <Sidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(!sidebarCollapsed)} />

      <motion.div
        animate={{ marginLeft: sidebarCollapsed ? 64 : 220 }}
        transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
        className="min-h-screen flex flex-col"
      >
        <Header connected={connected} />

        <motion.main
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4 }}
          className="flex-1 max-w-[1400px] w-full mx-auto px-5 sm:px-6 pb-8 pt-5 space-y-4"
        >
          {activeAlert && (
            <AlertBanner
              key={activeAlert.id}
              alert={activeAlert}
              onDismiss={() => setActiveAlert(null)}
            />
          )}

          <StatsBar data={stats} />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2 space-y-4">
              <CameraFeed
                status={status}
                fps={fps}
                frameUrl={frameUrl}
                connected={connected}
              />
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <DetectedObjects detections={mappedDetections} />
                <PoseAnalysis pose={pose} />
              </div>
            </div>

            <div className="space-y-4">
              <AlertHistory alerts={alertHistory} />
              <Timeline events={timeline} />
            </div>
          </div>

          <div className="text-center pt-4 pb-2">
            <p className="text-[11px] text-slate-600 font-medium">
              Baby Guardian v1.0 &middot; AI-powered safety monitoring
            </p>
          </div>
        </motion.main>
      </motion.div>
    </div>
  );
}

function App() {
  const [ready, setReady] = useState(false);
  const [selectedSource, setSelectedSource] = useState(DEFAULT_SOURCE);

  useEffect(() => {
    let active = true;

    async function loadStatus() {
      try {
        const status = await fetchJson('/api/status');
        if (!active) return;

        const storedSource = localStorage.getItem(SOURCE_STORAGE_KEY);
        const currentSource = status.source != null ? String(status.source) : (storedSource || DEFAULT_SOURCE);

        setSelectedSource(currentSource);
        setReady(Boolean(status.running && status.source != null));
      } catch {
        const storedSource = localStorage.getItem(SOURCE_STORAGE_KEY) || DEFAULT_SOURCE;
        if (!active) return;
        setSelectedSource(storedSource);
        setReady(false);
      }
    }

    loadStatus();
    return () => {
      active = false;
    };
  }, []);

  if (!ready) {
    return (
      <SourceSetupScreen
        initialSource={selectedSource}
        onStart={(source) => {
          setSelectedSource(source);
          setReady(true);
        }}
      />
    );
  }

  return <DashboardView />;
}

export default App;
