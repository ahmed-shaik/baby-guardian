import { useState, useEffect, useRef, useCallback } from 'react';

const RECONNECT_DELAY = 2000;

function getWebSocketUrl() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const fallbackPort = '8765';
  const port = window.location.port && window.location.port !== '5173'
    ? window.location.port
    : fallbackPort;
  return `${protocol}//${window.location.hostname}:${port}/ws/stream`;
}

export default function usePipelineSocket() {
  const [connected, setConnected] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [frameUrl, setFrameUrl] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const prevBlobUrl = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(getWebSocketUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      console.log('[WS] Connected to pipeline');
    };

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        // JSON analysis data
        try {
          const data = JSON.parse(event.data);
          setAnalysis(data);
        } catch (e) {
          console.warn('[WS] Bad JSON:', e);
        }
      } else if (event.data instanceof Blob) {
        // JPEG frame
        if (prevBlobUrl.current) {
          URL.revokeObjectURL(prevBlobUrl.current);
        }
        const url = URL.createObjectURL(event.data);
        prevBlobUrl.current = url;
        setFrameUrl(url);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      console.log('[WS] Disconnected, reconnecting...');
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
      if (prevBlobUrl.current) URL.revokeObjectURL(prevBlobUrl.current);
    };
  }, [connect]);

  return { connected, analysis, frameUrl };
}
