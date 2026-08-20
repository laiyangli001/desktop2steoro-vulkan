from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CalibrationTier:
    fps: int
    target_mbps: int
    peak_mbps: int


_FINGERPRINT_KEYS = (
    "Computing Device",
    "XR Headset Model",
    "Display Mode",
    "Depth Model",
    "Depth Resolution",
    "Capture Tool",
    "Video Encoder Backend",
    "Stream Protocol",
)


def build_calibration_fingerprint(settings: dict[str, Any]) -> dict[str, str]:
    return {
        key: "" if settings.get(key) is None else str(settings.get(key))
        for key in _FINGERPRINT_KEYS
    }


def calibration_tiers(maximum_fps: int, *, hevc: bool = False) -> list[CalibrationTier]:
    maximum = max(10, int(maximum_fps))
    fps_values = [value for value in (30, 40, 48, 50, 60) if value <= maximum]
    if not fps_values:
        fps_values = [maximum]
    bits_per_pixel = 0.075 if hevc else 0.12
    tiers = []
    for fps in fps_values:
        target = max(8, round(3840 * 2160 * fps * bits_per_pixel / 1_000_000))
        target = min(80 if not hevc else 100, target)
        peak = max(target, round(target * 1.15))
        tiers.append(CalibrationTier(fps=fps, target_mbps=target, peak_mbps=peak))
    return tiers


def evaluate_calibration_window(
    tier: CalibrationTier,
    receiver_reports: list[dict[str, Any]],
    sender_report: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    decoded = [float(item.get("decoded_fps", 0.0) or 0.0) for item in receiver_reports]
    dropped = sum(max(0, int(item.get("dropped_frames", 0) or 0)) for item in receiver_reports)
    freezes = sum(max(0, int(item.get("freeze_count", 0) or 0)) for item in receiver_reports)
    packet_loss = sum(max(0, int(item.get("packets_lost", 0) or 0)) for item in receiver_reports)
    jitter = [float(item.get("jitter_buffer_ms", 0.0) or 0.0) for item in receiver_reports]
    decoded_fps = statistics.median(decoded) if decoded else 0.0
    sender_fps = float((sender_report or {}).get("submitted_fps", 0.0) or 0.0)
    decoded_frames = max(1.0, sum(decoded))
    drop_ratio = dropped / decoded_frames
    passed = bool(
        len(receiver_reports) >= 5
        and sender_fps >= tier.fps * 0.95
        and decoded_fps >= tier.fps * 0.92
        and drop_ratio <= 0.01
        and freezes == 0
        and packet_loss <= 5
        and (not jitter or statistics.median(jitter) <= 100.0)
    )
    return passed, {
        "sender_fps": round(sender_fps, 2),
        "decoded_fps": round(decoded_fps, 2),
        "drop_ratio": round(drop_ratio, 5),
        "freeze_count": freezes,
        "packets_lost": packet_loss,
        "jitter_buffer_ms": round(statistics.median(jitter), 2) if jitter else 0.0,
        "samples": len(receiver_reports),
    }


class StreamCalibrationController:
    """Coordinate sender tiers with receiver-side WebRTC statistics."""

    def __init__(
        self,
        *,
        bind_port: int,
        stream_port: int,
        stream_key: str,
        maximum_fps: int,
        state_path: str | Path,
        profile_path: str | Path,
        hevc: bool = False,
        stage_seconds: float = 10.0,
        fingerprint: dict[str, str] | None = None,
        clock=time.monotonic,
    ) -> None:
        self.bind_port = int(bind_port)
        self.stream_port = int(stream_port)
        self.stream_key = str(stream_key)
        self.state_path = Path(state_path)
        self.profile_path = Path(profile_path)
        self.stage_seconds = max(2.0, float(stage_seconds))
        self.fingerprint = dict(fingerprint or {})
        self._clock = clock
        self._tiers = calibration_tiers(maximum_fps, hevc=hevc)
        self._tier_index = 0
        self._pending_tier: CalibrationTier | None = self._tiers[0]
        self._active_tier = self._tiers[0]
        self._best_tier: CalibrationTier | None = None
        self._bitrate_retried_fps: set[int] = set()
        self._receiver_reports: list[dict[str, Any]] = []
        self._sender_report: dict[str, Any] | None = None
        self._stage_started: float | None = None
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._status = "waiting_receiver"
        self._write_state()

    @property
    def calibration_url(self) -> str:
        return f"http://127.0.0.1:{self.bind_port}/"

    def start(self) -> None:
        controller = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.rstrip("/") in {"", "/index.html"}:
                    body = controller._viewer_html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/state":
                    controller._send_json(self, controller.state())
                    return
                self.send_error(404)

            def do_POST(self):
                if self.path != "/stats":
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    controller.add_receiver_report(payload)
                    controller._send_json(self, {"ok": True})
                except Exception as exc:
                    controller._send_json(self, {"ok": False, "error": str(exc)}, 400)

            def log_message(self, _format, *_args):
                return

        self._server = ThreadingHTTPServer(("0.0.0.0", self.bind_port), Handler)
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="StreamCalibrationHTTP",
            daemon=True,
        )
        self._server_thread.start()
        print(
            f"[StreamCalibration] Headset test page: http://<PC-IP>:{self.bind_port}/",
            flush=True,
        )

    @staticmethod
    def _send_json(handler, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)
            self._server_thread = None

    def take_pending_tier(self) -> CalibrationTier | None:
        with self._lock:
            tier = self._pending_tier
            self._pending_tier = None
            return tier

    def observe_sender(self, report: dict[str, Any]) -> None:
        with self._lock:
            self._sender_report = dict(report)
            self._maybe_finish_stage_locked()

    def add_receiver_report(self, report: dict[str, Any]) -> None:
        with self._lock:
            if self._status == "complete":
                return
            if self._stage_started is None:
                self._stage_started = self._clock()
                self._status = "testing"
            self._receiver_reports.append(dict(report))
            self._receiver_reports = self._receiver_reports[-60:]
            self._maybe_finish_stage_locked()
            self._write_state_locked()

    def _maybe_finish_stage_locked(self) -> None:
        if self._stage_started is None or self._status == "complete":
            return
        elapsed = self._clock() - self._stage_started
        if elapsed < self.stage_seconds:
            return
        passed, metrics = evaluate_calibration_window(
            self._active_tier,
            self._receiver_reports,
            self._sender_report,
        )
        if passed:
            self._best_tier = self._active_tier
        elif (
            metrics["sender_fps"] >= self._active_tier.fps * 0.95
            and self._active_tier.fps not in self._bitrate_retried_fps
        ):
            self._bitrate_retried_fps.add(self._active_tier.fps)
            reduced_target = max(8, round(self._active_tier.target_mbps * 0.75))
            reduced_peak = max(reduced_target, round(reduced_target * 1.15))
            self._active_tier = CalibrationTier(
                fps=self._active_tier.fps,
                target_mbps=reduced_target,
                peak_mbps=reduced_peak,
            )
            self._pending_tier = self._active_tier
            self._receiver_reports.clear()
            self._sender_report = None
            self._stage_started = None
            self._status = "reconnecting"
            self._write_state_locked(extra={
                "last_metrics": metrics,
                "adjustment": "lower_bitrate",
            })
            return
        next_index = self._tier_index + 1
        if passed and next_index < len(self._tiers):
            self._tier_index = next_index
            self._active_tier = self._tiers[next_index]
            self._pending_tier = self._active_tier
            self._receiver_reports.clear()
            self._sender_report = None
            self._stage_started = None
            self._status = "reconnecting"
            self._write_state_locked(extra={"last_metrics": metrics})
            return
        selected = self._best_tier or self._tiers[0]
        if selected != self._active_tier:
            self._pending_tier = selected
            self._active_tier = selected
        self._status = "complete"
        profile = {
            "version": 1,
            "created_at": time.time(),
            "stream_key": self.stream_key,
            "fps": selected.fps,
            "target_mbps": selected.target_mbps,
            "peak_mbps": selected.peak_mbps,
            "stability": "stable" if self._best_tier is not None else "limited",
            "metrics": metrics,
            "fingerprint": self.fingerprint,
        }
        self._write_json(self.profile_path, profile)
        self._write_state_locked(extra={"result": profile})
        print(
            f"[StreamCalibration] Complete: {selected.fps} FPS "
            f"target={selected.target_mbps}M peak={selected.peak_mbps}M",
            flush=True,
        )

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state_locked()

    def _state_locked(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = 0.0 if self._stage_started is None else self._clock() - self._stage_started
        payload = {
            "status": self._status,
            "tier_index": self._tier_index,
            "tier_count": len(self._tiers),
            "tier": asdict(self._active_tier),
            "stage_progress": min(1.0, elapsed / self.stage_seconds),
            "receiver_connected": bool(self._receiver_reports),
            "receiver_samples": len(self._receiver_reports),
            "sender": dict(self._sender_report or {}),
        }
        if extra:
            payload.update(extra)
        return payload

    def _write_state(self) -> None:
        with self._lock:
            self._write_state_locked()

    def _write_state_locked(self, extra: dict[str, Any] | None = None) -> None:
        self._write_json(self.state_path, self._state_locked(extra))

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _viewer_html(self) -> str:
        stream_port = self.stream_port
        stream_key = json.dumps(self.stream_key)
        return f"""<!doctype html>
<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Desktop2Stereo 自动校准</title>
<style>body{{margin:0;background:#090b10;color:#eef;font:16px system-ui;text-align:center}}
main{{max-width:1100px;margin:auto;padding:20px}}video{{width:100%;background:#000;border-radius:10px}}
#state{{margin:12px;padding:10px;background:#182033;border-radius:8px}}.metrics{{font-family:monospace}}</style>
</head><body><main><h2>Desktop2Stereo 自动网络与性能校准</h2>
<div id=\"state\">正在连接测试流…</div><video id=\"video\" autoplay playsinline controls></video>
<p class=\"metrics\" id=\"metrics\"></p></main><script>
const streamKey={stream_key}; let pc=null, timer=null, previous=null, reconnectTimer=null, sessionUrl=null;
const video=document.getElementById('video'), state=document.getElementById('state');
async function waitIce(p){{if(p.iceGatheringState==='complete')return;await new Promise(resolve=>{{
 const f=()=>{{if(p.iceGatheringState==='complete'){{p.removeEventListener('icegatheringstatechange',f);resolve();}}}};
 p.addEventListener('icegatheringstatechange',f);setTimeout(resolve,3000);}})}}
function reconnect(){{clearInterval(timer);if(sessionUrl)fetch(sessionUrl,{{method:'DELETE'}}).catch(()=>{{}});sessionUrl=null;
 if(pc)pc.close();pc=null;previous=null;clearTimeout(reconnectTimer);
 reconnectTimer=setTimeout(connect,1000);}}
async function connect(){{try{{state.textContent='正在连接测试流…';pc=new RTCPeerConnection();
 pc.addTransceiver('video',{{direction:'recvonly'}});pc.addTransceiver('audio',{{direction:'recvonly'}});
 pc.ontrack=e=>{{video.srcObject=e.streams[0];}};pc.onconnectionstatechange=()=>{{
  if(pc.connectionState==='connected')state.textContent='已连接，正在进行极限传输测试，请保持页面开启';
  if(['failed','closed','disconnected'].includes(pc.connectionState))reconnect();}};
 await pc.setLocalDescription(await pc.createOffer());await waitIce(pc);
 const url=`http://${{location.hostname}}:{stream_port}/${{encodeURIComponent(streamKey)}}/whep`;
 const response=await fetch(url,{{method:'POST',headers:{{'Content-Type':'application/sdp'}},body:pc.localDescription.sdp}});
 if(!response.ok)throw new Error(`WHEP ${{response.status}}`);
 const locationHeader=response.headers.get('Location');if(locationHeader)sessionUrl=new URL(locationHeader,url).href;
 await pc.setRemoteDescription({{type:'answer',sdp:await response.text()}});timer=setInterval(report,1000);
 }}catch(err){{state.textContent='等待测试流：'+err.message;reconnect();}}}}
async function report(){{if(!pc)return;const stats=await pc.getStats();let inbound=null;
 stats.forEach(s=>{{if(s.type==='inbound-rtp'&&s.kind==='video')inbound=s;}});if(!inbound)return;
 const now={{time:Date.now()/1000,framesDecoded:inbound.framesDecoded||0,framesDropped:inbound.framesDropped||0,
  packetsLost:inbound.packetsLost||0,bytesReceived:inbound.bytesReceived||0,freezeCount:inbound.freezeCount||0,
  jitterBufferDelay:inbound.jitterBufferDelay||0,jitterBufferEmittedCount:inbound.jitterBufferEmittedCount||0}};
 if(previous){{const dt=Math.max(.1,now.time-previous.time), emitted=now.jitterBufferEmittedCount-previous.jitterBufferEmittedCount;
  const payload={{decoded_fps:(now.framesDecoded-previous.framesDecoded)/dt,dropped_frames:now.framesDropped-previous.framesDropped,
   packets_lost:now.packetsLost-previous.packetsLost,freeze_count:now.freezeCount-previous.freezeCount,
   bitrate_mbps:(now.bytesReceived-previous.bytesReceived)*8/dt/1e6,
   jitter_buffer_ms:emitted>0?(now.jitterBufferDelay-previous.jitterBufferDelay)/emitted*1000:0,
   width:inbound.frameWidth||0,height:inbound.frameHeight||0}};
  document.getElementById('metrics').textContent=`解码 ${{payload.decoded_fps.toFixed(1)}} FPS · ${{payload.bitrate_mbps.toFixed(1)}} Mbps · 丢帧 ${{payload.dropped_frames}}`;
  fetch('/stats',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}}).catch(()=>{{}});
 }}previous=now;}}connect();
</script></body></html>"""
