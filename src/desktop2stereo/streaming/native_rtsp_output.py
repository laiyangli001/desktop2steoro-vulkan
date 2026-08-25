from __future__ import annotations

import ctypes
import os
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any


def _opus_library_candidates() -> list[str]:
    packaged = Path(__file__).resolve().with_name("native_rtsp_output") / "opus.dll"
    return [
        os.environ.get("D2S_OPUS_PATH", ""),
        str(packaged),
        "opus.dll",
    ]


def _pack_rtp_header(
    *, payload_type: int, marker: bool, sequence: int, timestamp: int, ssrc: int
) -> bytes:
    if not 0 <= payload_type <= 127:
        raise ValueError(f"RTP payload type must be 0..127, got {payload_type}")
    marker_payload = (0x80 if marker else 0) | payload_type
    return struct.pack(
        ">BBHII",
        0x80,
        marker_payload,
        sequence & 0xFFFF,
        timestamp & 0xFFFFFFFF,
        ssrc & 0xFFFFFFFF,
    )


class _Opus:
    OPUS_APPLICATION_AUDIO = 2049

    def __init__(self) -> None:
        candidates = _opus_library_candidates()
        error = None
        for candidate in candidates:
            if not candidate:
                continue
            try:
                self.lib = ctypes.CDLL(candidate)
                break
            except OSError as exc:
                error = exc
        else:
            raise RuntimeError(f"native Opus library unavailable: {error}")
        self.lib.opus_encoder_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.opus_encoder_create.restype = ctypes.c_void_p
        self.lib.opus_encode.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int
        ]
        self.lib.opus_encode.restype = ctypes.c_int
        self.lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
        self.encoder = ctypes.c_void_p()
        result = ctypes.c_int()
        self.encoder = self.lib.opus_encoder_create(
            48000, 2, self.OPUS_APPLICATION_AUDIO, ctypes.byref(result)
        )
        if not self.encoder or result.value != 0:
            raise RuntimeError(f"opus_encoder_create failed: {result.value}")

    def encode(self, pcm: bytes) -> bytes:
        samples = len(pcm) // (2 * 2)
        if samples != 960:
            raise ValueError(f"Opus input must be 20 ms stereo PCM, got {samples} frames")
        source = (ctypes.c_int16 * (samples * 2)).from_buffer_copy(pcm)
        destination = (ctypes.c_ubyte * 4000)()
        size = self.lib.opus_encode(self.encoder, source, samples, destination, 4000)
        if size < 0:
            raise RuntimeError(f"opus_encode failed: {size}")
        return bytes(destination[:size])

    def close(self) -> None:
        if self.encoder:
            self.lib.opus_encoder_destroy(self.encoder)
            self.encoder = ctypes.c_void_p()


class NativeRtspAvOutput:
    """FFmpeg-free RTSP/RTP publisher for timestamped NativeNVENC packets."""

    synchronous_submit = True

    def __init__(
        self,
        encoder: Any,
        *,
        host: str,
        port: int,
        stream_key: str,
        fps: int,
        audio_sender: Any | None = None,
    ) -> None:
        self.encoder = encoder
        self.host = str(host)
        self.port = int(port)
        self.stream_key = str(stream_key).strip("/") or "live"
        self.fps = max(1, int(fps))
        self.audio_sender = audio_sender
        self._socket: socket.socket | None = None
        self._socket_write_lock = threading.Lock()
        self._session = ""
        self._cseq = 0
        self._video_seq = 0
        self._audio_seq = 0
        self._video_ssrc = 0xD2500001
        self._audio_ssrc = 0xD2500002
        self._video_timestamp = 0
        self._audio_timestamp = 0
        self._audio_stop = threading.Event()
        self._audio_thread: threading.Thread | None = None
        self._audio_socket: socket.socket | None = None
        self._opus: _Opus | None = None
        self._audio_buffer = bytearray()
        self._url = f"rtsp://{self.host}:{self.port}/{self.stream_key}"

    def start(self) -> None:
        self._socket = socket.create_connection((self.host, self.port), timeout=5.0)
        self._socket.settimeout(5.0)
        sdp = (
            "v=0\r\n"
            "o=- 0 0 IN IP4 127.0.0.1\r\n"
            "s=Desktop2Stereo NativeNVENC\r\n"
            "t=0 0\r\n"
            "a=control:*\r\n"
            "m=video 0 RTP/AVP/TCP 96\r\n"
            "a=rtpmap:96 H264/90000\r\n"
            "a=fmtp:96 packetization-mode=1\r\n"
            "a=control:trackID=0\r\n"
            "m=audio 0 RTP/AVP/TCP 111\r\n"
            "a=rtpmap:111 opus/48000/2\r\n"
            "a=fmtp:111 useinbandfec=1\r\n"
            "a=control:trackID=1\r\n"
        ).encode()
        self._request("ANNOUNCE", self._url, {"Content-Type": "application/sdp"}, sdp)
        response = self._request(
            "SETUP", self._url + "/trackID=0",
            {"Transport": "RTP/AVP/TCP;unicast;interleaved=0-1;mode=record"},
        )
        self._session = response["headers"].get("session", "").split(";", 1)[0]
        self._request(
            "SETUP", self._url + "/trackID=1",
            {
                "Transport": "RTP/AVP/TCP;unicast;interleaved=2-3;mode=record",
                "Session": self._session,
            },
        )
        self._request("RECORD", self._url, {"Session": self._session})
        if self.audio_sender is not None:
            self._opus = _Opus()
            self._audio_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._audio_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            self._audio_socket.bind(("127.0.0.1", int(self.audio_sender.port)))
            self._audio_socket.settimeout(0.2)
            self._audio_thread = threading.Thread(
                target=self._audio_loop, name="NativeNvencOpus", daemon=True
            )
            self._audio_thread.start()
        print(
            "[DirectSbsStream] NativeNVENC RTSP/RTP active: "
            f"video=H264/90000 audio={'Opus/48000' if self._opus else 'none'} "
            "ffmpeg=disabled",
            flush=True,
        )

    def _request(self, method: str, url: str, headers: dict[str, str], body: bytes = b""):
        if self._socket is None:
            raise RuntimeError("Native RTSP socket is not connected")
        self._cseq += 1
        lines = [f"{method} {url} RTSP/1.0", f"CSeq: {self._cseq}"]
        if body:
            lines.append(f"Content-Length: {len(body)}")
        lines.extend(f"{key}: {value}" for key, value in headers.items())
        self._socket.sendall(("\r\n".join(lines) + "\r\n\r\n").encode() + body)
        raw = bytearray()
        while b"\r\n\r\n" not in raw:
            chunk = self._socket.recv(4096)
            if not chunk:
                raise RuntimeError("MediaMTX closed native RTSP connection")
            raw.extend(chunk)
        header, _, remainder = bytes(raw).partition(b"\r\n\r\n")
        lines = header.decode("latin1").split("\r\n")
        status = int(lines[0].split()[1])
        response_headers = {}
        for line in lines[1:]:
            key, _, value = line.partition(":")
            response_headers[key.casefold()] = value.strip()
        length = int(response_headers.get("content-length", "0"))
        while len(remainder) < length:
            remainder += self._socket.recv(4096)
        if status >= 300:
            raise RuntimeError(f"RTSP {method} failed: {status} {lines[0]}")
        return {"headers": response_headers}

    @staticmethod
    def _annexb_nals(data: bytes):
        markers: list[tuple[int, int]] = []
        index = 0
        while index + 3 < len(data):
            if data[index:index + 4] == b"\x00\x00\x00\x01":
                markers.append((index, index + 4))
                index += 4
            elif data[index:index + 3] == b"\x00\x00\x01":
                markers.append((index, index + 3))
                index += 3
            else:
                index += 1
        if not markers:
            if data:
                yield data
            return
        for pos, (_marker, start) in enumerate(markers):
            end = markers[pos + 1][0] if pos + 1 < len(markers) else len(data)
            if end > start:
                yield data[start:end]

    @staticmethod
    def _interleaved_frame(channel: int, packet: bytes) -> bytes:
        return b"$" + bytes([channel]) + struct.pack(">H", len(packet)) + packet

    def _interleaved(self, channel: int, packet: bytes) -> None:
        self._interleaved_batch(self._interleaved_frame(channel, packet))

    def _interleaved_batch(self, frames: bytes | bytearray) -> None:
        if self._socket is None:
            raise RuntimeError("Native RTSP socket is closed")
        if not frames:
            return
        with self._socket_write_lock:
            self._socket.sendall(frames)

    def _send_h264(self, data: bytes, timestamp: int) -> None:
        batch = bytearray()
        nals = list(self._annexb_nals(data))
        for nal_index, nal in enumerate(nals):
            if not nal:
                continue
            marker = nal_index == len(nals) - 1
            if len(nal) <= 1400:
                payload = nal
                packet = _pack_rtp_header(
                    payload_type=96,
                    marker=marker,
                    sequence=self._video_seq,
                    timestamp=timestamp,
                    ssrc=self._video_ssrc,
                ) + payload
                self._video_seq = (self._video_seq + 1) & 0xFFFF
                batch.extend(self._interleaved_frame(0, packet))
                if len(batch) >= 64 * 1024:
                    self._interleaved_batch(batch)
                    batch.clear()
                continue
            first = True
            offset = 1
            while offset < len(nal):
                piece = nal[offset:offset + 1399]
                offset += len(piece)
                last = offset >= len(nal)
                fu_header = (0x80 if first else 0) | (0x40 if last else 0) | (nal[0] & 0x1F)
                payload = bytes([(nal[0] & 0xE0) | 28, fu_header]) + piece
                packet = _pack_rtp_header(
                    payload_type=96,
                    marker=marker and last,
                    sequence=self._video_seq,
                    timestamp=timestamp,
                    ssrc=self._video_ssrc,
                ) + payload
                self._video_seq = (self._video_seq + 1) & 0xFFFF
                batch.extend(self._interleaved_frame(0, packet))
                if len(batch) >= 64 * 1024:
                    self._interleaved_batch(batch)
                    batch.clear()
                first = False
        self._interleaved_batch(batch)

    def _audio_loop(self) -> None:
        assert self._audio_socket is not None and self._opus is not None
        while not self._audio_stop.is_set():
            try:
                self._audio_buffer.extend(self._audio_socket.recv(65536))
            except socket.timeout:
                continue
            frame_size = 960 * 2 * 2
            while len(self._audio_buffer) >= frame_size:
                pcm = bytes(self._audio_buffer[:frame_size])
                del self._audio_buffer[:frame_size]
                encoded = self._opus.encode(pcm)
                header = _pack_rtp_header(
                    payload_type=111,
                    marker=False,
                    sequence=self._audio_seq,
                    timestamp=self._audio_timestamp,
                    ssrc=self._audio_ssrc,
                )
                self._audio_seq = (self._audio_seq + 1) & 0xFFFF
                self._audio_timestamp += 960
                self._interleaved(2, header + encoded)

    def submit_cuda_frame(self, frame: Any) -> None:
        for packet in self.encoder.encode_timed(frame):
            if packet.data:
                timestamp = max(0, int(packet.pts)) * 90000 // self.fps
                self._send_h264(packet.data, timestamp)

    def close(self) -> None:
        self._audio_stop.set()
        if self._audio_thread is not None:
            self._audio_thread.join(timeout=1.0)
        if self._audio_socket is not None:
            self._audio_socket.close()
        if self._opus is not None:
            self._opus.close()
        if self._socket is not None:
            try:
                if self._session:
                    self._request("TEARDOWN", self._url, {"Session": self._session})
            except Exception:
                pass
            self._socket.close()
            self._socket = None
        close = getattr(self.encoder, "close", None)
        if callable(close):
            close()
