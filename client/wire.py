"""Anvil's transport: REST over `http.client`, the event stream over a raw socket.

The contract is `docs/vendor/anvil-protocol.md`; this module implements it and
cites it by section rather than restating it. Everything here is standard
library, so the Anvil participant adds no dependency to `requirements.txt` —
Anvil's own repo drives its wire from a raw socket in
`tests/tools/pong_ordering_probe.py`, so this is the house pattern.

Two distinctions are load-bearing and both are easy to lose in a helper.

**A verdict is not a fault.** Vendored §2 `POST /api/order`: *every* engine
verdict — accept or reject — comes back `200` with `{accepted, reason, id}`,
because the POST was well formed and reached the engine. Only `503` (engine
busy), `504` (engine timeout), `400` (malformed HTTP) and `403` (blocked origin)
mean the engine never answered. So :class:`OrderResult` carries a business
verdict and :class:`TransportFault` is raised for the rest. A client that
collapsed the two could not tell "the engine said no" from "the engine never
heard me", and would report the second as the first.

**`seq` is not an ordering.** Vendored §1 and §4: one global engine-thread stamp,
so a single-ticker socket sees a sparse subsequence that can step *backwards*,
and ring-overflow loss is not signalled on the wire at all. This module
therefore exposes `seq` as an opaque token — a reconnect watermark and a dedupe
key — and offers no "next expected" anything. Recovery is transport-driven: the
socket closes, you reconnect, and the fresh snapshot is the new baseline.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import socket
import ssl
import struct
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

#: The wire version this client was written against, from the vendored snapshot's
#: own header. `tests/test_vendored_protocol.py` asserts the two agree, and
#: :meth:`Venue.require_wire_version` refuses to proceed against anything else —
#: a bump is a breaking change by Anvil's definition, so parsing hopefully would
#: be the wrong response to one.
WIRE_VERSION = 1

#: The session cookie Anvil mints on first `POST /api/order`. Possession is the
#: ownership principal: there is no login and no server-side session store, so
#: replaying this is the only thing that lets the client cancel its own orders
#: (vendored §2, *Session & ownership*).
SESSION_COOKIE = "anvil_session"

#: What "nothing to read yet" looks like, across both transports. A plaintext
#: socket at a zero timeout raises `BlockingIOError` and at a positive one
#: `socket.timeout`; a **TLS** socket raises `ssl.SSLWantReadError`, which is an
#: `OSError` but is *not* a `BlockingIOError` and carries no `errno`. The
#: steady-state drain polls at a zero timeout, so on `wss://` that difference is
#: not an edge case — it is every single poll, and without this the first one
#: would propagate as a fault and end the run.
_WOULD_BLOCK: tuple[type[BaseException], ...] = (
    TimeoutError,
    socket.timeout,
    BlockingIOError,
    ssl.SSLWantReadError,
    ssl.SSLWantWriteError,
)

#: RFC 6455's fixed GUID, and the opcodes this reader handles.
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_OP_CONTINUATION, _OP_TEXT, _OP_BINARY = 0x0, 0x1, 0x2
_OP_CLOSE, _OP_PING, _OP_PONG = 0x8, 0x9, 0xA


class TransportFault(RuntimeError):
    """The engine did not answer — `503`, `504`, `400`, `403`, or no response.

    Deliberately *not* an order rejection. See the module docstring: the
    distinction is the whole reason `POST /api/order` returns `200` on a reject.
    """


class WireVersionMismatch(RuntimeError):
    """The server speaks a wire version this client was not written against."""


class StreamClosed(RuntimeError):
    """The WebSocket closed. Recovery is a reconnect, never a `seq` inspection."""


@dataclass(frozen=True)
class OrderResult:
    """One engine verdict, exactly as the body carried it.

    `accepted` is the business outcome and `id` is the server-minted order id —
    present even on a reject, because the allocator only advances (vendored §2).
    `id` is what makes a fill attributable: the `trade` frame's `takerId` is this
    string when the fill is ours.
    """

    accepted: bool
    id: str
    reason: str = ""
    status: int = 200
    line: str = ""

    def __str__(self) -> str:
        verdict = "accepted" if self.accepted else f"rejected ({self.reason})"
        return f"{self.line!r} -> {self.id} {verdict}"


def new_line(ticker: int, side: str, qty: int, price_ticks: int) -> str:
    """A New-order engine CSV line: six fields, **the id field empty**.

    Vendored §2: the server owns id assignment and splices its minted id into
    field 3 before `parse_line`. Six fields with the third blank — five fields is
    a wrong-column rejection, deliberately, so a client that drops the field
    earns an error instead of being silently repaired.
    """
    if side not in ("B", "S"):
        raise ValueError(f"side is 'B' or 'S', got {side!r}")
    if qty <= 0:
        raise ValueError(f"quantity must be positive, got {qty}")
    return f"{int(ticker)},N,,{side},{int(qty)},{format_price(price_ticks)}"


def cancel_line(ticker: int, order_id: str) -> str:
    """A Cancel line carrying the **server-assigned** id.

    Side, quantity and price are non-authoritative echoes on a Cancel; the id and
    the calling session are what the engine acts on. Accepted only for an order
    this session owns — and the reject reason is uniform for "not yours" and
    "unknown id", so a failed cancel says nothing about which ids are live.
    """
    return f"{int(ticker)},C,{order_id},B,1,1.0"


def format_price(ticks: int) -> str:
    """Ticks to the wire's shortest-decimal string, matching `append_price`.

    Four decimal places with trailing zeros trimmed, which is how the engine
    serialises a price everywhere — so a price this client sends round-trips
    byte-identically to the one that comes back on a `trade` frame, and equality
    on the string is a legitimate test.
    """
    if ticks <= 0:
        raise ValueError(f"a price must be positive, got {ticks} ticks")
    whole, fraction = divmod(int(ticks), 10_000)
    if fraction == 0:
        return str(whole)
    return f"{whole}.{fraction:04d}".rstrip("0")


def parse_price(text: str) -> int:
    """The wire's decimal string back to integer ticks, exactly.

    String arithmetic rather than `float(text) * 10_000`: the wire carries at
    most four decimals, and rounding a decimal through binary is precisely the
    kind of one-tick error that would make a predicted fill and a realised fill
    disagree for no reason anybody could find.
    """
    body = text.strip()
    if not body:
        raise ValueError("empty price")
    whole, _, fraction = body.partition(".")
    if len(fraction) > 4:
        raise ValueError(f"price {text!r} carries more than four decimals")
    return int(whole or "0") * 10_000 + int((fraction or "").ljust(4, "0"))


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------


@dataclass
class Venue:
    """The REST half of one Anvil server, with the session it was given."""

    host: str = "127.0.0.1"
    port: int = 18080
    timeout: float = 10.0
    scheme: str = "http"
    session: str | None = None
    #: Every order this client has had minted, newest last. The attribution set.
    order_ids: list[str] = field(default_factory=list)

    @property
    def origin(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    def _connection(self) -> http.client.HTTPConnection:
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout)
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def _request(self, method: str, path: str, body: str | None = None) -> tuple:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "text/plain"
        if self.session:
            headers["Cookie"] = f"{SESSION_COOKIE}={self.session}"
        connection = self._connection()
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read().decode("utf-8")
            status, cookies = response.status, response.getheader("Set-Cookie")
        except (OSError, http.client.HTTPException) as error:
            raise TransportFault(f"{method} {path}: {error!r}") from error
        finally:
            connection.close()
        self._absorb_cookie(cookies)
        return status, payload

    def _absorb_cookie(self, header: str | None) -> None:
        """Store `anvil_session` the first time the server mints one."""
        if not header:
            return
        for part in header.split(";"):
            name, _, value = part.strip().partition("=")
            if name == SESSION_COOKIE and value:
                self.session = value
                return

    def _json(self, method: str, path: str, body: str | None = None) -> dict:
        status, payload = self._request(method, path, body)
        if status != 200:
            raise TransportFault(f"{method} {path} returned {status}: {payload!r}")
        try:
            return json.loads(payload)
        except json.JSONDecodeError as error:
            raise TransportFault(f"{method} {path} returned non-JSON: {payload!r}") from error

    def health(self) -> dict:
        return self._json("GET", "/api/health")

    def require_wire_version(self, expected: int = WIRE_VERSION) -> dict:
        """Refuse to run against a server speaking a different wire version.

        The first thing the client does, before a socket is opened or an order is
        sent. Vendored header: a breaking change bumps the wire version precisely
        so a client can detect the mismatch on connect — parsing hopefully and
        finding out later is what the field exists to prevent.
        """
        document = self.health()
        reported = document.get("wireVersion")
        if reported != expected:
            raise WireVersionMismatch(
                f"{self.origin} reports wire version {reported!r}; this client "
                f"was written against {expected} "
                "(docs/vendor/anvil-protocol.md). Re-vendor the protocol and "
                "re-read it before changing this number."
            )
        return document

    def book(self, ticker: int, depth: int = 0) -> dict:
        """The snapshot-shaped one-shot. Same parser as a `snapshot` frame (§2)."""
        return self._json("GET", f"/api/book?ticker={int(ticker)}&depth={int(depth)}")

    def summary(self) -> dict:
        """The cross-ticker roster.

        Used for the resting totals only. **Never for a price**: `last` is the
        last *traded* price, not a mid — see :func:`client.book.arrival_mid`.
        """
        return self._json("GET", "/api/summary")

    def send(self, line: str) -> OrderResult:
        """One engine CSV line. A reject is a verdict; a non-`200` is a fault."""
        status, payload = self._request("POST", "/api/order", body=line)
        if status != 200:
            raise TransportFault(
                f"POST /api/order returned {status} for {line!r}: {payload!r}. "
                "That is a non-verdict — the engine did not answer — not a "
                "rejection (vendored §2)."
            )
        try:
            document = json.loads(payload)
        except json.JSONDecodeError as error:
            raise TransportFault(f"order verdict was not JSON: {payload!r}") from error
        result = OrderResult(
            accepted=bool(document.get("accepted")),
            id=str(document.get("id", "")),
            reason=str(document.get("reason", "")),
            status=status,
            line=line,
        )
        if result.id:
            self.order_ids.append(result.id)
        return result


# ---------------------------------------------------------------------------
# The event stream
# ---------------------------------------------------------------------------


class Stream:
    """One WebSocket to one ticker, read as a sequence of decoded JSON frames.

    A hand-rolled RFC 6455 client rather than a dependency, and a *blocking*
    reader with an explicit deadline rather than an event loop: this client's
    shape is one decision per bin with a wall-clock budget, and asyncio would add
    a scheduler to a program whose concurrency is "drain, then act".

    Control frames are handled where they arrive. A server ping is answered
    (Anvil never sends one — vendored §3, *Keepalive* — but a reader that would
    choke on one is a reader with a latent bug), a pong is surfaced so
    :meth:`ping` can time it, and a close ends the stream with
    :class:`StreamClosed` rather than an empty read somewhere downstream.
    """

    def __init__(
        self,
        host: str,
        port: int,
        ticker: int,
        *,
        depth: int = 0,
        timeout: float = 10.0,
        tls: bool = False,
    ) -> None:
        self.host, self.port, self.ticker = host, int(port), int(ticker)
        self.depth, self.timeout = int(depth), float(timeout)
        #: `wss://` rather than `ws://`. The REST half got TLS free from
        #: `http.client.HTTPSConnection`; this half is a hand-rolled RFC 6455
        #: reader over a raw socket, so it has to ask for it.
        self.tls = bool(tls)
        self._socket: socket.socket | None = None
        self._buffer = b""
        #: Application frames that arrived while :meth:`ping` was waiting on its
        #: pong. Kept rather than dropped — a freshness measurement must not cost
        #: the caller the book updates it was taken to justify trusting.
        self.pending: list[dict] = []

    # -- lifecycle ----------------------------------------------------------

    def __enter__(self) -> "Stream":
        self.connect()
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    def connect(self) -> None:
        """Open the socket and complete the upgrade (vendored §3, *Handshake*).

        Under `tls` the socket is wrapped before a byte is written, with SNI —
        `server_hostname` is not optional against a name-based virtual host, and
        omitting it earns a certificate the client cannot match rather than an
        obvious failure. Certificate verification is the default context's, on
        purpose: this client sends order lines and replays a session cookie, and
        a participant that would talk to anything holding the right IP is not a
        participant, it is a leak.

        The `Host` header drops the port under TLS. `Host: example.com:443` is
        legal but a reverse proxy matching `server_name example.com` may not
        match it, and the failure is a 404 on the upgrade rather than anything
        that names the cause.
        """
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        path = f"/ws?ticker={self.ticker}"
        if self.depth:
            path += f"&depth={self.depth}"
        default_port = 443 if self.tls else 80
        authority = self.host if self.port == default_port else f"{self.host}:{self.port}"
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {authority}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        ).encode("ascii")
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        if self.tls:
            sock = ssl.create_default_context().wrap_socket(
                sock, server_hostname=self.host
            )
        sock.sendall(request)
        self._socket = sock
        self._buffer = b""

        head = self._read_until(b"\r\n\r\n")
        status = head.split(b"\r\n", 1)[0].decode("latin-1")
        if "101" not in status:
            self.close()
            raise TransportFault(f"the /ws upgrade was refused: {status}")
        import hashlib

        expected = base64.b64encode(
            hashlib.sha1((key + _WS_GUID).encode("ascii")).digest()
        ).decode("ascii")
        if expected.lower() not in head.decode("latin-1").lower():
            self.close()
            raise TransportFault("the /ws upgrade returned a bad Sec-WebSocket-Accept")

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:  # pragma: no cover - already gone
                pass
            self._socket = None

    def reconnect(self) -> None:
        """Drop everything and open a fresh socket.

        Recovery is **transport-driven** (vendored §4): a reconnect is triggered
        by the socket closing and never by an unexpected `seq`, which a client
        could not detect anyway. The buffer and any pending frames go with the
        old socket — anything half-read is half of a frame nobody will finish,
        and anything whole is superseded by the `snapshot` the server sends on
        connect, which is the new baseline by definition.
        """
        self.close()
        self._buffer = b""
        self.pending = []
        self.connect()

    # -- reading ------------------------------------------------------------

    def _read_until(self, terminator: bytes) -> bytes:
        while terminator not in self._buffer:
            chunk = self._socket.recv(65536)
            if not chunk:
                raise StreamClosed("the socket closed during the handshake")
            self._buffer += chunk
        head, _, rest = self._buffer.partition(terminator)
        self._buffer = rest
        return head + terminator

    def _take_frame(self) -> tuple[bool, int, bytes] | None:
        """One complete frame out of the buffer, or `None` if it is not all here.

        Parsing is **atomic against the buffer**: nothing is consumed until the
        whole frame is present. That is not tidiness. A reader that consumed the
        two header bytes and then hit a socket timeout waiting for the payload
        would resume mid-frame and interpret the body as a header — silently, and
        only under load, which is exactly when a book frame is 8 kB and arrives
        split across two reads. This client polls with a zero timeout in its
        steady state, so that path is the *normal* one here, not an edge case.
        """
        buffer = self._buffer
        if len(buffer) < 2:
            return None
        first, second = buffer[0], buffer[1]
        fin, opcode = bool(first & 0x80), first & 0x0F
        masked, length = bool(second & 0x80), second & 0x7F
        offset = 2
        if length == 126:
            if len(buffer) < offset + 2:
                return None
            (length,) = struct.unpack_from("!H", buffer, offset)
            offset += 2
        elif length == 127:
            if len(buffer) < offset + 8:
                return None
            (length,) = struct.unpack_from("!Q", buffer, offset)
            offset += 8
        mask = b""
        if masked:  # pragma: no cover - a server must not mask
            if len(buffer) < offset + 4:
                return None
            mask, offset = buffer[offset : offset + 4], offset + 4
        if len(buffer) < offset + length:
            return None
        payload = buffer[offset : offset + length]
        self._buffer = buffer[offset + length :]
        if mask:  # pragma: no cover - a server must not mask
            payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        return fin, opcode, payload

    def _read_frame(self) -> tuple[int, bytes]:
        """The next *application-level* frame as `(opcode, payload)`.

        Continuation frames are reassembled. Crow does not fragment, but a reader
        that assumed so would fail rarely and unreproducibly, which is the worst
        kind of assumption to hold about somebody else's server.
        """
        while True:
            frame = self._take_frame()
            if frame is None:
                chunk = self._socket.recv(65536)
                if not chunk:
                    raise StreamClosed("the socket closed mid-frame")
                self._buffer += chunk
                continue
            fin, opcode, payload = frame
            if fin:
                return opcode, payload
            body, first_opcode = payload, opcode
            while not fin:
                more = self._take_frame()
                if more is None:
                    chunk = self._socket.recv(65536)
                    if not chunk:
                        raise StreamClosed("the socket closed mid-fragment")
                    self._buffer += chunk
                    continue
                fin, _, part = more
                body += part
            return first_opcode, body

    def _send_frame(self, opcode: int, payload: bytes = b"") -> None:
        """Client -> server frames are masked; RFC 6455 requires it."""
        mask = os.urandom(4)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        header = bytes([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header += bytes([0x80 | length])
        elif length < 1 << 16:
            header += bytes([0x80 | 126]) + struct.pack("!H", length)
        else:  # pragma: no cover - this client sends nothing large
            header += bytes([0x80 | 127]) + struct.pack("!Q", length)
        self._socket.sendall(header + mask + masked)

    def read(self, timeout: float | None = None) -> dict:
        """The next application frame, decoded. Control frames are absorbed."""
        deadline = None if timeout is None else time.perf_counter() + timeout
        while True:
            if deadline is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("no frame within the deadline")
                self._socket.settimeout(remaining)
            opcode, payload = self._read_frame()
            if opcode in (_OP_TEXT, _OP_BINARY):
                return json.loads(payload.decode("utf-8"))
            if opcode == _OP_PING:  # pragma: no cover - Anvil never pings
                self._send_frame(_OP_PONG, payload)
            elif opcode == _OP_PONG:
                continue  # a stray pong; :meth:`ping` waits for its own token
            elif opcode == _OP_CLOSE:
                raise StreamClosed("the server closed the stream")

    def drain(self, timeout: float = 0.0) -> list[dict]:
        """Every frame already waiting, oldest first; never blocks past `timeout`.

        The client's steady state. Vendored §4's slow-consumer note is the reason
        this exists: the server queues rather than sheds, so a client that
        consumes slower than the publish cadence gets *every* frame, late, with
        lag growing linearly and no plateau. Draining to empty before acting is
        what keeps "priced against the book just observed" true.
        """
        frames: list[dict] = []
        deadline = time.perf_counter() + timeout
        while True:
            remaining = max(0.0, deadline - time.perf_counter())
            try:
                self._socket.settimeout(remaining if remaining > 0 else 0.0)
                frames.append(self.read(timeout=None))
            except _WOULD_BLOCK:
                return frames
            except OSError as error:  # pragma: no cover - platform variance
                if error.errno in (10035, 11):  # WSAEWOULDBLOCK / EAGAIN
                    return frames
                raise

    def ping(self, timeout: float = 5.0) -> float:
        """Round-trip a ping and return the seconds it took.

        Vendored §3, *Keepalive*: the pong is appended to the same per-connection
        send buffer as `book` and `trade`, so it cannot overtake queued frames.
        That makes this a measure of **true end-to-end stream freshness** rather
        than of TCP liveness — a socket 110 seconds behind gets its pong back 110
        seconds late — which is exactly the number a client needs before it
        claims to have priced against a current book.

        Frames that arrive while waiting are returned to the caller's world by
        being left decoded in :attr:`pending`; nothing is discarded.

        Timed with `perf_counter`, not `monotonic`. On Windows `time.monotonic`
        is `GetTickCount64`, whose resolution is ~15.6 ms — a localhost
        round-trip quantises to exactly zero, and thirteen consecutive readings
        of `0.0` are what made that visible. A freshness measurement that reads
        zero because the clock cannot see it is worse than no measurement.
        """
        token = os.urandom(8)
        started = time.perf_counter()
        self._send_frame(_OP_PING, token)
        deadline = started + timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("no pong within the deadline")
            self._socket.settimeout(remaining)
            opcode, payload = self._read_frame()
            if opcode == _OP_PONG and payload == token:
                return time.perf_counter() - started
            if opcode in (_OP_TEXT, _OP_BINARY):
                self.pending.append(json.loads(payload.decode("utf-8")))
            elif opcode == _OP_CLOSE:
                raise StreamClosed("the server closed the stream")

    def take_pending(self) -> list[dict]:
        """Frames collected while waiting on a pong, in arrival order."""
        frames, self.pending = self.pending, []
        return frames

    def __iter__(self) -> Iterator[dict]:  # pragma: no cover - convenience
        while True:
            yield self.read()
