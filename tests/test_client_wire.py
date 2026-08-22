"""The transport's pure parts, and the two distinctions that are easy to lose.

Nothing here opens a socket to Anvil. What is checked is the arithmetic and the
framing — the parts that are wrong silently rather than loudly:

* **prices round-trip exactly**, because a decimal put through a float is how a
  predicted fill and a realised fill end up one tick apart for no findable
  reason;
* **an order line is six fields with the third blank**, because five fields earns
  a wrong-column rejection and a client that "helpfully" dropped the empty field
  would look like it was working until it wasn't;
* **a `200` reject is a verdict and a `503` is a fault**, because collapsing them
  makes "the engine said no" indistinguishable from "the engine never heard me";
* **frame parsing is atomic against the buffer**, because a reader that consumed
  a header and then hit a timeout would resume mid-frame and read a payload as a
  header — under load, silently, and only when a book frame is 8 kB.

The WebSocket half is driven against a loopback socket rather than a mock: the
thing being tested is byte handling, and a mock that hands over whole frames
tests the opposite of the property that matters.
"""

from __future__ import annotations

import json
import socket
import struct
import threading

import pytest

from client.wire import (
    OrderResult,
    Stream,
    TransportFault,
    Venue,
    WireVersionMismatch,
    cancel_line,
    format_price,
    new_line,
    parse_price,
)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ticks", "wire"),
    [
        (100_000, "10"),
        (99_900, "9.99"),
        (99_999, "9.9999"),
        (100_100, "10.01"),
        (1, "0.0001"),
        (10_000, "1"),
        (97_600, "9.76"),
    ],
)
def test_prices_serialise_the_way_the_engine_serialises_them(ticks, wire):
    """Shortest decimal, trailing zeros trimmed — Anvil's own `append_price`.

    Byte-for-byte matters because the price on a `trade` frame is the resting
    order's price: a client that sent `9.9900` and compared against `9.99` would
    be doing string equality on two spellings of one number.
    """
    assert format_price(ticks) == wire
    assert parse_price(wire) == ticks


@pytest.mark.parametrize("ticks", [1, 9_999, 100_000, 123_456, 1_000_000_000])
def test_the_price_round_trip_is_exact(ticks):
    assert parse_price(format_price(ticks)) == ticks


def test_a_price_with_five_decimals_is_refused():
    """The wire carries four. A fifth is a client bug, not a rounding decision."""
    with pytest.raises(ValueError, match="four decimals"):
        parse_price("9.99999")


def test_a_non_positive_price_is_refused():
    with pytest.raises(ValueError):
        format_price(0)


# ---------------------------------------------------------------------------
# Order lines
# ---------------------------------------------------------------------------


def test_a_new_order_is_six_fields_with_the_third_blank():
    """The server mints the id and splices it into field 3 (vendored §2)."""
    line = new_line(101, "S", 421, 99_200)
    assert line == "101,N,,S,421,9.92"
    fields = line.split(",")
    assert len(fields) == 6, "five fields is a wrong-column rejection"
    assert fields[2] == "", "the client never mints an id"


def test_a_cancel_carries_the_server_assigned_id():
    line = cancel_line(101, "o7")
    assert line.split(",")[:3] == ["101", "C", "o7"]
    assert len(line.split(",")) == 6, "a cancel is six fields too"


@pytest.mark.parametrize(("side", "qty"), [("X", 10), ("B", 0), ("S", -5)])
def test_a_malformed_order_is_refused_before_it_reaches_the_wire(side, qty):
    with pytest.raises(ValueError):
        new_line(101, side, qty, 99_900)


# ---------------------------------------------------------------------------
# Verdicts and faults
# ---------------------------------------------------------------------------


class _FakeRest:
    """A one-shot HTTP server that answers whatever it was handed."""

    def __init__(self, status: int, body: str, cookie: str | None = None) -> None:
        self.status, self.body, self.cookie = status, body, cookie
        self.requests: list[str] = []
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(4)
        self.port = self.socket.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.socket.accept()
            except OSError:
                return
            with conn:
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                head, _, rest = data.partition(b"\r\n\r\n")
                length = 0
                for line in head.decode("latin-1").split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        length = int(line.split(":", 1)[1])
                while len(rest) < length:
                    rest += conn.recv(4096)
                self.requests.append((head + b"\r\n\r\n" + rest).decode("latin-1"))
                headers = [
                    f"HTTP/1.1 {self.status} X",
                    "Content-Type: application/json",
                    f"Content-Length: {len(self.body)}",
                ]
                if self.cookie:
                    headers.append(f"Set-Cookie: {self.cookie}")
                conn.sendall(
                    ("\r\n".join(headers) + "\r\n\r\n" + self.body).encode("utf-8")
                )

    def close(self) -> None:
        self.socket.close()


@pytest.fixture
def rest():
    servers: list[_FakeRest] = []

    def make(status: int, body: str, cookie: str | None = None) -> _FakeRest:
        server = _FakeRest(status, body, cookie)
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.close()


def test_a_rejected_order_is_a_verdict_not_an_exception(rest):
    """`200 {accepted: false}` is the engine answering. It must not raise."""
    server = rest(200, json.dumps({"accepted": False, "reason": "bad qty", "id": "o2"}))
    venue = Venue(port=server.port)
    result = venue.send("101,N,,B,0,10.00")
    assert isinstance(result, OrderResult)
    assert result.accepted is False
    assert result.reason == "bad qty"
    assert result.id == "o2", "a rejected New still spends its minted id"


@pytest.mark.parametrize("status", [503, 504, 400, 403])
def test_a_non_verdict_status_raises(rest, status):
    """The engine never answered. That is categorically different from a reject."""
    server = rest(status, json.dumps({"reason": "engine busy"}))
    venue = Venue(port=server.port)
    with pytest.raises(TransportFault, match="non-verdict"):
        venue.send("101,N,,B,500,10.00")


def test_the_session_cookie_is_stored_and_replayed(rest):
    """Possession of `anvil_session` is the ownership principal (vendored §2)."""
    server = rest(
        200,
        json.dumps({"accepted": True, "id": "o1"}),
        cookie="anvil_session=deadbeef; HttpOnly; SameSite=Lax; Path=/",
    )
    venue = Venue(port=server.port)
    venue.send("101,N,,B,500,10.00")
    assert venue.session == "deadbeef"
    venue.send("101,C,o1,B,1,1.0")
    assert "Cookie: anvil_session=deadbeef" in server.requests[-1]


def test_a_wire_version_mismatch_refuses_to_start(rest):
    """A bump is a breaking change; the client stops rather than parsing hopefully."""
    server = rest(200, json.dumps({"status": "ok", "wireVersion": 2}))
    venue = Venue(port=server.port)
    with pytest.raises(WireVersionMismatch, match="wire version 2"):
        venue.require_wire_version(1)


def test_the_matching_wire_version_is_accepted(rest):
    server = rest(200, json.dumps({"status": "ok", "wireVersion": 1, "clients": 0}))
    assert Venue(port=server.port).require_wire_version(1)["wireVersion"] == 1


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def _frame(payload: bytes, opcode: int = 0x1, fin: bool = True) -> bytes:
    head = bytes([(0x80 if fin else 0x00) | opcode])
    length = len(payload)
    if length < 126:
        return head + bytes([length]) + payload
    if length < 1 << 16:
        return head + bytes([126]) + struct.pack("!H", length) + payload
    return head + bytes([127]) + struct.pack("!Q", length) + payload


class _FakeStream:
    """A `Stream` wired to a pair of connected sockets, with no handshake.

    The handshake is exercised against the real server in the marked live tests;
    what needs a fake is the byte-level reader, and specifically its behaviour
    when bytes arrive in pieces.
    """

    def __init__(self) -> None:
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        self.client = socket.create_connection(listener.getsockname())
        self.server, _ = listener.accept()
        listener.close()
        self.stream = Stream("127.0.0.1", 0, 101)
        self.stream._socket = self.client

    def feed(self, data: bytes) -> None:
        self.server.sendall(data)

    def close(self) -> None:
        self.server.close()
        self.client.close()


@pytest.fixture
def fake_stream():
    pair = _FakeStream()
    yield pair
    pair.close()


def test_a_whole_frame_decodes(fake_stream):
    payload = json.dumps({"type": "book", "seq": 7, "ticker": 101})
    fake_stream.feed(_frame(payload.encode()))
    assert fake_stream.stream.read(timeout=2.0)["seq"] == 7


def test_a_large_frame_uses_the_sixteen_bit_length(fake_stream):
    """A real book frame is ~8 kB, which is the 126 length form."""
    rows = [{"price": "9.99", "qty": 100, "orders": 1} for _ in range(200)]
    payload = json.dumps({"type": "book", "seq": 9, "ticker": 101, "bids": rows, "asks": []})
    assert len(payload) > 126
    fake_stream.feed(_frame(payload.encode()))
    frame = fake_stream.stream.read(timeout=2.0)
    assert len(frame["bids"]) == 200


def test_a_frame_split_across_reads_survives_a_timeout(fake_stream):
    """The bug this reader was rewritten for.

    Feed a header, poll with a zero timeout — which is the client's steady state
    — and then feed the payload. A reader that consumed the header before the
    body arrived would resume mid-frame and read the payload as a header.
    """
    payload = json.dumps({"type": "trade", "seq": 11, "ticker": 101}).encode()
    whole = _frame(payload)
    fake_stream.feed(whole[:2])
    assert fake_stream.stream.drain(timeout=0.05) == []
    fake_stream.feed(whole[2:])
    frames = fake_stream.stream.drain(timeout=0.5)
    assert [frame["seq"] for frame in frames] == [11]


def test_two_frames_in_one_read_both_decode(fake_stream):
    first = _frame(json.dumps({"type": "book", "seq": 1}).encode())
    second = _frame(json.dumps({"type": "trade", "seq": 2}).encode())
    fake_stream.feed(first + second)
    frames = fake_stream.stream.drain(timeout=0.5)
    assert [frame["seq"] for frame in frames] == [1, 2]


def test_a_fragmented_frame_is_reassembled(fake_stream):
    """Crow does not fragment. A reader that assumed so would fail rarely."""
    payload = json.dumps({"type": "book", "seq": 3, "ticker": 101}).encode()
    half = len(payload) // 2
    fake_stream.feed(_frame(payload[:half], opcode=0x1, fin=False))
    fake_stream.feed(_frame(payload[half:], opcode=0x0, fin=True))
    assert fake_stream.stream.read(timeout=2.0)["seq"] == 3


def test_a_close_frame_ends_the_stream(fake_stream):
    from client.wire import StreamClosed

    fake_stream.feed(_frame(b"", opcode=0x8))
    with pytest.raises(StreamClosed):
        fake_stream.stream.read(timeout=2.0)


def _ping_token(frame: bytes) -> bytes:
    """The payload of a client ping, unmasked.

    Client -> server frames are masked, as RFC 6455 requires, so a fake server
    that echoed the raw bytes back would return a pong the client rightly refuses
    to recognise. Unmasking here is the fake server doing its job, and the fact
    that it is needed is itself a check that the client masks.
    """
    mask, payload = frame[2:6], frame[6:14]
    return bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))


def test_a_ping_is_answered_and_timed(fake_stream):
    """The pong the client waits for is its own token, not any pong.

    Anvil never pings, but the round-trip is the only true freshness signal the
    wire offers, so the path has to work.
    """
    stream = fake_stream.stream
    listener = threading.Thread(
        target=lambda: fake_stream.server.sendall(
            _frame(_ping_token(fake_stream.server.recv(64)), opcode=0xA)
        ),
        daemon=True,
    )
    listener.start()
    elapsed = stream.ping(timeout=5.0)
    assert elapsed >= 0.0
    listener.join(timeout=5.0)


def test_frames_arriving_during_a_ping_are_kept_not_dropped(fake_stream):
    """A freshness measurement must not cost the book updates it justifies."""
    stream = fake_stream.stream

    def respond():
        token = _ping_token(fake_stream.server.recv(64))
        fake_stream.server.sendall(
            _frame(json.dumps({"type": "book", "seq": 42}).encode())
            + _frame(token, opcode=0xA)
        )

    worker = threading.Thread(target=respond, daemon=True)
    worker.start()
    stream.ping(timeout=5.0)
    worker.join(timeout=5.0)
    assert [frame["seq"] for frame in stream.take_pending()] == [42]
    assert stream.take_pending() == [], "pending is drained once"
