"""The Anvil participant — the only code in this repo that touches a network.

Constitution invariant 8 states the seam from one side: nothing under `temper/`
may import a module that can reach a socket. This package is the other side, and
`tests/test_repo_invariants.py` asserts both halves — that the package is clean,
and that `client/` is the only place the imports appear.

What lives here is a *client*, not a strategy. It works one parent order on a
live Anvil book by asking the trained policy for a fraction of remaining
inventory each bin, pricing that quantity to cross the book it just observed,
and reading back what actually filled. Everything it knows about the wire it
knows from `docs/vendor/anvil-protocol.md`, the versioned snapshot of Anvil's
`PROTOCOL.md`; code here cites that document by section and never restates a
frame schema, because a copy drifts and a drifted copy reads as authority.

Three deliberate constraints.

**Standard library plus numpy, and nothing else.** `requirements.txt` is
unchanged by this package: REST goes through `http.client`, the event stream
through a hand-rolled RFC 6455 reader over `socket`, and the policy's forward
pass through numpy. Anvil's own repo drives its wire from a raw socket in
`tests/tools/pong_ordering_probe.py`, so this is the house pattern rather than
an economy.

**No torch.** The policy is read out of the committed `.npz` by
`client.inference`, which is a second implementation of the forward pass pinned
against the training-time one on the checkpoint's own evaluation trajectory.
That keeps the training stack off the client's import path — the seam
`ARCHITECTURE.md` §3 already implies — and it is the first step of the backlog's
C++/ONNX inference leg rather than a detour from it.

**Integer ticks, not floats.** Anvil's prices are integers on the wire (four
decimal places, serialised as shortest-decimal strings), so this package works in
ticks and converts only at the boundary. That is not a reversal of §7's *floats,
not integer ticks*: that decision is about Temper's own simulator, whose dynamics
are analytic and whose exactness is enforced by goldens. Here the venue really is
integer-priced, and matching its arithmetic is what makes a predicted fill
comparable with a realised one byte for byte.

**This is a demo, not an evaluation** (`ARCHITECTURE.md` §7). The flow is
synthetic, the sample is one order, and there is no baseline the number can
fairly be compared against. Nothing here may be quoted as execution quality.
"""

from __future__ import annotations

__all__ = []
