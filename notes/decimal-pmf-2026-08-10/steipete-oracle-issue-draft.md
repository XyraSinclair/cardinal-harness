# POSTED 2026-08-11 as https://github.com/steipete/oracle/issues/367 (Xyra approved 2026-08-11 23:41)

Title: Browser engine's cookie-copy mode can silently invalidate the user's live
ChatGPT session (token rotation race) — propose a persistent automation profile

## Body

**Problem.** When the browser engine copies ChatGPT session cookies out of the
user's real Chrome profile into a separate automation browser (the launch path
used when not attaching to a running Chrome), it clones a *live* session token.
ChatGPT rotates session tokens on use. The rotation lands in the automation
browser's cookie jar; the user's real browser keeps the pre-rotation token.
Depending on server-side grace behavior, the user's live ChatGPT session then
degrades — requests fail or the account behaves logged-out-ish until the user
forces a fresh session (e.g. switching accounts and back). The failure is
delayed and detached from the oracle run that caused it, so users don't
attribute it to oracle.

Observed repeatedly under heavy use (multiple background oracle consults per
night, macOS, Chrome stable, oracle 0.17.1): the user's interactive
chatgpt.com sessions intermittently broke in exactly this pattern, and the
breakage stopped when we removed cookie copying from the flow.

**Why `--browser-attach-running` doesn't have this bug.** Attaching to the
user's real Chrome uses the session *in place*: the rotated token lands in the
same jar that owns it, so everything stays coherent. The race exists only when
cookies are cloned into a second browser. But attach mode requires driving the
user's real browser, which background/agent use cases can't accept.

**Proposal.** Add a first-class *persistent automation profile* mode:

1. `oracle setup-profile` launches Chrome on a dedicated durable user-data-dir
   (e.g. `~/.local/state/oracle/chrome-profile`) and asks the user to sign in
   to ChatGPT once in that window.
2. Subsequent browser-engine runs use that profile directly (or clone from it
   per-run) and never read the user's real Chrome profile.
3. Cookie-copy from the real profile stays available as an explicit opt-in
   fallback with a warning naming this hazard.

This also fixes a second, related pain: the in-app cookie copy frequently
misses the real session (surfacing as "Unable to locate the ChatGPT model
selector button", which is an auth failure in disguise). A profile the user
signed into once is authenticated by construction, keeps a stable browser
fingerprint for Cloudflare clearance, and quarantines token rotation entirely
to the automation side.

We run this design as a wrapper in production (background `open -g` Chrome +
seeded profile) and can share the shell implementation if useful:
https://github.com/XyraSinclair/quiet-oracle
