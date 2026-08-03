---
name: avoid-caching-prefer-wait
description: User strongly prefers re-computation over caching; a brief wait is usually cheaper than the long-tail cost of cache cleverness
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a8a4325f-377f-4561-8637-1c6db75d6fcd
---

**Avoid caching if possible. A user wait is usually WAY less "costly" down the line than cleverness now.**

**Why:** Caches introduce staleness bugs, invalidation logic, debugging surprise ("but I plugged it in!"), and ongoing maintenance. The "cleverness tax" — bugs, complexity, support load — compounds over time. A few extra seconds of startup is a one-time cost the user pays; a stale cache is a recurring tax everyone pays. Confirmed in the context of evaluating an EdsonLuiz fork that cached OpenVINO device detection to `devices.json` — solved a 2-4s OpenVINO import cost but introduced stale-on-hardware-change failure modes.

**How to apply:**
- When proposing an optimization, default to "just re-compute it" rather than "cache it."
- Only suggest caching when (a) the wait is demonstrably painful, (b) the cache invalidation story is clean and obvious, and (c) staleness is harmless or self-healing.
- Be skeptical of caches that store discovered hardware/environment state — those are stale-prone by definition.
- For things like model lists, device lists, file scans: prefer fresh queries on every run.

**Sanctioned exception (2026-06-28):** the user chose **prefix (KV) caching default-on** in NoLlama for agent loops — it meets all three criteria above: (a) re-prefilling a ~21k-token agent system prompt every turn is demonstrably painful (24.4s→0.5s, ~47×), (b) invalidation is exact and automatic (any prefix byte change), (c) staleness is impossible — it's KV for an exact prefix match, self-healing by re-prefill. So this is *consistent* with the rule, not a violation. The rule still bars stale-prone state caches (device/model/file scans). See [[openclaw-nollama-integration]].

Related: [[intel-llm-stack-landscape]] — the fork that prompted this conversation.
