---
name: vendor-cli-api-wrap-tos-risk
description: "Raise ToS / account-risk concerns BEFORE building, when a user proposes wrapping a vendor's interactive CLI as an HTTP/API backend."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 926aecde-d39c-4168-810c-9325f0126746
---

When a user proposes wrapping a vendor's interactive CLI tool (GitHub Copilot
CLI, Cursor agent, etc.) as an HTTP/OpenAI-compatible API so other apps can
hit it, **flag the ToS and account-risk concern up front**, before any code is
written.

**Why:** On 2026-05-26 the user (`tommyl_qfree`, company SSO via Q-Free)
proposed wrapping `copilot.exe` to expose its Haiku 4.5 backend as an API for
a NoLlama-style web UI. I built the scaffold without raising the obvious ToS
issue. The user spotted it themselves: "I have a sneaking suspicion that
making an API for their cli-tool will likely anger Github?" — correct
suspicion. The blast radius is amplified for company SSO accounts because
abuse flags can spill into the employer's Copilot Business/Enterprise
relationship, not just the individual seat.

**How to apply:** When the proposal is "wrap a vendor interactive CLI as an
HTTP/API endpoint", before generating code:
  - Name the ToS / acceptable-use concern explicitly **once**.
  - Note that quotas are typically sized for interactive human use, not
    programmatic multiplexing.
  - Call out SSO/enterprise account risk specifically if the login is a
    company seat.
  - Suggest blessed alternatives and let the user choose.

**Then drop it.** If the user confirms personal-scope use and the vendor
explicitly ships automation surfaces (e.g. `copilot -p`, `--allow-all-tools`,
`--output-format json`), the vendor anticipates this category of use. Don't
relitigate ToS turn after turn. The user from the 2026-05-26 spike called
this out directly: "if Copilot CLI didn't want me to automate stuff, why ship
`-p`?" — correct. Abuse-detection language in projects like ericc-ch/copilot-api
targets *volume* (rapid/bulk fanout, multi-client multiplexing), not the act
of automating one's own workflow. Raise once, calibrate to user scope, move on.
