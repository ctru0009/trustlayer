# Security policy

## Scope

This is a learning project, not production software. The demo credentials
(`alice`, `bob`, `carol`, `admin` with no passwords) and the dev JWT secret
are intentional. They demonstrate the permission model; they are not a
vulnerability.

## What counts

Anything that breaks the core promise: a search, answer, classification, or
document fetch that reveals content (or its existence) to a user without
permission. That includes ACL bypasses in the gateway or service, distinct
error behaviour for restricted vs missing documents, and leakage through logs,
latency, citations, or error messages.

Out of scope: the demo auth design itself, missing rate limiting, missing
TLS, dependency CVEs in pinned-but-aging packages (report those only if you
can show exploitability through this repo's code paths).

## Reporting

Do not open a public issue. Email the maintainer (see the repo profile) with:

- what you did, step by step, against which commit,
- what you observed vs what the permission model says should happen,
- whether the direct path (`make leak`) or the API path (`make api-leak`)
  catches it, if you checked.

I will confirm receipt within a few days and fix verified access-control bugs
before anything else in the queue. No bug bounty. Credit in the fix commit
message if you want it.
