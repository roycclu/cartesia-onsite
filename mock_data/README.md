# Mock Data

This directory simulates Acme Insurance's internal data systems for prototype purposes only.

In production these would be replaced by:
- **Claims data** — read-only REST API provided by Acme Engineering team (Claim Status API)
- **Policy data** — read-only REST API provided by Acme Engineering team (Policy Lookup API)
- **Identity verification** — read-only query against Acme's existing policy/SSN store, requires VPC peering and IAM access granted by Acme Infra team

No data in this directory would exist in a production deployment.
All production data access is read-only and subject to week 1 API contract sign-off.
