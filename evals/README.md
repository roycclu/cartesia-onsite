# Evals

## LLM Evals (run post-call automatically)
Two LLM-as-judge evals run automatically after every call:
- **PII Safety** — did agent share sensitive data before verification? (score 0 = violation)
- **Intent Acknowledgment** — did agent acknowledge caller's request before asking for credentials?

Results logged to compliance_log table with event_type 'eval_pii_safety' and 'eval_intent_acknowledgment'.

## Latency Report (run manually)
```bash
# Local
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/voice_agent python evals/latency_report.py

# Against live DB
DATABASE_URL=<your-db-url> python evals/latency_report.py
```

Reports: perceived latency p50/p75/p95, LLM response time, speculative hit rate, containment rate.
Alerts if: p95 >500ms, speculative hit rate <80%, containment <60%.
