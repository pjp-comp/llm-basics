# Engineering Notes — Internal

## Incident 2026-03-14: cache stampede

At 09:12 UTC the session cache expired for all keys simultaneously because every
entry had been written during a single migration the previous night, giving them
identical TTLs. When they expired together, roughly 40,000 requests hit
PostgreSQL within four seconds and the connection pool was exhausted.

The fix was to add jitter: TTLs are now set to the base value plus a random
offset between 0 and 300 seconds. We also raised the pool size from 20 to 60.

Recovery took 11 minutes. The lesson is that identical TTLs are a hidden
coupling — anything written in a batch will expire in a batch.

## Why we moved off the shared queue

The shared RabbitMQ instance served both the deployment pipeline and the metrics
ingest. Metrics traffic is bursty and would fill the queue, delaying deployment
jobs by up to nine minutes during spikes.

We split them in January 2026. Deployments now use a dedicated instance, and
metrics moved to Kafka. Deployment queue latency dropped from a p99 of 9 minutes
to under 20 seconds.

## Database migration policy

All migrations must be backwards compatible with the previous release. We deploy
the migration first, then the code that uses it, never together. Dropping a
column requires two releases: one that stops writing to it, and a later one that
removes it.

Migrations that take longer than 30 seconds must be run with the online schema
change tool rather than directly, because direct ALTER statements hold locks.

## Rate limiting design

We use a token bucket per API key, stored in Redis. The bucket refills at the
plan's rate and holds a burst of double the per-second limit. Hobby keys get 10
requests per second, Team gets 100, and Enterprise is negotiated per contract.

Rejected requests return HTTP 429 with a `Retry-After` header. We deliberately do
not queue rejected requests, because queuing turns a rate limit problem into a
latency problem.

## On-call rotation

The rotation is one week long, handed over on Wednesday at 10:00 UTC rather than
Monday, so that a difficult handover does not land next to a weekend. Secondary
on-call exists only for Enterprise-severity incidents.
