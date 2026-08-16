# Nimbus Deploy — Frequently Asked Questions

## What is Nimbus Deploy?

Nimbus Deploy is a container deployment platform. You push a Git repository and
it builds, tests, and deploys the result to managed infrastructure without you
configuring servers.

## Which languages are supported?

Official buildpacks exist for Python, Node.js, Go, Ruby, and Java. Anything else
can be deployed using a custom Dockerfile placed in the repository root.

## How much does it cost?

There are three plans:

- **Hobby** — free, 1 project, 512 MB RAM, sleeps after 30 minutes of inactivity
- **Team** — £18 per user per month, unlimited projects, 4 GB RAM per service
- **Enterprise** — custom pricing, dedicated infrastructure, 24/7 support

Bandwidth beyond 100 GB per month is billed at £0.08 per GB on all paid plans.

## What are the deployment limits?

The Hobby plan allows 50 deployments per month. Team and Enterprise plans have
no deployment limit. Build time is capped at 30 minutes on Hobby, 60 minutes on
Team, and 120 minutes on Enterprise.

## How do rollbacks work?

Every deployment is retained for 30 days. Roll back from the dashboard or run
`nimbus rollback --to <deployment-id>`. Rollbacks take about 20 seconds and do
not rebuild the image, because the previous image is reused.

## Is there a database?

Nimbus provides managed PostgreSQL and Redis. PostgreSQL starts at £12 per month
for 10 GB. Redis starts at £8 per month for 1 GB. Automatic daily backups are
retained for 7 days on Team and 30 days on Enterprise.

## What regions are available?

London, Frankfurt, Virginia, Oregon, Singapore, and Mumbai. Choose the region
when creating a project; it cannot be changed afterwards without recreating the
project.

## How do I get support?

Hobby users get community forum support. Team plans include email support with a
24-hour response target. Enterprise includes a dedicated Slack channel and a
1-hour response target for critical issues.
