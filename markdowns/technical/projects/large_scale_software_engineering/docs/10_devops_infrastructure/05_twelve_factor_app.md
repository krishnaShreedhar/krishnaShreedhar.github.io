# The Twelve-Factor App

The Twelve-Factor App is a methodology for building software-as-a-service applications that are portable, scalable, and maintainable. Formalized by Heroku engineers based on patterns from deploying thousands of applications, these twelve factors remain the canonical guide for cloud-native application design.

## The Twelve Factors

```mermaid
mindmap
  root((12-Factor\nApp))
    I - Codebase
      One codebase in version control
      Many deploys from same codebase
    II - Dependencies
      Explicitly declare and isolate
      Never rely on system packages
    III - Config
      Store in environment not code
      Config varies per deploy
    IV - Backing Services
      Treat as attached resources
      Swap without code changes
    V - Build Release Run
      Strictly separated stages
      Immutable releases
    VI - Processes
      Stateless and share-nothing
      Persist to backing services
    VII - Port Binding
      Export via port binding
      Self-contained HTTP server
    VIII - Concurrency
      Scale via process model
      Horizontal scaling
    IX - Disposability
      Fast startup and graceful shutdown
      Designed for crashes
    X - Dev-Prod Parity
      Keep environments similar
      Minimize time code-deploy-gap
    XI - Logs
      Treat as event streams
      stdout not files
    XII - Admin Processes
      Run as one-off processes
      Same environment as app
```

## Config Management (Factor III)

```mermaid
graph LR
    subgraph Violation[Violation: Config in Code]
        Code[config.py\nDB_HOST = production-db.internal\nAPI_KEY = sk-prod-abc123\nDEBUG = False]
        Problem[Hardcoded for one environment\nSecrets in git\nDifferent config per env requires code changes]
        style Code fill:#fee2e2,stroke:#dc2626
        style Problem fill:#fee2e2,stroke:#dc2626
    end

    subgraph Correct[Correct: Config in Environment]
        EnvVars[Environment Variables\nDB_HOST loaded from env\nAPI_KEY from secrets manager\nDEBUG from env]
        SameCode[Same code deployed to\ndev, staging, production\nDifferent env vars per environment]
        style EnvVars fill:#dcfce7,stroke:#16a34a
        style SameCode fill:#dcfce7,stroke:#16a34a
    end
```

## Build-Release-Run (Factor V)

```mermaid
graph LR
    Code[Source Code\ngit commit abc123] --> Build[Build Stage\nCompile, package\nDocker image: app:1.2.3]
    Build --> Release[Release Stage\nImage + config\napp:1.2.3 + prod-config\nImmutable - never changed]
    Release --> Run[Run Stage\nExecute the release\nin the runtime environment]

    subgraph ReleaseImmutability[Immutable Releases]
        R1[Release 1.2.3 - deployed Monday]
        R2[Release 1.2.4 - deployed Tuesday]
        R3[Bug found - rollback to 1.2.3\nNo code change needed]
    end

    style Release fill:#fef3c7,stroke:#d97706,stroke-width:2px
```

## Disposability (Factor IX)

```mermaid
graph TD
    subgraph FastStartup[Fast Startup]
        FS[Application starts in seconds\nReadiness probe passes quickly\nKubernetes can schedule\nand start pods rapidly\nScale-out adds capacity fast]
        style FS fill:#dcfce7,stroke:#16a34a
    end

    subgraph GracefulShutdown[Graceful Shutdown - SIGTERM]
        GS1[Receive SIGTERM]
        GS2[Stop accepting new requests]
        GS3[Complete in-flight requests within 30s]
        GS4[Flush buffers and close connections]
        GS5[Exit cleanly]
        GS1 --> GS2 --> GS3 --> GS4 --> GS5
        style GS1 fill:#fef3c7,stroke:#d97706
    end
```

## Key Concepts

- **I. Codebase**: One codebase per application tracked in version control. Multiple deploys (dev, staging, prod) come from the same codebase. Different apps are different codebases, with shared code extracted into libraries.

- **II. Dependencies**: All dependencies must be explicitly declared in a dependency manifest (requirements.txt, package.json, go.mod) and never rely on implicit system-level packages. Dependency isolation (virtualenv, node_modules) ensures the declared dependencies are the actual runtime dependencies.

- **III. Config**: Any configuration that varies between deploys (database URLs, API keys, feature flags) must be stored in environment variables, not in code. The test: could the codebase be made public without compromising credentials? Config-in-code fails this test.

- **IV. Backing Services**: Treat all backing services (databases, caches, message brokers, email services) as attached resources, accessible via URL in config. Swap a local Postgres for an RDS instance without code changes — only the config URL changes.

- **V. Build, Release, Run**: Strictly separate the build stage (compile code), release stage (combine build with config into an immutable release), and run stage (execute the release). Immutable releases enable rollback — redeploy a previous release.

- **VI. Processes**: Applications execute as stateless, share-nothing processes. State is externalised to backing services (databases, caches). This enables horizontal scaling — any process can handle any request.

- **XI. Logs**: Applications should write logs to stdout as an event stream without managing log files or log rotation. The execution environment (Docker, Kubernetes) captures stdout and routes it to the appropriate log aggregator.

- **XII. Admin Processes**: Database migrations, one-time scripts, and admin tasks run as one-off processes in the same environment as the application (same release, same config). Never run admin tasks on production machines with a different environment than the app.

## Trade-offs

| Factor | Benefit | Requirement |
|--------|---------|-------------|
| Config in env vars | One codebase for all envs | Env var management infrastructure |
| Stateless processes | Trivial horizontal scaling | External state stores |
| Immutable releases | Instant rollback | Artifact storage, registry |
| Logs as stdout | No log management in app | Log aggregation infrastructure |

## When to Apply

- Apply all twelve factors to new applications from day one
- Retroactively applying to existing apps: start with Factor III (config) and Factor VI (stateless) as they have the highest ROI for cloud deployment
- Factor XI (logs as stdout) is immediately beneficial for containerized applications
