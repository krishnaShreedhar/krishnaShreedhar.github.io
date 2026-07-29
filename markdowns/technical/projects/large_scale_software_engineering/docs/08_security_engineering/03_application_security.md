---
title: "Application Security"
subtitle: "Application security addresses vulnerabilities in software code and design. The OWASP Top 10 provides the canonical list of critical web application security risks. Addressing these during development is far cheaper..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-08-21
reading_time: 4
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/08_security_engineering/03_application_security.html"
---
Application security addresses vulnerabilities in software code and design. The OWASP Top 10 provides the canonical list of critical web application security risks. Addressing these during development is far cheaper than remediating them after deployment.

## OWASP Top 10 Overview

```mermaid
graph TD
    OWASP[OWASP Top 10 - 2021]

    OWASP --> A1[A01 - Broken Access Control\nMost critical\nUnauthorized data access]
    OWASP --> A2[A02 - Cryptographic Failures\nWeak encryption\nData exposed in transit or at rest]
    OWASP --> A3[A03 - Injection\nSQL NoSQL Command LDAP\nUntrusted data sent to interpreter]
    OWASP --> A4[A04 - Insecure Design\nMissing security controls\nThreat modeling not done]
    OWASP --> A5[A05 - Security Misconfiguration\nDefault credentials\nOpen S3 buckets\nVerbose error messages]
    OWASP --> A6[A06 - Vulnerable Components\nOutdated dependencies\nKnown CVEs]
    OWASP --> A7[A07 - Authn and Authn Failures\nWeak passwords\nNo MFA\nExposed session tokens]
    OWASP --> A8[A08 - Software and Data Integrity\nInsecure CI/CD\nNo signature verification]
    OWASP --> A9[A09 - Security Logging Failures\nNo audit logs\nSecurity events not alerted]
    OWASP --> A10[A10 - SSRF\nServer-Side Request Forgery\nFetch arbitrary URLs]

    style A1 fill:#fee2e2,stroke:#dc2626
    style A3 fill:#fee2e2,stroke:#dc2626
```

## SQL Injection Prevention

```mermaid
graph LR
    subgraph Vulnerable[SQL Injection Vulnerable]
        Input[User input: 1 OR 1=1]
        Query[SELECT * FROM users\nWHERE id = 1 OR 1=1]
        Result[Returns ALL users!\nData breach]
        Input --> Query --> Result
        style Vulnerable fill:#fee2e2,stroke:#dc2626
    end

    subgraph Safe[Parameterized Query - Safe]
        Input2[User input: 1 OR 1=1]
        Prep[Prepared Statement:\nSELECT * FROM users\nWHERE id = dollar sign 1\nBinding: dollar sign 1 = 1 OR 1=1]
        Result2[id column is integer\nCannot be 1 OR 1=1\nQuery fails safely]
        Input2 --> Prep --> Result2
        style Safe fill:#dcfce7,stroke:#16a34a
    end
```

## XSS and CSRF

```mermaid
graph TD
    subgraph XSS[Cross-Site Scripting - XSS]
        XSSInput[Attacker: POST comment\nscript alert document.cookie close script]
        XSSStore[DB stores malicious comment]
        XSSRender[Victim loads page\nbrowser executes attacker script]
        XSSResult[Cookie stolen / account hijacked]
        XSSInput --> XSSStore --> XSSRender --> XSSResult

        XSSFix[Fix: Output encoding\nContent Security Policy\nHTTPOnly cookies]
        style XSSFix fill:#dcfce7,stroke:#16a34a
        style XSSResult fill:#fee2e2,stroke:#dc2626
    end

    subgraph CSRF[Cross-Site Request Forgery]
        CSRFPage[Attacker page: image src equals bank.com/transfer?to=attacker&amount=1000]
        CSRFVictim[Victim visits attacker page\nbrowser sends request with victim cookies]
        CSRFBank[Bank executes transfer]
        CSRFPage --> CSRFVictim --> CSRFBank

        CSRFFix[Fix: CSRF tokens\nSameSite cookies\nOrigin header validation]
        style CSRFFix fill:#dcfce7,stroke:#16a34a
        style CSRFBank fill:#fee2e2,stroke:#dc2626
    end
```

## Security Testing Pipeline

```mermaid
graph LR
    Code[Developer Code] --> SAST[SAST\nStatic Analysis\nSemgrep, CodeQL\nin CI pipeline]
    SAST --> DepScan[Dependency Scanning\nSCA - Snyk, Dependabot\nCVE detection]
    DepScan --> SecTests[Security Unit Tests\nSQL injection tests\nauth bypass tests]
    SecTests --> DAST[DAST\nDynamic Analysis\nOWASP ZAP\nagainst staging]
    DAST --> PenTest[Penetration Testing\nannual or per major release]

    style SAST fill:#dbeafe,stroke:#2563eb
    style DepScan fill:#fef3c7,stroke:#d97706
    style DAST fill:#dcfce7,stroke:#16a34a
```

## Key Concepts

- **SQL Injection**: Untrusted data is inserted into SQL queries as code, allowing attackers to manipulate queries. Prevention: always use parameterized queries (prepared statements) or an ORM that uses them internally. Never concatenate user input into SQL strings. ORMs are not inherently safe if raw query methods are used with string concatenation.

- **Cross-Site Scripting (XSS)**: Attacker injects malicious scripts into content served to other users' browsers. Stored XSS: malicious script is persisted in the database. Reflected XSS: malicious script is reflected in the response immediately. DOM XSS: client-side JavaScript manipulates the DOM insecurely. Prevention: output encoding, Content Security Policy (CSP), HttpOnly cookies.

- **CSRF (Cross-Site Request Forgery)**: A malicious site tricks a user's browser into making an authenticated request to another site using the user's credentials. Prevention: CSRF tokens (unique per session, verified server-side), SameSite cookie attribute (Strict or Lax), checking the Origin/Referer header.

- **SSRF (Server-Side Request Forgery)**: A vulnerability where the server can be induced to make HTTP requests to arbitrary URLs, including internal services. An attacker can use this to reach cloud metadata endpoints (AWS IMDSv1 at 169.254.169.254) and steal IAM credentials. Prevention: validate and allowlist URLs the server can fetch, block metadata IPs, use IMDSv2.

- **SAST (Static Application Security Testing)**: Analyzes source code for security vulnerabilities without executing it. Fast, integrated into CI pipelines. Produces false positives — requires tuning. Examples: Semgrep, CodeQL, SonarQube, Checkmarx.

- **DAST (Dynamic Application Security Testing)**: Tests a running application by sending malicious inputs and observing responses. Finds vulnerabilities that SAST misses (runtime configuration, business logic flaws). Examples: OWASP ZAP, Burp Suite. Slower than SAST — typically run against staging environments.

- **SCA (Software Composition Analysis)**: Inventories third-party dependencies and checks them against CVE databases. Snyk, Dependabot, OWASP Dependency-Check. Critical — the log4shell vulnerability (Log4j2, CVE-2021-44228) affected millions of applications through a transitive dependency.

## Trade-offs

| Control | Effectiveness | Developer Friction |
|---------|-------------|-------------------|
| Parameterized queries | High | Low (ORM handles it) |
| Output encoding | High | Low (templating engines) |
| CSP headers | High | Medium (policy tuning) |
| SAST in CI | Medium | Low |
| DAST | High | Medium (staging required) |
| Penetration testing | Very high | Low (external team) |

## When to Apply

- Parameterized queries: always — no exceptions
- Output encoding: whenever rendering user-controlled data in HTML
- CSRF tokens: all state-changing requests from browser clients
- Dependency scanning: run in every CI pipeline, block on high/critical CVEs
- SAST: integrate into CI pipelines as a non-blocking informational check initially, harden over time