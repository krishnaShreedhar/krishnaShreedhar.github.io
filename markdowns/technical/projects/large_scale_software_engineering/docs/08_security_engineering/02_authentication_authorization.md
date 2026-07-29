---
title: "Authentication and Authorization"
subtitle: "Authentication (AuthN) verifies identity — who are you? Authorization (AuthZ) verifies permission — what are you allowed to do? These are distinct concerns that must be carefully designed for security, usability, and..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-07-08
reading_time: 4
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/08_security_engineering/02_authentication_authorization.html"
---
Authentication (AuthN) verifies identity — who are you? Authorization (AuthZ) verifies permission — what are you allowed to do? These are distinct concerns that must be carefully designed for security, usability, and maintainability.

## OAuth 2.0 and OIDC Flow

```mermaid
sequenceDiagram
    participant User
    participant Client as Client App
    participant AuthServer as Auth Server (Okta/Auth0)
    participant ResourceServer as Resource Server (API)

    User->>Client: Click "Login with Google"
    Client->>AuthServer: Authorization Request\n(client_id, scope, redirect_uri, state)
    AuthServer->>User: Login page
    User->>AuthServer: Credentials + MFA
    AuthServer-->>Client: Authorization Code (short-lived)
    Client->>AuthServer: Token Exchange\n(code + client_secret)
    AuthServer-->>Client: Access Token + ID Token + Refresh Token
    Client->>ResourceServer: API Request\n(Bearer: access_token)
    ResourceServer->>ResourceServer: Validate JWT signature\nCheck expiry, scopes
    ResourceServer-->>Client: Protected resource
```

## JWT Structure

```mermaid
graph LR
    subgraph JWT[JSON Web Token]
        Header[Header\nBase64Url encoded\nalg: RS256\ntyp: JWT]
        Payload[Payload\nBase64Url encoded\nsub: user123\nexp: 1717236000\nscopes: read write\nroles: admin]
        Signature[Signature\nRSA-SHA256\nHeader.Payload\nsigned with private key]

        Header --> Dot1[.] --> Payload --> Dot2[.] --> Signature
    end

    VerifyFlow[Verification:\n1. Decode header + payload\n2. Fetch public key by kid\n3. Verify signature\n4. Check exp not expired\n5. Check aud matches API\n6. Extract claims]
```

## RBAC vs ABAC

```mermaid
graph TD
    subgraph RBAC[Role-Based Access Control - RBAC]
        User1[User Alice] --> Role1[Role: Editor]
        User2[User Bob] --> Role2[Role: Viewer]
        User3[User Carol] --> Role3[Role: Admin]

        Role1 --> Perm1[read_posts\nwrite_posts\ndelete_own_posts]
        Role2 --> Perm2[read_posts]
        Role3 --> Perm3[read_posts\nwrite_posts\ndelete_any_post\nmanage_users]

        Note[Roles are groups of permissions\nUsers assigned to roles\nSimple to understand and audit]
    end

    subgraph ABAC[Attribute-Based Access Control - ABAC]
        Policy[Policy: allow write_post\nif user.department == post.department\nand user.clearance >= post.sensitivity\nand context.time within business_hours]

        Attrs[Attributes evaluated:\nUser: dept=engineering, clearance=3\nResource: dept=engineering, sensitivity=2\nEnvironment: time=10:30am]
    end

    style RBAC fill:#dcfce7,stroke:#16a34a
    style ABAC fill:#dbeafe,stroke:#2563eb
```

## Multi-Factor Authentication

```mermaid
graph TD
    subgraph Factors[MFA Factor Categories]
        Know[Something You Know\nPassword, PIN, Security questions]
        Have[Something You Have\nTOTP app, Hardware key FIDO2\nSMS OTP, Email OTP]
        Are[Something You Are\nBiometrics: fingerprint\nface, voice]
    end

    subgraph Strength[MFA Strength]
        Weak[SMS OTP - weak\nSIM swapping attacks]
        Medium[TOTP app - medium\nPhishing resistant if used with bound certs]
        Strong[FIDO2 / WebAuthn - strong\nPhishing resistant by design\nCryptographic binding to origin]
    end

    Know + Have --> MFA[Multi-Factor Authentication]
    Know + Are --> MFA
    Have + Are --> MFA

    style Strong fill:#dcfce7,stroke:#16a34a
    style Weak fill:#fee2e2,stroke:#dc2626
```

## Key Concepts

- **OAuth 2.0**: An authorization framework, not an authentication protocol. Enables a client application to obtain delegated access to resources on behalf of a resource owner (user). Tokens (access tokens) grant scoped permissions without sharing credentials. The four grant types: Authorization Code (web apps), PKCE (SPAs/mobile), Client Credentials (service-to-service), Device Code (TVs/CLI).

- **OpenID Connect (OIDC)**: An authentication layer on top of OAuth 2.0 that adds the ID token (a JWT containing user identity claims). OIDC enables SSO — once authenticated with the identity provider, users are authenticated across all relying party applications.

- **JWT (JSON Web Token)**: A self-contained token format with three Base64URL-encoded parts: header (algorithm), payload (claims), and signature. The signature allows the recipient to verify authenticity without querying the issuer. JWTs should be short-lived (15-60 minutes) to limit exposure if stolen. Never store sensitive data in JWT payload (it's encoded, not encrypted).

- **RBAC (Role-Based Access Control)**: Users are assigned to roles; roles are assigned permissions. Simpler to understand and audit than ABAC. Best for systems with a small number of distinct roles and clear permission boundaries. Limitation: cannot express context-dependent or attribute-dependent access rules.

- **ABAC (Attribute-Based Access Control)**: Access decisions are based on attributes of the subject (user), resource, and environment, evaluated against policies. Highly flexible — can express "users can access their own data" or "access is allowed only from corporate network during business hours." More complex to implement and audit.

- **API Keys**: Long-lived opaque credentials for service-to-service or developer API access. Unlike JWTs, API keys are not self-describing — the server must look them up in a database to determine permissions. Must be hashed before storage (treat like passwords). Rotate regularly.

- **Password Hashing**: Passwords must be hashed with a slow, salted hashing algorithm designed for passwords: bcrypt, Argon2id, or scrypt. SHA-256/MD5 are NOT acceptable for password storage — they are too fast and allow brute-force attacks.

## Trade-offs

| Approach | Security | Flexibility | Complexity |
|----------|----------|-------------|-----------|
| Session cookies | Good | Low | Low |
| JWT (stateless) | Good | Medium | Medium |
| OAuth 2.0 + OIDC | Excellent | High | High |
| RBAC | Good | Low (rigid roles) | Low |
| ABAC | Excellent | Very high | High |
| SMS MFA | Weak (SIM swap) | High (universal) | Low |
| FIDO2/WebAuthn | Excellent (phishing-resistant) | Medium | Medium |

## When to Use

- **OAuth 2.0 + OIDC**: Any application with user login — do not implement custom authentication
- **Client Credentials**: Service-to-service authentication (service account tokens)
- **RBAC**: Most applications — start with RBAC and add ABAC only when role explosion occurs
- **FIDO2**: High-security applications — require hardware keys for privileged users and admins
- **Short-lived JWTs**: All token-based authentication — combine with refresh tokens for session persistence