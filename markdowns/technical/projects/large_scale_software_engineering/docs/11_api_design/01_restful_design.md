---
title: "RESTful API Design"
subtitle: "REST (Representational State Transfer) is the dominant architectural style for public and internal HTTP APIs. A well-designed REST API is resource-oriented, uses HTTP conventions correctly, and is consistent enough..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-02-04
reading_time: 3
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/11_api_design/01_restful_design.html"
---
REST (Representational State Transfer) is the dominant architectural style for public and internal HTTP APIs. A well-designed REST API is resource-oriented, uses HTTP conventions correctly, and is consistent enough that clients can predict how to interact with new endpoints.

## Resource Hierarchy and URL Design

```mermaid
graph TD
    subgraph URLHierarchy[RESTful URL Hierarchy]
        Collection[Collection Resource\nGET /orders\nPOST /orders]
        Item[Item Resource\nGET /orders/123\nPUT /orders/123\nPATCH /orders/123\nDELETE /orders/123]
        Nested[Nested Resource\nGET /orders/123/items\nPOST /orders/123/items]
        Action[Non-CRUD Action\nPOST /orders/123/cancel\nPOST /orders/123/refund]

        Collection --> Item --> Nested & Action
    end
```

## HTTP Verbs and Semantics

```mermaid
graph TD
    subgraph HTTPVerbs[HTTP Verb Semantics]
        GET[GET\nRead resource or collection\nIdempotent and safe\nCacheable\nNo request body]
        POST[POST\nCreate new resource\nSubmit data\nNot idempotent\nResponse: 201 with Location header]
        PUT[PUT\nFull replacement of resource\nIdempotent\nMust include all fields\nResponse: 200 or 204]
        PATCH[PATCH\nPartial update\nNot necessarily idempotent\nJSONPatch or merge-patch\nResponse: 200]
        DELETE[DELETE\nRemove resource\nIdempotent\nResponse: 204 No Content]
    end
```

## HTTP Status Codes

```mermaid
graph LR
    subgraph StatusCodes[HTTP Status Code Guide]
        S2xx[2xx - Success\n200 OK\n201 Created\n204 No Content\n206 Partial Content]
        S3xx[3xx - Redirection\n301 Moved Permanently\n302 Found\n304 Not Modified]
        S4xx[4xx - Client Errors\n400 Bad Request\n401 Unauthorized\n403 Forbidden\n404 Not Found\n409 Conflict\n422 Unprocessable Entity\n429 Too Many Requests]
        S5xx[5xx - Server Errors\n500 Internal Server Error\n502 Bad Gateway\n503 Service Unavailable\n504 Gateway Timeout]

        S2xx --> S3xx --> S4xx --> S5xx
    end
```

## API Versioning Strategies

```mermaid
graph TD
    subgraph URLPath[URL Path Versioning]
        UP[/v1/orders\n/v2/orders\nMost common\nEasiest to test and debug\nVersion in URL is explicit]
        style UP fill:#dcfce7,stroke:#16a34a
    end

    subgraph Header[Header Versioning]
        H[Accept: application/vnd.api+json;version=2\nCleaner URLs\nHarder to test in browser\nHidden version]
        style H fill:#dbeafe,stroke:#2563eb
    end

    subgraph QueryParam[Query Parameter]
        QP[/orders?version=2\nEasy to default\nOften discouraged]
        style QP fill:#fef3c7,stroke:#d97706
    end

    subgraph Strategy[Versioning Strategy]
        S1[Never break existing consumers]
        S2[Add fields - backward compatible]
        S3[Remove fields - require major version bump]
        S4[Change field semantics - require major version bump]
        S5[Support N and N-1 versions simultaneously]
    end
```

## Pagination Design

```mermaid
graph TD
    subgraph CursorPagination[Cursor-Based Pagination - Recommended]
        CP[GET /orders?cursor=eyJpZCI6MTAwfQ==&limit=20\nResponse includes:\nnext_cursor: eyJpZCI6MTIwfQ==\nhas_more: true]
        CPNote[Stable under concurrent writes\nConstant time O1 regardless of offset\nCannot jump to page N\nBest for feeds and large datasets]
        style CP fill:#dcfce7,stroke:#16a34a
    end

    subgraph OffsetPagination[Offset Pagination - Simpler]
        OP[GET /orders?page=5&per_page=20\nor GET /orders?offset=100&limit=20\nResponse includes:\ntotal: 1500\nnext: /orders?page=6]
        OPNote[Familiar to developers\nAllows jumping to page N\nItems can be missed or duplicated\nif data changes between requests]
        style OP fill:#fef3c7,stroke:#d97706
    end
```

## Key Concepts

- **Resource Naming**: Use nouns, not verbs, for URLs. Resources should be plural: `/users`, `/orders`, `/products`. Sub-resources express hierarchy: `/orders/123/items`. Actions that don't fit CRUD use POST with a verb-noun URL: `POST /orders/123/cancel`.

- **Idempotency**: GET, PUT, and DELETE are idempotent — calling them multiple times produces the same result. POST is not idempotent — repeated calls create multiple resources. Use idempotency keys (client-provided unique ID per request) to make POST safely retryable.

- **HATEOAS**: Hypermedia as the Engine of Application State — responses include links to related actions and resources. Enables client-driven navigation without prior knowledge of URL structure. Rarely implemented in practice due to complexity.

- **API Versioning**: Breaking changes require a new major version. Breaking changes include: removing or renaming fields, changing field types, changing semantics of existing fields. Non-breaking changes (adding optional fields, adding new endpoints) can be deployed to existing versions.

- **Pagination**: Use cursor-based pagination for large, frequently changing datasets (social media feeds). Use offset pagination for stable, sortable datasets where users need to jump to arbitrary pages. Always include a `Link` header or `links` object with next/prev/first/last URLs.

- **Error Response Format**: Standardize error responses across all endpoints. Include: HTTP status code, machine-readable error code, human-readable message, and where appropriate, a validation errors array. The RFC 7807 Problem Details format is a widely adopted standard.

- **Content Negotiation**: Support `Accept` header for response format (JSON, XML) and `Accept-Language` for locale. Always default to JSON for REST APIs.

## Trade-offs

| Decision | Benefit | Cost |
|----------|---------|------|
| URL versioning | Explicit, easy to test | Version prefix in all URLs |
| Cursor pagination | Stable, scalable | Cannot jump to page N |
| HATEOAS | Client discovery | Implementation complexity |
| Strict REST | Consistency, tooling | Some workflows don't fit |

## When to Apply

- Strict REST for public APIs — consumers rely on predictability
- Pragmatic REST for internal APIs — prioritize simplicity over purity
- Never change the semantics of existing fields — add new ones instead
- Support the previous major version for at least 12 months after introducing the next