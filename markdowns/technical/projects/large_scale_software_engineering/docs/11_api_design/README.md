---
title: "API Design"
subtitle: "APIs are contracts between systems. A well-designed API is intuitive, stable, secure, and self-documenting. A poorly designed API is a maintenance burden that breaks clients on every change. API design decisions are..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-08-05
reading_time: 1
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/11_api_design/index.html"
---
APIs are contracts between systems. A well-designed API is intuitive, stable, secure, and self-documenting. A poorly designed API is a maintenance burden that breaks clients on every change. API design decisions are among the hardest to reverse because they affect all existing consumers.

## Overview

```mermaid
mindmap
  root((API Design))
    RESTful Design
      Resource modeling
      HTTP verbs
      Status codes
      HATEOAS
      Versioning
      Pagination
    API Security
      Authentication
      Authorization
      Rate limiting
      Input validation
      TLS
      CORS
    API Documentation
      OpenAPI Specification
      AsyncAPI
      Developer experience
      SDKs and examples
    GraphQL Design
      Schema design
      Query optimization
      N plus 1 problem
      Subscriptions
      Federation
```

## Topics in This Section

| File | Topic | Key Concepts |
|------|-------|--------------|
| [01_restful_design.md](01_restful_design.md) | RESTful Design | Resources, HTTP verbs, versioning, pagination |
| [02_api_security.md](02_api_security.md) | API Security | AuthN/AuthZ, rate limiting, CORS |
| [03_api_documentation.md](03_api_documentation.md) | API Documentation | OpenAPI, AsyncAPI, developer experience |
| [04_graphql_design.md](04_graphql_design.md) | GraphQL Design | Schema, N+1 problem, federation |