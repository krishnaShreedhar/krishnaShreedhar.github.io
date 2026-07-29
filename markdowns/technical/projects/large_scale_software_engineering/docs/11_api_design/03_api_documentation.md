---
title: "API Documentation"
subtitle: "API documentation is the primary interface between API providers and consumers. Good documentation enables developers to integrate quickly without support. The OpenAPI Specification is the industry standard for..."
category: technical
project: large_scale_software_engineering
project_title: "Large Scale Software Engineering"
date: 2025-11-16
reading_time: 3
tags:
  - large-scale-software-engineering
  - docs
author: "Shreedhar Kodate"
output: "blogs/technical/posts/large_scale_software_engineering/docs/11_api_design/03_api_documentation.html"
---
API documentation is the primary interface between API providers and consumers. Good documentation enables developers to integrate quickly without support. The OpenAPI Specification is the industry standard for documenting REST APIs; AsyncAPI covers event-driven APIs.

## OpenAPI Specification Structure

```mermaid
graph TD
    subgraph OpenAPIDoc[OpenAPI 3.1 Document Structure]
        Info[info\ntitle version description\ncontact license]
        Servers[servers\nproduction staging]
        Paths[paths\nendpoints operations]
        Components[components\nschemas securitySchemes\nresponses parameters]
        Security[security\nglobal auth requirements]
        Tags[tags\ngrouping for UI]
    end

    subgraph PathItem[Path Item Structure]
        Op[Operation\nGET POST PUT PATCH DELETE]
        Op --> Params[parameters\npath query header]
        Op --> ReqBody[requestBody\nschema validation]
        Op --> Responses[responses\n200 201 400 401 422 500]
        Op --> Auth[security\noverride global auth]
    end

    Info & Servers & Paths & Components & Security & Tags --> OpenAPIDoc
    Paths --> PathItem
```

## OpenAPI Example

```mermaid
graph LR
    subgraph Spec[OpenAPI Spec Concepts]
        Schema[Schema Object\ntype: object\nproperties:\n  id: string uuid\n  status: string\n  amount: number\nrequired: id status amount]

        Response[Response Object\ndescription: Order created\ncontent:\n  application/json:\n    schema: ref OrderSchema\nheaders:\n  Location: url]

        Parameter[Parameter Object\nin: path\nname: order_id\nrequired: true\nschema: type string uuid]
    end
```

## Documentation-Driven Development

```mermaid
graph LR
    DesignFirst[Design API First\nWrite OpenAPI spec\nbefore implementation] --> Review[API Review\nWith consumers\nBefore building]
    Review --> Generate[Code Generation\nServer stubs\nClient SDKs\nTest mocks]
    Generate --> Implement[Implement Server\nSpec is the contract]
    Implement --> Validate[Validate Implementation\nAgainst spec\nContract testing]
    Validate --> Publish[Publish Docs\nSwagger UI ReDoc\nDeveloper portal]

    style DesignFirst fill:#dcfce7,stroke:#16a34a,stroke-width:2px
```

## AsyncAPI for Event-Driven APIs

```mermaid
graph TD
    subgraph AsyncAPIDoc[AsyncAPI Document - Kafka API]
        Channels[channels\norders.placed\norders.cancelled\npayments.processed]
        Messages[messages\nOrderPlaced payload schema\nOrderCancelled payload schema]
        Operations[operations\npublish OrderPlaced\nsubscribe PaymentProcessed]
        Bindings[bindings\nkafka partition key\nconsumer group]
    end

    Channels --> Messages & Operations & Bindings
```

## Key Concepts

- **OpenAPI Specification (OAS)**: A language-agnostic standard for describing HTTP APIs. The spec document (YAML or JSON) describes every endpoint, parameter, request body, response schema, and security requirement. Tools generate documentation UIs (Swagger UI, ReDoc), client SDKs, server stubs, and test cases from the spec.

- **Design-First vs. Code-First**: Design-first writes the API spec before implementation, enabling API design review with consumers before engineering effort is committed. Code-first generates the spec from annotations in code — faster to start but risks implementation-shaped API design.

- **API Reference Documentation**: Documents every endpoint: what it does, all parameters, request and response schemas, possible error codes, and example requests/responses. Must be accurate — wrong documentation is worse than no documentation.

- **Getting Started Guide**: A tutorial-style guide that walks a new developer from zero to first successful API call in 15 minutes. Code examples in popular languages (Python, JavaScript, Go, Java). This is often more important than the reference documentation for developer adoption.

- **AsyncAPI**: The OpenAPI equivalent for event-driven and message-based APIs. Describes Kafka topics, AMQP exchanges, WebSocket channels — the messages they carry, their schemas, and the operations (publish/subscribe). Enables generating consumers and producers from the spec.

- **SDK Generation**: OpenAPI specs can be used to generate typed client SDKs (using tools like OpenAPI Generator, Speakeasy, Stainless). SDKs dramatically improve developer experience — instead of constructing raw HTTP requests, developers use idiomatic library calls with type safety.

- **Changelog and Deprecation Policy**: Document every API change. Announce deprecations in the API response via `Sunset` and `Deprecation` headers. Give consumers at least 12 months notice before removing deprecated functionality.

## Trade-offs

| Approach | DX Quality | Engineering Overhead |
|---------|------------|---------------------|
| Design-first | High (reviewed design) | Medium |
| Code-first | Lower (implementation-shaped) | Lower |
| Hand-written docs | Potentially high | High (maintenance) |
| Generated SDK | Very high | Medium (spec quality matters) |
| Interactive docs (Swagger) | High | Low (generated from spec) |

## When to Apply

- Design-first for all public or partner-facing APIs
- Code-first for internal APIs with a single consumer team
- Always publish machine-readable specs (OpenAPI) even for internal APIs — enables contract testing
- Generate SDKs for public APIs with significant developer ecosystems