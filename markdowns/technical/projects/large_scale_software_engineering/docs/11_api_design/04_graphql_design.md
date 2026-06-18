# GraphQL Design

GraphQL is a query language and runtime for APIs that enables clients to request exactly the data they need. Unlike REST (where the server defines what each endpoint returns), GraphQL puts data-fetching control in the client's hands — eliminating over-fetching and under-fetching.

## GraphQL Architecture

```mermaid
graph TD
    subgraph GraphQLLayer[GraphQL API Layer]
        Schema[GraphQL Schema\nType definitions\nQuery Mutation Subscription]
        Resolvers[Resolvers\nField-level data fetching]
        DataLoaders[DataLoader\nBatch and cache DB calls\nSolves N+1 problem]
    end

    subgraph Clients[Clients]
        Mobile[Mobile App\nRequest only id name for list view]
        Web[Web SPA\nRequest all fields for detail view]
        Dashboard[Admin Dashboard\nRequest aggregated analytics fields]
    end

    subgraph Backends[Data Sources]
        UserDB[(User DB)]
        OrderDB[(Order DB)]
        ProductSvc[Product Service]
        RecommSvc[Recommendation Service]
    end

    Mobile & Web & Dashboard --> Schema
    Schema --> Resolvers
    Resolvers --> DataLoaders
    DataLoaders --> UserDB & OrderDB & ProductSvc & RecommSvc

    style Schema fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style DataLoaders fill:#dcfce7,stroke:#16a34a
```

## N+1 Problem and DataLoader Solution

```mermaid
sequenceDiagram
    participant Client
    participant GraphQL
    participant DB

    Note over Client,DB: Without DataLoader - N+1 Problem
    Client->>GraphQL: query: orders with user names
    GraphQL->>DB: SELECT * FROM orders LIMIT 10
    DB-->>GraphQL: 10 orders
    loop for each of 10 orders
        GraphQL->>DB: SELECT * FROM users WHERE id = ?
        DB-->>GraphQL: user record
    end
    Note over DB: 11 queries total!

    Note over Client,DB: With DataLoader - Batched
    Client->>GraphQL: query: orders with user names
    GraphQL->>DB: SELECT * FROM orders LIMIT 10
    DB-->>GraphQL: 10 orders (user_ids: 1,2,3,4,5)
    GraphQL->>DB: SELECT * FROM users WHERE id IN (1,2,3,4,5)
    DB-->>GraphQL: 5 users
    Note over DB: 2 queries total!
```

## GraphQL Schema Design

```mermaid
graph TD
    subgraph Schema[Schema Design Principles]
        Types[Type System\nObject types\nInput types\nEnum types\nScalar types\nInterface types\nUnion types]

        Queries[Query Type\nRead-only operations\nFetchable from any client\nShould be deterministic]

        Mutations[Mutation Type\nState-changing operations\nReturn affected objects\nInput type per mutation]

        Subscriptions[Subscription Type\nReal-time updates\nWebSocket transport\nFor live data feeds]
    end

    subgraph BestPractices[Schema Design Best Practices]
        BP1[Use connection pattern for lists\norders edge node and pageInfo\nnot just orders array]
        BP2[Return modified object from mutations\nnot just success boolean]
        BP3[Nullable by default - explicit non-null]
        BP4[Deprecate fields with reason]
        BP5[Consistent naming conventions]
    end
```

## GraphQL Federation

```mermaid
graph TD
    Client[Client] --> Gateway[Apollo Federation Gateway\nor GraphQL Mesh]

    subgraph Subgraphs[Federated Subgraphs]
        UserGraph[User Subgraph\ntype User at key id\nname email profile]
        OrderGraph[Order Subgraph\nextends User type\norders: Order items]
        ProductGraph[Product Subgraph\ntype Product\nname price inventory]
    end

    Gateway --> UserGraph & OrderGraph & ProductGraph

    subgraph EntityResolution[Cross-Subgraph Entity Resolution]
        Query[query: user 123 with orders and products]
        Step1[Gateway: fetch User from UserGraph]
        Step2[Gateway: fetch Orders from OrderGraph by user_id]
        Step3[Gateway: fetch Products from ProductGraph by product_ids]
        Query --> Step1 --> Step2 --> Step3
    end
```

## Key Concepts

- **Schema as Contract**: The GraphQL schema defines the entire API surface — all types, fields, queries, mutations, and subscriptions. It is strongly typed and self-documenting (via introspection). Changes to the schema affect all clients, so schema evolution requires the same care as REST versioning.

- **Resolvers**: Each field in the schema has a resolver function that fetches the data for that field. Resolvers can call databases, microservices, or any data source. Nested resolvers enable composing data from multiple sources in a single query.

- **N+1 Problem**: A common GraphQL performance anti-pattern. When resolving a list of objects, and each object's resolver makes a database call for a related entity, the result is N+1 queries for N items. The DataLoader pattern solves this by batching and deduplicating resolver calls within a single request.

- **DataLoader**: A utility (originally by Facebook) that batches multiple resolver calls in a single tick into one batched data load. All individual "fetch user by ID" calls made during one request are batched into a single "fetch users by IDs" call. Also caches results within the request to avoid duplicate fetches.

- **Mutations and Input Types**: Mutations should use dedicated Input types (not the same types as output objects) to allow separate evolution of write and read contracts. Mutations should return the modified objects so clients can update their caches without additional queries.

- **Federation (Apollo Federation)**: A microservices pattern for GraphQL where each team owns a subgraph (partial schema). A gateway stitches subgraphs into a unified schema. Entities (types with a key) can be extended across subgraphs — the Order subgraph can add `orders` field to the `User` type owned by the User subgraph.

- **Introspection**: GraphQL APIs are self-describing — clients can query the schema itself (`__schema`, `__type` queries). Tools like GraphiQL use introspection to provide autocompletion and documentation. Disable introspection in production for public APIs to limit schema exposure.

## Trade-offs

| Aspect | GraphQL | REST |
|--------|---------|------|
| Client control | High | Low |
| Over/under fetching | Solved | Common |
| Caching | Hard (query-specific) | Easy (HTTP cache) |
| N+1 risk | High | Low |
| Type safety | Built-in | Via OpenAPI |
| Tooling maturity | Growing | Very mature |
| Learning curve | Higher | Lower |

## When to Use

- **GraphQL**: APIs with many diverse clients (mobile, web, partner) with different data needs; complex graph-shaped data; frontend teams that want control over data fetching
- **Avoid**: Simple CRUD APIs, APIs with heavy caching requirements, teams not ready for DataLoader complexity
- **Federation**: When multiple teams own different parts of the graph and need independent deployment
