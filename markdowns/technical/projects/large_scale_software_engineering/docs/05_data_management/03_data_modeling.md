# Data Modeling

Data modeling is the process of designing the structure and relationships of data for storage and retrieval. Good data models balance normalization (preventing redundancy) with denormalization (optimizing for access patterns), and evolve with the system without costly migrations.

## Entity-Relationship Modeling

```mermaid
graph TD
    subgraph ERDiagram[E-R Model - E-Commerce]
        User[User\nuser_id PK\nname\nemail\nphone]
        Order[Order\norder_id PK\nuser_id FK\nstatus\ncreated_at]
        OrderItem[OrderItem\nitem_id PK\norder_id FK\nproduct_id FK\nqty\nprice]
        Product[Product\nproduct_id PK\nname\nsku\ncategory_id FK]
        Category[Category\ncategory_id PK\nname\nparent_id FK]
        Address[Address\naddress_id PK\nuser_id FK\nstreet\ncity\ntype]

        User -->|1:N| Order
        User -->|1:N| Address
        Order -->|1:N| OrderItem
        OrderItem -->|N:1| Product
        Product -->|N:1| Category
        Category -->|self-referential| Category
    end
```

## Normalization Progression

```mermaid
graph LR
    subgraph UNF[Unnormalized Form]
        U[order_id\ncustomer_name\ncustomer_email\nitems: id name qty\ncustomer_city]
        style U fill:#fee2e2,stroke:#dc2626
    end

    subgraph OneNF[1NF - Atomic Values]
        O[order_id\ncustomer_name\ncustomer_email\nitem_id name qty\n- no repeating groups]
        style O fill:#fef3c7,stroke:#d97706
    end

    subgraph TwoNF[2NF - No Partial Dependencies]
        T[Separate Orders and Items tables\n- item name not dependent\n  on order_id alone]
        style T fill:#dbeafe,stroke:#2563eb
    end

    subgraph ThreeNF[3NF - No Transitive Dependencies]
        TH[Separate Customers table\n- customer_city depends on\n  customer not on order]
        style TH fill:#dcfce7,stroke:#16a34a
    end

    UNF --> OneNF --> TwoNF --> ThreeNF
```

## Denormalization Patterns

```mermaid
graph TD
    subgraph Normalized[Normalized - Multiple Queries]
        NQ1[SELECT order FROM orders WHERE id=X]
        NQ2[SELECT user FROM users WHERE id=order.user_id]
        NQ3[SELECT items FROM order_items WHERE order_id=X]
        NQ4[SELECT product FROM products WHERE id=item.product_id]
        NQ1 --> NQ2 & NQ3 --> NQ4
        style NQ1 fill:#fee2e2,stroke:#dc2626
    end

    subgraph Denormalized[Denormalized - Single Query]
        DQ[order_detail view\norder_id user_name user_email\nitems array with product names\nSingle read, pre-joined]
        style DQ fill:#dcfce7,stroke:#16a34a
    end

    subgraph Patterns[Denormalization Patterns]
        Embed[Document embedding\nstore user in order doc]
        Cache[Materialized views\npre-computed aggregates]
        Replicate[Data replication\ncopy fields from related tables]
    end
```

## Polyglot Persistence

```mermaid
graph TD
    Application[Application Services]

    Application -->|User profiles| Postgres[(PostgreSQL\nRelational\nUser data, orders)]
    Application -->|Session data| Redis[(Redis\nKey-Value\nSessions, caches)]
    Application -->|Product catalog| MongoDB[(MongoDB\nDocument\nProduct info)]
    Application -->|Search| Elasticsearch[(Elasticsearch\nFull-text search\nProduct search)]
    Application -->|Recommendations| Neo4j[(Neo4j\nGraph\nUser-product relationships)]
    Application -->|Metrics| InfluxDB[(InfluxDB\nTime-series\nApplication telemetry)]
    Application -->|Vectors| Pinecone[(Pinecone\nVector\nSemantic search)]

    style Application fill:#dbeafe,stroke:#2563eb,stroke-width:2px
```

## Key Concepts

- **Entity-Relationship (ER) Modeling**: A conceptual modeling technique that identifies entities (objects), their attributes, and the relationships between them. ER models are implementation-independent — they describe the domain, not the storage. Used as the basis for relational schema design.

- **First Normal Form (1NF)**: All attributes are atomic (no arrays, no multi-valued attributes), and every row is uniquely identifiable by a primary key. A table with an "items" text column containing comma-separated values is not in 1NF.

- **Second Normal Form (2NF)**: Must be in 1NF, and every non-key attribute must be fully dependent on the entire primary key (no partial dependencies). Relevant only for composite primary keys. Example: if an OrderItem table has key (order_id, product_id), the product name must not live there — it depends only on product_id.

- **Third Normal Form (3NF)**: Must be in 2NF, and no non-key attribute should depend on another non-key attribute (no transitive dependencies). If the orders table has customer_city, and customer_city depends on customer_id (which is not the PK), that's a transitive dependency — customer_city belongs in a customers table.

- **Denormalization**: Intentionally introducing redundancy to optimize read performance. Trade storage space and write complexity for faster reads. Common strategies: document embedding (store related data in one document), materialized views (pre-computed query results), and summary tables (pre-aggregated analytics).

- **Polyglot Persistence**: Using multiple different database technologies in the same application, each chosen for its strengths relative to a specific use case. An e-commerce system might use Postgres for orders, Redis for sessions, Elasticsearch for search, and a graph database for recommendations.

- **Schema Versioning**: Managing evolution of data schemas over time without breaking running applications. Strategies include: expand/contract migrations (add new columns before removing old ones), backward-compatible changes, and blue-green schema migrations.

## Trade-offs

| Approach | Benefit | Cost |
|----------|---------|------|
| High normalization | No redundancy, easy writes | Many JOINs for reads |
| Denormalization | Fast reads, fewer queries | Write complexity, consistency risk |
| Polyglot persistence | Best tool per use case | Operational complexity, data sync |
| Single model | Operational simplicity | Performance compromises |

## When to Use

- **High normalization**: OLTP systems with frequent updates and complex queries — normalize first, denormalize selectively based on measured query costs
- **Denormalization**: Read-heavy APIs, analytics, when the same data is read from many places in the same form
- **Polyglot persistence**: Large systems where different data types have genuinely different access patterns — justify each additional database with a clear need
