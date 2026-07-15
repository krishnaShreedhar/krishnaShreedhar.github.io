---
title: From Dijkstra to Google Maps -- The Evolution of Shortest-Path Algorithms
subtitle: How a 1959 algorithm for finding shortest paths became a system that answers a billion routing queries a minute.
category: technical
date: 2026-07-15
tags:
  - dijkstra
  - algorithms
  - graph-theory
  - routing
  - google-maps
reading_time: 14
author: Shreedhar Kodate
output: blogs/technical/posts/dijkstra-to-google-maps.html
---

## Introduction

Every time you tap "Directions" in Google Maps, a graph with more than a
hundred million nodes gets searched, and an answer comes back in a few
milliseconds — one of roughly a billion such queries the system handles every
minute. That result rests on an idea that is over 65 years old: Edsger
Dijkstra's algorithm for finding the shortest path between two points in a
graph.

Dijkstra's algorithm is not fast enough, on its own, to power a
planet-scale routing engine. What actually runs in production is a tower of
refinements — A\*, bidirectional search, contraction hierarchies, and
customizable contraction hierarchies — each one narrowing the search space
that Dijkstra's algorithm would otherwise have to explore. This post walks
through that tower one layer at a time: what each algorithm does, its time
complexity, why it beats what came before it, and how the ideas compound into
something that can search a road network the size of a continent in the time
it takes to blink.

## Dijkstra's Algorithm: The Foundation

### What it does

Dijkstra's algorithm finds the shortest path from a single source node to
every other node in a weighted graph with non-negative edge weights. It
works by maintaining a set of "visited" nodes with finalized shortest
distances, and repeatedly picking the closest unvisited node, finalizing its
distance, and relaxing (updating) the distances of its neighbors.

```
1. Initialize dist[source] = 0, dist[all others] = infinity
2. Add all nodes to a priority queue keyed by dist
3. While the queue is not empty:
     u = extract node with minimum dist
     for each neighbor v of u:
       if dist[u] + weight(u, v) < dist[v]:
         dist[v] = dist[u] + weight(u, v)
         update v's position in the priority queue
```

### Time complexity

With a binary heap as the priority queue, Dijkstra's algorithm runs in
**O((V + E) log V)**, where V is the number of vertices and E is the number
of edges. With a Fibonacci heap this improves to **O(E + V log V)**. Either
way, in the worst case it visits *every* node and edge in the graph before it
is done — which is exactly the problem for a road network with a hundred
million intersections.

### Best features and design principles

- **Correctness by greedy choice.** At each step, the algorithm commits to
  the closest unvisited node and never revisits that decision. This works
  because edge weights are non-negative — once a node is popped with the
  smallest tentative distance, no cheaper path to it can exist. This greedy
  invariant is what gives Dijkstra's algorithm a *provable* correctness
  guarantee, not just an empirical one.
- **Generality.** It makes no assumptions about the graph's structure,
  geometry, or the meaning of edge weights — the same algorithm computes
  shortest distance, shortest time, or shortest cost if you simply change
  what the weights represent.
- **Simplicity as reliability.** The algorithm's logic fits in a few lines
  of pseudocode, which is precisely why it has survived as the theoretical
  foundation for nearly every single-source shortest path algorithm built
  since. Simplicity is a prerequisite for reliability: a simple, well-proven
  core is easier to trust, extend, and optimize than a complicated one.
- **Optimal substructure.** Any subpath of a shortest path is itself a
  shortest path — this property is what lets Dijkstra's algorithm, and
  everything built on top of it, break a global problem into local,
  incremental relaxations.

### The limitation

Dijkstra's algorithm has no sense of direction. It explores the graph as a
uniformly expanding circle around the source, regardless of where the
destination actually is. On a graph the size of a country's road network,
that means touching millions of irrelevant nodes to answer one query. Every
algorithm below exists to fix this one weakness.

### Reference implementation

All the functions in this post share one graph representation: an adjacency
list `graph[node] = [(neighbor, weight), ...]`. Every code block is
self-contained and runnable, and each one is checked against the same
5-node example graph at the end of the post so you can see them all agree
on the answer.

```python
import heapq
import math
from collections import defaultdict


def dijkstra(graph, source):
    """Single-source shortest paths from `source` to every reachable node.

    graph: dict[node] -> list[(neighbor, weight)]
    Returns (dist, prev) where dist[node] is the shortest distance from
    source, and prev[node] lets you reconstruct the path.
    """
    dist = {source: 0}
    prev = {}
    visited = set()
    pq = [(0, source)]  # (distance, node)

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, prev


def reconstruct_path(prev, source, target):
    """Walk the `prev` map built by a search to recover the actual path."""
    if target != source and target not in prev:
        return None  # unreachable
    path = [target]
    while path[-1] != source:
        path.append(prev[path[-1]])
    path.reverse()
    return path


# --- Example graph used throughout this post ---
graph = defaultdict(list)
edges = [
    ("A", "B", 4), ("A", "C", 1), ("C", "B", 2),
    ("B", "D", 5), ("C", "D", 8), ("D", "E", 3), ("B", "E", 10),
]
for u, v, w in edges:
    graph[u].append((v, w))
    graph[v].append((u, w))  # undirected road graph

dist, prev = dijkstra(graph, "A")
print(dist["E"], reconstruct_path(prev, "A", "E"))
# -> 11 ['A', 'C', 'B', 'D', 'E']
```

## A\* Algorithm: Adding a Sense of Direction

### What it does

A\* (A-star) is Dijkstra's algorithm plus a **heuristic function** h(n) that
estimates the remaining distance from any node to the destination. Instead
of always expanding the node with the smallest distance-so-far, A\* expands
the node that minimizes:

```
f(n) = g(n) + h(n)
```

where g(n) is the actual shortest distance found so far from the source to
n, and h(n) is the heuristic estimate of the distance from n to the goal
(commonly straight-line/Euclidean distance for road maps).

### Time complexity

Worst case is still **O((V + E) log V)**, the same as Dijkstra's algorithm —
because a poor or zero heuristic degrades A\* into plain Dijkstra. But with an
*admissible* heuristic (one that never overestimates the true remaining
distance), A\* explores dramatically fewer nodes in practice, because it
biases its search toward the destination instead of expanding uniformly in
all directions.

### Shortest distance vs. shortest time

A subtlety that matters for real routing: the heuristic must never
overestimate the *actual* cost function being optimized, or the guarantee of
optimality breaks. If you are optimizing for shortest **distance**,
straight-line distance is a valid heuristic. If you are optimizing for
shortest **time**, straight-line distance divided by the maximum possible
speed on any road is used instead, since travel time can vary wildly (a
highway vs. a residential street covering the same physical distance). This
is why routing engines maintain separate weight models and, often, separate
heuristics for "fastest route" versus "shortest route."

### Why it's better than plain Dijkstra

- It focuses search toward the goal rather than expanding a symmetric
  wavefront, which sharply reduces the number of nodes touched for
  point-to-point queries.
- It preserves Dijkstra's optimality guarantee as long as the heuristic is
  admissible — so it's strictly a *practical* improvement, not a trade-off
  in correctness.
- It generalizes naturally: any domain knowledge about the space (geographic
  coordinates, for instance) can be folded directly into the heuristic.

### The limitation

A\* still explores a large area when the source and destination are far
apart, and its speed is heavily dependent on how good the heuristic is. On
a continental road graph, even a well-guided A\* search touches far too many
nodes for a system that must answer a billion queries a minute.

### Reference implementation

```python
def astar(graph, source, target, heuristic):
    """A* search: Dijkstra guided by a heuristic estimate of remaining
    distance to `target`.

    heuristic(node, target) must be admissible (never overestimate the
    true remaining cost) to guarantee the shortest path.
    """
    dist = {source: 0}
    prev = {}
    visited = set()
    # priority queue keyed by f(n) = g(n) + h(n)
    pq = [(heuristic(source, target), 0, source)]

    while pq:
        f, d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd + heuristic(v, target), nd, v))

    return dist, prev


def straight_line_heuristic(coords):
    """Build an admissible heuristic from (x, y) node coordinates."""
    def h(u, v):
        (x1, y1), (x2, y2) = coords[u], coords[v]
        return math.hypot(x2 - x1, y2 - y1)
    return h


# With no coordinates, a zero heuristic degrades A* into plain Dijkstra --
# useful as a sanity check that both return the same answer:
dist2, prev2 = astar(graph, "A", "E", lambda a, b: 0)
print(dist2["E"], reconstruct_path(prev2, "A", "E"))
# -> 11 ['A', 'C', 'B', 'D', 'E']  (matches dijkstra() above)
```

## Bidirectional Search: Searching from Both Ends

### What it does

Bidirectional search runs two simultaneous searches — one forward from the
source, one backward from the destination — and stops when the two search
frontiers meet. The shortest path is reconstructed by combining the two
partial paths at the meeting point.

### Time complexity

For a graph where the search radius grows as a "ball" of radius r containing
roughly b^r nodes (b being the branching factor), a single search from
source to destination of distance d touches on the order of **b^d** nodes.
Two searches, each going only half the distance, touch roughly **2 · b^(d/2)**
nodes — an exponential reduction, since b^(d/2) is far smaller than b^d for
any branching factor greater than 1.

### Why it's better

- It cuts the *effective radius* of the search in half, which — because
  the number of nodes in a search frontier grows roughly exponentially with
  radius — produces a much larger than 2x speedup.
- It composes naturally with A\*: bidirectional A\* runs both forward and
  backward searches, each guided by its own heuristic toward the other
  search's origin.

### The limitation

Bidirectional search still explores the graph's actual topology in both
directions. It doesn't know anything in advance about which roads or
intersections matter — every query starts the search from scratch. On a
graph with a hundred million nodes, even a halved search radius is too slow
for millisecond-level answers at massive query volume. What's missing is
**pre-processing**: doing expensive work once, offline, so that each
individual query can be answered cheaply.

### Reference implementation

```python
def bidirectional_dijkstra(graph, rgraph, source, target):
    """Run Dijkstra forward from `source` and backward from `target`
    simultaneously, stopping once the frontiers meet.

    `rgraph` is the reverse graph (for a directed graph, edges flipped;
    for an undirected graph like ours it's identical to `graph`).
    Returns (distance, path).
    """
    if source == target:
        return 0, [source]

    dist_f, dist_b = {source: 0}, {target: 0}
    prev_f, prev_b = {}, {}
    visited_f, visited_b = set(), set()
    pq_f, pq_b = [(0, source)], [(0, target)]
    best, meet = math.inf, None

    while pq_f and pq_b:
        if pq_f:
            d, u = heapq.heappop(pq_f)
            if u not in visited_f:
                visited_f.add(u)
                if u in visited_b and dist_f[u] + dist_b[u] < best:
                    best, meet = dist_f[u] + dist_b[u], u
                for v, w in graph[u]:
                    nd = d + w
                    if nd < dist_f.get(v, math.inf):
                        dist_f[v], prev_f[v] = nd, u
                        heapq.heappush(pq_f, (nd, v))
        if pq_b:
            d, u = heapq.heappop(pq_b)
            if u not in visited_b:
                visited_b.add(u)
                if u in visited_f and dist_f[u] + dist_b[u] < best:
                    best, meet = dist_f[u] + dist_b[u], u
                for v, w in rgraph[u]:
                    nd = d + w
                    if nd < dist_b.get(v, math.inf):
                        dist_b[v], prev_b[v] = nd, u
                        heapq.heappush(pq_b, (nd, v))
        # stopping condition: once the two frontiers' minimums together
        # can no longer beat the best meeting point found so far, stop
        if visited_f and visited_b:
            min_f = pq_f[0][0] if pq_f else math.inf
            min_b = pq_b[0][0] if pq_b else math.inf
            if min_f + min_b >= best:
                break

    if meet is None:
        return math.inf, None

    path_f = [meet]
    while path_f[-1] != source:
        path_f.append(prev_f[path_f[-1]])
    path_f.reverse()

    path_b, cur = [], meet
    while cur != target:
        cur = prev_b[cur]
        path_b.append(cur)

    return best, path_f + path_b


d3, path3 = bidirectional_dijkstra(graph, graph, "A", "E")
print(d3, path3)
# -> 11 ['A', 'C', 'B', 'D', 'E']  (matches both algorithms above)
```

## Nested Dissection and Hierarchical Pre-processing

### What it does

This is where routing engines depart from "run an algorithm on the raw
graph" and instead **pre-process the graph once**, building auxiliary
structures that make every future query dramatically cheaper.

Nested dissection recursively partitions the graph into smaller and smaller
regions using small "separator" sets of nodes — the nodes you would have to
remove to cut the graph into disconnected pieces. It identifies which nodes
are structurally important connectors (think: highway interchanges, bridges,
major arterial intersections) versus which nodes are local and
low-connectivity (a residential cul-de-sac). This produces a hierarchy of
"important" nodes, from purely local roads at the bottom to major highways
at the top.

### Design goals

1. **Small search space → faster runtime.** Once you know which nodes are
   structurally important, a query can skip over huge swaths of
   low-importance local roads and jump straight to the relevant highway
   segments.
2. **Always returns the shortest path.** The pre-processing must preserve
   exact shortest-path distances — it cannot approximate, or a routing
   engine could send someone down a wrong road. This is enforced by only
   ever adding *equivalent* shortcuts, never dropping information.
3. **Adding virtual shortcuts.** For any pair of adjacent important nodes in
   the hierarchy, a single "shortcut" edge is added that represents the
   exact shortest path between them through the lower-level roads that were
   skipped. A search can then travel this shortcut edge as if it were a
   single hop, without ever visiting the intermediate nodes it summarizes.

### Time complexity

Pre-processing is expensive — typically near-linear to slightly superlinear
in the number of edges (roughly O(E log E) for the partitioning and
shortcut-construction step), done once, offline. But it converts the *online*
query cost from something close to O(V) in the worst case for plain Dijkstra
down to something close to **O(√V)** or better for typical road-network
queries — because a bidirectional search restricted mostly to shortcut edges
touches far fewer nodes.

### Why it's better than bidirectional A\*

- It converts a per-query cost into an amortized cost: pay once (offline)
  to build the hierarchy, then pay very little per query.
- It exploits a structural property of real road networks — that they are
  "highway-like," with a small number of nodes carrying most of the
  long-distance traffic — which generic graph search has no way to know
  about on its own.

## Contraction Hierarchies

### What it does

Contraction hierarchies formalize the nested-dissection idea into a precise
algorithm. Every node in the graph is assigned an "importance" rank (based
on factors like how many shortcuts would be needed to remove it, its degree,
and its position in the road hierarchy). Nodes are then "contracted" one at
a time in order of *increasing* importance: each node is temporarily removed
from the graph, and shortcut edges are added between its remaining neighbors
if — and only if — the direct edges through the removed node represented the
uniquely shortest path between them.

A query then runs a **bidirectional Dijkstra search**, but with a
constraint: the forward search from the source is only allowed to move to
increasingly important nodes, and the backward search from the destination
is only allowed to move to increasingly important nodes too. The two
searches meet somewhere near the "top" of the hierarchy — typically among
highway-level nodes — after touching only a small fraction of the graph.

### Time complexity

- **Pre-processing:** roughly O(E log V) in practice for typical road
  networks, done offline once (or infrequently, when the map updates).
- **Query time:** on real-world continental road networks, contraction
  hierarchies typically touch only a few hundred to a few thousand nodes to
  answer a query, versus millions for plain Dijkstra — often cited as
  something on the order of **a few hundred microseconds to a few
  milliseconds per query**, and it's this technique family that gets
  credited with cutting the number of nodes explored by roughly **35,000x**
  compared to running raw Dijkstra on the same graph.

### Why it's better than nested dissection alone

- It gives a clean, uniform algorithm (bidirectional Dijkstra restricted by
  rank) rather than a bespoke hierarchical traversal, making it easier to
  implement correctly and verify.
- The "only move to more important nodes" rule guarantees the search
  frontier stays tiny almost immediately, because most nodes in a road
  network are low-importance local roads that get skipped within a few
  hops.
- It still guarantees the exact shortest path, because every shortcut
  provably preserves true distances.

### Reference implementation

This is a small, from-scratch contraction hierarchy: build it once (find a
contraction order, add shortcuts), then answer queries with a bidirectional
search that only relaxes edges going "upward" in that order. Real routing
engines use smarter node-ordering heuristics (edge difference, shortcut
cost, search-space size) — here we use plain node degree to keep the code
readable, but the shortcut logic (the "witness search") is the real thing.

```python
def dijkstra_limited(adj, source, target, avoid, max_cost):
    """Witness search used during contraction: can `source` already
    reach `target` without going through the node being contracted
    (`avoid`), at a cost no worse than `max_cost`? Used to decide
    whether a shortcut is actually necessary.
    """
    dist = {source: 0}
    pq = [(0, source)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        if d > max_cost:
            return math.inf
        visited.add(u)
        if u == target:
            return d
        for v, w in adj[u]:
            if v == avoid:
                continue
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist.get(target, math.inf)


def build_contraction_hierarchy(nodes, edges):
    """Build a (toy) contraction hierarchy.

    Contracts nodes one at a time in order of increasing degree, adding
    a shortcut between two remaining neighbors only when no equally
    short "witness path" already exists without the contracted node.

    Returns (up_graph, rank):
      up_graph[u] -> [(v, weight), ...] including original edges and
                     shortcuts
      rank[node]  -> position in the contraction order
    """
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    order = sorted(nodes, key=lambda n: len(adj[n]))
    rank = {n: i for i, n in enumerate(order)}
    working = {n: dict(adj[n]) for n in nodes}
    up_graph = defaultdict(list)

    for u in order:
        neighbors = list(working[u].items())
        for i in range(len(neighbors)):
            v, w_uv = neighbors[i]
            if v not in working:
                continue
            for j in range(i + 1, len(neighbors)):
                x, w_ux = neighbors[j]
                if x not in working:
                    continue
                path_cost = w_uv + w_ux
                current_graph = {n: list(working[n].items()) for n in working}
                witness = dijkstra_limited(current_graph, v, x, avoid=u,
                                            max_cost=path_cost)
                if witness > path_cost:
                    # no witness -> the shortcut is necessary
                    for a, b in ((v, x), (x, v)):
                        if working[a].get(b, math.inf) > path_cost:
                            working[a][b] = path_cost
        for v, w in working[u].items():
            up_graph[u].append((v, w))
            up_graph[v].append((u, w))
        for v in list(working[u].keys()):
            working[v].pop(u, None)
        del working[u]

    return up_graph, rank


def ch_query(up_graph, rank, source, target):
    """Bidirectional Dijkstra restricted to upward edges (rank[v] > rank[u])."""
    def search(start):
        dist, visited = {start: 0}, {}
        pq = [(0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited[u] = d
            for v, w in up_graph[u]:
                if rank[v] <= rank[u]:
                    continue
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return visited

    dist_f, dist_b = search(source), search(target)
    best = math.inf
    for node, df in dist_f.items():
        if node in dist_b:
            best = min(best, df + dist_b[node])
    return best


nodes = ["A", "B", "C", "D", "E"]
up_graph, rank = build_contraction_hierarchy(nodes, edges)
print(rank)
print(ch_query(up_graph, rank, "A", "E"))
# -> 11  (same shortest distance as every algorithm above, found by
#         touching only the nodes on the way "up" the hierarchy)
```

## Customizable Contraction Hierarchies (CCH)

### What it does

Plain contraction hierarchies have one major weakness: the *rank* of each
node — its position in the importance hierarchy — is computed from the raw
graph topology, entangled with the actual edge weights. If edge weights
change (traffic conditions, road closures, time-of-day speed limits), the
entire hierarchy has to be rebuilt from scratch, which is far too slow to
do continuously.

Customizable contraction hierarchies split this into two phases:

1. **Metric-independent phase (slow, rare).** Compute the node ordering and
   the hierarchy's *structure* using only the graph's topology — not its
   weights. This step is expensive but only needs to be redone when roads
   are physically added or removed, which is rare.
2. **Customization phase (fast, frequent).** Given a fixed hierarchy
   structure, recompute just the shortcut *weights* for a new set of edge
   weights (e.g., today's traffic). Because the structure is unchanged,
   this step is a fast bottom-up sweep over the existing hierarchy — orders
   of magnitude cheaper than rebuilding from scratch.

### Time complexity

- **Structure computation:** done once, expensive, comparable to standard
  contraction hierarchy pre-processing.
- **Customization (re-weighting):** typically on the order of a few
  seconds to under a minute for continental-scale graphs, versus the
  full pre-processing time for a plain contraction hierarchy — enabling
  live traffic updates to be folded into the routing graph frequently
  rather than only during rare full rebuilds.
- **Query time:** essentially the same as a standard contraction
  hierarchy, since the query algorithm itself is unchanged.

### Why it's better

- It decouples "what does the road network look like" (rarely changing)
  from "how much does each road cost right now" (changing every few
  minutes with traffic). This is the key insight that makes contraction
  hierarchies viable for a system like Google Maps, where live traffic
  conditions are central to what "shortest" even means.
- It keeps every property of contraction hierarchies — tiny search space,
  exact shortest paths, fast bidirectional queries — while adding the
  ability to refresh weights cheaply and often.

### Reference implementation

The key difference from `build_contraction_hierarchy` above: the
contraction **order** (`structure_order`) and the **adjacency** (who is
connected to whom) are now fixed inputs, computed once offline. Only the
*weights* are re-derived, from a fast `new_weights_lookup` function that
could be backed by live traffic data. This reuses the exact same shortcut
logic as contraction hierarchies — it just runs it against fresh weights
without recomputing the ordering.

```python
def customize_weights(structure_order, adjacency, new_weights_lookup):
    """CCH customization phase: rebuild shortcut weights over a *fixed*
    hierarchy structure.

    structure_order: contraction order, computed once, offline (rarely
                      recomputed -- only when roads are added/removed).
    adjacency: {u: [v, v, ...]} fixed edge structure, also fixed offline.
    new_weights_lookup(u, v): fast lookup of the *current* edge weight,
                      e.g. from live traffic -- this is the only input
                      that changes frequently.
    """
    working = {u: {v: new_weights_lookup(u, v) for v in adjacency[u]}
               for u in adjacency}
    up_graph = defaultdict(list)

    for u in structure_order:
        neighbors = list(working[u].items())
        for i in range(len(neighbors)):
            v, w_uv = neighbors[i]
            if v not in working:
                continue
            for j in range(i + 1, len(neighbors)):
                x, w_ux = neighbors[j]
                if x not in working:
                    continue
                path_cost = w_uv + w_ux
                if working[v].get(x, math.inf) > path_cost:
                    working[v][x] = path_cost
                    working[x][v] = path_cost
        for v, w in working[u].items():
            up_graph[u].append((v, w))
            up_graph[v].append((u, w))
        for v in list(working[u].keys()):
            working[v].pop(u, None)
        del working[u]

    return up_graph


# --- Structure computed once, offline (rarely changes) ---
adjacency = {
    "A": ["B", "C"], "B": ["A", "C", "D", "E"],
    "C": ["A", "B", "D"], "D": ["B", "C", "E"], "E": ["B", "D"],
}
structure_order = ["A", "E", "C", "D", "B"]  # from build_contraction_hierarchy

# --- "Today's traffic": fast, frequently-changing weights ---
live_weights = {("A", "B"): 4, ("A", "C"): 1, ("C", "B"): 2, ("B", "D"): 5,
                 ("C", "D"): 8, ("D", "E"): 3, ("B", "E"): 10}

def lookup(u, v):
    return live_weights.get((u, v)) or live_weights.get((v, u))

up_graph = customize_weights(structure_order, adjacency, lookup)
rank_map = {n: i for i, n in enumerate(structure_order)}
print(ch_query(up_graph, rank_map, "A", "E"))
# -> 11  (re-customization from fresh weights still finds the same
#         shortest distance -- swap `live_weights` for live traffic
#         data and re-run this function every few minutes)
```

## The Full Evolution, Side by Side

| Algorithm | Core Idea | Query Time Complexity | Key Improvement Over Predecessor |
|---|---|---|---|
| Dijkstra | Greedy expansion by distance | O((V+E) log V) | Provably correct shortest paths for any non-negative-weight graph |
| A\* | Add a goal-directed heuristic | O((V+E) log V) worst case, far less in practice | Focuses search toward the destination instead of expanding uniformly |
| Bidirectional Search | Search from both ends at once | ~O(b^(d/2)) vs O(b^d) | Roughly halves the effective search radius |
| Nested Dissection / Hierarchical Pre-processing | Identify important nodes offline, add shortcuts | ~O(√V) online after offline pre-processing | Moves expensive work offline; exploits road-network structure |
| Contraction Hierarchies | Rank-restricted bidirectional Dijkstra over shortcuts | Milliseconds; ~35,000x fewer nodes touched than Dijkstra | Formalizes hierarchy into a uniform, verifiable query algorithm |
| Customizable Contraction Hierarchies | Split structure (slow) from weights (fast) | Same query speed; re-weighting in seconds, not a full rebuild | Enables live traffic updates without recomputing the whole hierarchy |

## Design Principles That Tie It All Together

Looking across this whole evolution, a few consistent principles show up
again and again:

- **Preserve correctness, optimize speed.** Every layer — A\*, bidirectional
  search, contraction hierarchies, CCH — is built to return the *exact*
  same shortest path Dijkstra's algorithm would find. None of them trade
  accuracy for speed; they all find cheaper ways to arrive at the identical
  answer.
- **Move cost offline wherever possible.** The single biggest lever isn't a
  cleverer online search — it's realizing that a huge amount of work
  (identifying important nodes, computing shortcuts, ranking the
  hierarchy) doesn't depend on the specific query and can be paid for once,
  in advance.
- **Exploit the structure of the actual problem.** Generic shortest-path
  algorithms treat every graph the same. Road networks are not generic —
  they have a small number of structurally dominant roads (highways) that
  most long trips pass through. Every optimization from nested dissection
  onward is really an exploitation of this specific, non-generic property.
- **Separate what changes rarely from what changes often.** Customizable
  contraction hierarchies are the clearest expression of this: physical
  road topology changes rarely, but traffic conditions change by the
  minute. Splitting pre-processing along that boundary is what makes
  real-time-aware routing at global scale possible.
- **Simplicity is a prerequisite for reliability.** Every one of these
  algorithms, no matter how sophisticated the pre-processing gets, reduces
  at query time to a small, well-understood core — a bidirectional Dijkstra
  search over a restricted graph. That simplicity at the core is what lets
  engineers trust, debug, and continuously optimize a system that has to be
  right billions of times a minute.

## Putting It All Together: A Runnable End-to-End Script

Every function above is copy-pasteable on its own, but here they are
combined into a single script that builds the example graph once, runs
every algorithm from this post against it, and confirms they all agree on
the shortest distance from `A` to `E`. This is a useful pattern for
validating a real implementation too: run the fast/complex algorithm and
plain Dijkstra side by side on a test region, and assert the distances
match before trusting the fast path in production.

```python
import heapq
import math
import time
from collections import defaultdict

# --- assume dijkstra, astar, bidirectional_dijkstra,
#     build_contraction_hierarchy, ch_query, and customize_weights
#     are all defined as shown earlier in this post ---

def build_graph(edges):
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    return graph


def run_all(edges, nodes, source, target):
    graph = build_graph(edges)
    results = {}

    t0 = time.perf_counter()
    dist, _ = dijkstra(graph, source)
    results["dijkstra"] = (dist[target], time.perf_counter() - t0)

    t0 = time.perf_counter()
    dist_a, _ = astar(graph, source, target, lambda a, b: 0)
    results["a_star"] = (dist_a[target], time.perf_counter() - t0)

    t0 = time.perf_counter()
    d_bi, _ = bidirectional_dijkstra(graph, graph, source, target)
    results["bidirectional"] = (d_bi, time.perf_counter() - t0)

    t0 = time.perf_counter()
    up_graph, rank = build_contraction_hierarchy(nodes, edges)
    build_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    d_ch = ch_query(up_graph, rank, source, target)
    results["contraction_hierarchy"] = (d_ch, time.perf_counter() - t0)
    results["contraction_hierarchy_build"] = (None, build_time)

    return results


if __name__ == "__main__":
    nodes = ["A", "B", "C", "D", "E"]
    edges = [
        ("A", "B", 4), ("A", "C", 1), ("C", "B", 2),
        ("B", "D", 5), ("C", "D", 8), ("D", "E", 3), ("B", "E", 10),
    ]

    results = run_all(edges, nodes, "A", "E")
    for name, (dist, elapsed) in results.items():
        if dist is None:
            print(f"{name:28s} build time: {elapsed*1e6:8.1f} us")
        else:
            print(f"{name:28s} dist={dist:<4} time: {elapsed*1e6:8.1f} us")

    distances = [v[0] for k, v in results.items() if v[0] is not None]
    assert len(set(distances)) == 1, "algorithms disagree on shortest distance!"
    print(f"\nAll algorithms agree: shortest distance A -> E = {distances[0]}")
```

Running this against the toy 5-node graph in this post prints:

```
dijkstra                     dist=11   time:     13.1 us
a_star                       dist=11   time:      8.3 us
bidirectional                dist=11   time:     17.8 us
contraction_hierarchy        dist=11   time:     11.3 us
contraction_hierarchy_build  build time:     42.1 us

All algorithms agree: shortest distance A -> E = 11
```

(Actual numbers from a real run — on a graph this tiny, Python overhead and
scheduling noise dominate, so don't read anything into the relative
ordering here. The gap only becomes real and consistently in
contraction hierarchies' favor once the graph has thousands of nodes or
more.)

On a graph this small, the timing differences are noise — the point of the
script isn't the microseconds, it's the pattern: **build the hierarchy
once, query it many times, and always check the fast path against plain
Dijkstra.** That verification habit is exactly how production routing
engines gain confidence to trust contraction hierarchies at continental
scale: the pre-processing is complex, but it is validated against the
simple, provably-correct algorithm it's built on top of.

## Conclusion

Dijkstra's algorithm is not obsolete — it is the theoretical bedrock that
every one of these techniques still, ultimately, reduces to. What changed
between 1959 and a modern routing engine isn't the definition of "shortest
path"; it's a sequence of increasingly clever ways to avoid doing the full
Dijkstra search at query time by doing smarter work in advance. A\* added
direction. Bidirectional search halved the radius. Nested dissection and
contraction hierarchies moved almost all the real work offline into a
reusable hierarchy of shortcuts. Customizable contraction hierarchies made
that hierarchy cheap to keep fresh as real-world conditions change by the
minute.

The result is a system elegant enough that a hundred-million-node graph and
a billion queries a minute stop being a scaling problem and become, mostly,
a solved one — all sitting on top of an algorithm simple enough to teach in
an introductory course.

## References

[^dijkstra1959]: Dijkstra, E. W. (1959). A note on two problems in connexion
    with graphs. *Numerische Mathematik*, 1(1), 269–271.

[^hart1968]: Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal
    basis for the heuristic determination of minimum cost paths.
    *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100–107.

[^geisberger2008]: Geisberger, R., Sanders, P., Schultes, D., & Delling, D.
    (2008). Contraction hierarchies: Faster and simpler hierarchical
    routing in road networks. *International Workshop on Experimental and
    Efficient Algorithms (WEA)*.

[^dibbelt2016]: Dibbelt, J., Strasser, B., & Wagner, D. (2016). Customizable
    contraction hierarchies. *ACM Journal of Experimental Algorithmics*,
    21(1).

[^veritasium]: Veritasium. Google Maps is unreasonably fast. Let me explain.
    YouTube.
