"""Fixed, ordered author-side prompt proposals; not approved training data.

All options retain the original reference implementation. No automatic
truncation, tokenizer change, or fallback to ambiguous prior prompts.
"""
# Prefer the fuller expression; the second is an explicitly authored concise
# alternative, never an algorithmically truncated version.
RANGE = (
    "n q,整数A[n],q組l r。両端含むA[l..r]の和を各行出力。1<=n<=30,1<=q<=20,|A[i]|<=100,0<=l<=r<n。",
    "n q,整数A[n],q組l r。sum(A[i],l<=i<=r)を各行出力。1<=n<=30,1<=q<=20,|A[i]|<=100,0<=l<=r<n。",
)
INTERVAL = (
    "nと整数端点l rをn組。半開区間[l,r)の和集合長を出力。0<=n<=30,-100<=l<=r<=100。",
    "n,整数区間[l,r)をn組。和集合長を出力。0<=n<=30,-100<=l<=r<=100。",
)
EDIT = "2語 [a-z]{1,20} の編集距離を出力。挿入/削除/置換=1。"
GRAPH = (
    "n m,無向辺(u v)*m,a b。V=[0,n),n=1..20,m=0..190;",
    "n m (u v)*m a b;無向辺,V=[0,n),n=1..20,m=0..190;",
)
PROPOSALS = {
    "range_sum_queries-prefix-ja": tuple(p+"累積和。" for p in RANGE),
    "range_sum_queries-direct-ja": tuple(p+"直接加算。" for p in RANGE),
    "interval_union_length-coverage-ja": (
        INTERVAL[0]+"単位区間をboolean配列に記録。",
        INTERVAL[1]+"単位区間のboolean配列。",
    ),
    "interval_union_length-sort_merge-ja": tuple(p+"ソート併合。" for p in INTERVAL),
    "edit_distance-rolling-ja": (EDIT+"1次元DPで行更新。", EDIT+"1次元DP行交替。"),
    "unweighted_graph_distance-bfs-ja": (
        GRAPH[0]+"キューBFS:a-b最短辺数(不可-1)",
        GRAPH[1]+"queue BFS:a-b最短辺数(不可-1)",
    ),
    "unweighted_graph_distance-floyd-ja": (
        GRAPH[0]+"Floyd-Warshall:a-b最短辺数(不可-1)",
        GRAPH[1]+"Floyd:a-b最短辺数(不可-1)",
        "n m (u v)*m a b;無向,V=[0,n),n=1..20,m=0..190; Floyd:a-b最短辺数(不可-1)",
    ),
}
REVIEW_FOCUS = {
    "range_sum_queries-prefix-ja": "inclusive index, zero-based bounds, per-query stdout, prefix sums",
    "range_sum_queries-direct-ja": "inclusive index, zero-based bounds, per-query stdout, direct summation",
    "interval_union_length-coverage-ja": "integer endpoints, half-open intervals, boolean unit coverage",
    "interval_union_length-sort_merge-ja": "integer endpoints, half-open intervals, sort and merge",
    "edit_distance-rolling-ja": "ASCII domain, unit costs, rolling one-dimensional rows",
    "unweighted_graph_distance-bfs-ja": "undirected input order, minimum edge count, unreachable -1, queue BFS",
    "unweighted_graph_distance-floyd-ja": "undirected input order, minimum edge count, unreachable -1, Floyd-Warshall",
}
