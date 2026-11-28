"""Development prompt replacements for the ten v1 rows excluded at context 384."""

RANGE_SUM = (
    "n q, 整数列A[n], q組l r。A[l..r]の和を各行に出力。"
    "1<=n<=30,1<=q<=20,|A[i]|<=100,0<=l<=r<n。"
)
INTERVALS = "nと[l,r)をn組。和集合長を出力。0<=n<=30,-100<=l<=r<=100。"
EDIT_DISTANCE = "2語 [a-z]{1,20} の編集距離を出力。挿入/削除/置換=1。"
GRAPH_JA = "無向 n m (u v)*m a b; V=[0,n),n=1..20,m=0..190; "
GRAPH_EN = "n m (u v)*m a b; undirected, V=0..n-1,n=1..20,m=0..190. Print min hops a to b or -1. "

COMPACT_TASKS = {
    "range_sum_queries-prefix-ja": RANGE_SUM + "累積和。",
    "range_sum_queries-direct-ja": RANGE_SUM + "直接加算。",
    "interval_union_length-coverage-ja": INTERVALS + "単位区間の被覆をboolean配列に記録。",
    "interval_union_length-sort_merge-ja": INTERVALS + "ソート併合。",
    "edit_distance-matrix-ja": EDIT_DISTANCE + "2次元DP。",
    "edit_distance-rolling-ja": EDIT_DISTANCE + "行交替DP。",
    "unweighted_graph_distance-bfs-ja": GRAPH_JA + "キューBFS:a-b最短辺数(不可-1)",
    "unweighted_graph_distance-floyd-ja": GRAPH_JA + "Floyd:a-b距離(不可-1)",
    "unweighted_graph_distance-bfs-en": GRAPH_EN + "Queue BFS.",
    "unweighted_graph_distance-floyd-en": GRAPH_EN + "Floyd-Warshall.",
}
