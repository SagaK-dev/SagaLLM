"""Small original algorithm curriculum prototype. No training or heldout role."""
from __future__ import annotations
import hashlib
import json
import random

HEADER="import java.util.*;\npublic class Main {\n public static void main(String[] args) {\n Scanner s=new Scanner(System.in);\n"
FOOTER="\n }\n}\n"
SPECS={
 "range_sum_queries":{
  "en":"Read n, q, n integers, then q pairs l r (0-based inclusive). Print each range sum on its own line. 1<=n<=30, 1<=q<=20; values -100..100; 0<=l<=r<n.",
  "ja":"n,q、n個の整数、続いてq組のl,rを読み取る。添字は0始まり、両端を含む。各区間和を1行ずつ出力する。nは1〜30、qは1〜20、値は-100〜100、0<=l<=r<n。",
  "variants":{
   "prefix":{"en":"Use prefix sums.","ja":"累積和を用いる。","body":"int n=s.nextInt(),q=s.nextInt(); long[] p=new long[n+1]; for(int i=0;i<n;i++)p[i+1]=p[i]+s.nextInt(); while(q-->0){int l=s.nextInt(),r=s.nextInt();System.out.println(p[r+1]-p[l]);}"},
   "direct":{"en":"Use direct summation for each query.","ja":"各区間を直接加算する。","body":"int n=s.nextInt(),q=s.nextInt();int[] a=new int[n];for(int i=0;i<n;i++)a[i]=s.nextInt();while(q-->0){int l=s.nextInt(),r=s.nextInt();long sum=0;for(int i=l;i<=r;i++)sum+=a[i];System.out.println(sum);}"}}
 },
 "interval_union_length":{
  "en":"Read n then n pairs l r describing half-open integer intervals [l,r). Print the total union length. 0<=n<=30, -100<=l<=r<=100.",
  "ja":"n、続いて半開区間[l,r)を表すn組の整数l,rを読む。区間の和集合の長さを出力する。nは0〜30、-100<=l<=r<=100。",
  "variants":{
   "sort_merge":{"en":"Sort and merge overlapping intervals.","ja":"ソートして区間を併合する。","body":"int n=s.nextInt();int[][] a=new int[n][2];for(int[] p:a){p[0]=s.nextInt();p[1]=s.nextInt();}Arrays.sort(a,Comparator.comparingInt(p->p[0]));int total=0,l=0,r=0;boolean started=false;for(int[] p:a){if(!started){l=p[0];r=p[1];started=true;}else if(p[0]<=r){r=Math.max(r,p[1]);}else{total+=r-l;l=p[0];r=p[1];}}if(started)total+=r-l;System.out.println(total);"},
   "coverage":{"en":"Mark covered unit intervals in a bounded boolean array.","ja":"有界なboolean配列で単位区間を記録する。","body":"int n=s.nextInt();boolean[] covered=new boolean[200];for(int i=0;i<n;i++){int l=s.nextInt(),r=s.nextInt();for(int j=l;j<r;j++)covered[j+100]=true;}int total=0;for(boolean b:covered)if(b)total++;System.out.println(total);"}}
 },
 "edit_distance":{
  "en":"Read two nonempty lowercase ASCII words of length 1..20. Print their Levenshtein distance (insertion, deletion, substitution each cost 1).",
  "ja":"長さ1〜20の空でないASCII小文字の単語を2つ読む。挿入・削除・置換のコストを各1とする編集距離を出力する。",
  "variants":{
   "matrix":{"en":"Use a two-dimensional dynamic programming table.","ja":"二次元の動的計画法を用いる。","body":"String a=s.next(),b=s.next();int m=a.length(),n=b.length();int[][] d=new int[m+1][n+1];for(int i=0;i<=m;i++)d[i][0]=i;for(int j=0;j<=n;j++)d[0][j]=j;for(int i=1;i<=m;i++)for(int j=1;j<=n;j++)d[i][j]=Math.min(Math.min(d[i-1][j]+1,d[i][j-1]+1),d[i-1][j-1]+(a.charAt(i-1)==b.charAt(j-1)?0:1));System.out.println(d[m][n]);"},
   "rolling":{"en":"Use rolling one-dimensional dynamic programming rows.","ja":"一次元配列を交互に更新する動的計画法を用いる。","body":"String a=s.next(),b=s.next();int[] prev=new int[b.length()+1];for(int j=0;j<prev.length;j++)prev[j]=j;for(int i=1;i<=a.length();i++){int[] cur=new int[prev.length];cur[0]=i;for(int j=1;j<cur.length;j++)cur[j]=Math.min(Math.min(prev[j]+1,cur[j-1]+1),prev[j-1]+(a.charAt(i-1)==b.charAt(j-1)?0:1));prev=cur;}System.out.println(prev[b.length()]);"}}
 },
 "unweighted_graph_distance":{
  "en":"Read n,m, then m undirected edges u v, then vertices a b. Print the shortest number of edges from a to b, or -1 if unreachable. 1<=n<=20, 0<=m<=190, vertices 0..n-1.",
  "ja":"n,m、無向辺u,vをm組、最後に頂点a,bを読む。aからbまでの最短辺数を出力し、到達不能なら-1を出力する。nは1〜20、mは0〜190、頂点は0〜n-1。",
  "variants":{
   "bfs":{"en":"Use breadth-first search with a queue.","ja":"キューによる幅優先探索を用いる。","body":"int n=s.nextInt(),m=s.nextInt();boolean[][] g=new boolean[n][n];for(int i=0;i<m;i++){int u=s.nextInt(),v=s.nextInt();g[u][v]=g[v][u]=true;}int a=s.nextInt(),b=s.nextInt();int[] d=new int[n];Arrays.fill(d,-1);d[a]=0;ArrayDeque<Integer> q=new ArrayDeque<>();q.add(a);while(!q.isEmpty()){int u=q.remove();for(int v=0;v<n;v++)if(g[u][v]&&d[v]<0){d[v]=d[u]+1;q.add(v);}}System.out.println(d[b]);"},
   "floyd":{"en":"Use the Floyd-Warshall all-pairs distance algorithm.","ja":"ワーシャルフロイド法を用いる。","body":"int n=s.nextInt(),m=s.nextInt();int[][] d=new int[n][n];for(int i=0;i<n;i++){Arrays.fill(d[i],1000);d[i][i]=0;}for(int i=0;i<m;i++){int u=s.nextInt(),v=s.nextInt();if(u!=v)d[u][v]=d[v][u]=1;}int a=s.nextInt(),b=s.nextInt();for(int k=0;k<n;k++)for(int i=0;i<n;i++)for(int j=0;j<n;j++)d[i][j]=Math.min(d[i][j],d[i][k]+d[k][j]);System.out.println(d[a][b]>=1000?-1:d[a][b]);"}}
 }
}

def rows():
    for family,spec in sorted(SPECS.items()):
        for variant,v in sorted(spec["variants"].items()):
            for lang in ("ja","en"):
                yield {"id":family+"-"+variant+"-"+lang,"family":family,"variant":variant,"language":lang,
                       "task":spec[lang]+" "+v[lang],"source":HEADER+v["body"]+FOOTER}

def cases(family):
    """Fixed edge cases + seeded random cases; development oracle inputs."""
    rng=random.Random(441)
    if family=="range_sum_queries":
        yield "1 2\n-9\n0 0\n0 0\n"
        yield "4 3\n1 -1 2 -2\n0 3\n1 2\n2 2\n"
        for _ in range(10):
            n=rng.randint(1,30);q=rng.randint(1,20);a=[rng.randint(-100,100) for _ in range(n)]
            pairs=[sorted((rng.randrange(n),rng.randrange(n))) for _ in range(q)]
            yield f"{n} {q}\n"+" ".join(map(str,a))+"\n"+"\n".join(f"{l} {r}" for l,r in pairs)+"\n"
    elif family=="interval_union_length":
        yield "0\n";yield "5\n-5 0\n0 3\n-2 2\n3 3\n9 10\n"
        for _ in range(10):
            a=[sorted((rng.randint(-100,100),rng.randint(-100,100))) for _ in range(rng.randint(0,30))]
            yield str(len(a))+"\n"+"".join(f"{l} {r}\n" for l,r in a)
    elif family=="edit_distance":
        yield "a a\n";yield "kitten sitting\n"
        for _ in range(10):
            yield " ".join("".join(rng.choice("abcde") for _ in range(rng.randint(1,20))) for j in range(2))+"\n"
    elif family=="unweighted_graph_distance":
        yield "1 0\n0 0\n";yield "3 1\n0 1\n0 2\n"
        for _ in range(10):
            n=rng.randint(2,20);edges=[(u,v) for u in range(n) for v in range(u+1,n) if rng.random()<.2]
            yield f"{n} {len(edges)}\n"+"".join(f"{u} {v}\n" for u,v in edges)+f"{rng.randrange(n)} {rng.randrange(n)}\n"
    else:raise ValueError("Unknown family")

def oracle(family,stdin):
    from functools import lru_cache
    if family=="edit_distance":
        a,b=stdin.split()
        @lru_cache(None)
        def dist(i,j):
            if i==len(a):return len(b)-j
            if j==len(b):return len(a)-i
            return min(1+dist(i+1,j),1+dist(i,j+1),(a[i]!=b[j])+dist(i+1,j+1))
        return str(dist(0,0))+"\n"
    nums=list(map(int,stdin.split()));n=nums[0]
    if family=="range_sum_queries":
        q=nums[1];a=nums[2:2+n];p=nums[2+n:]
        return "".join(str(sum(a[p[2*i]:p[2*i+1]+1]))+"\n" for i in range(q))
    if family=="interval_union_length":
        covered={x for i in range(n) for x in range(nums[1+2*i],nums[2+2*i])}
        return str(len(covered))+"\n"
    if family=="unweighted_graph_distance":
        m=nums[1];a,b=nums[-2:];edges=[tuple(nums[2+2*i:4+2*i]) for i in range(m)]
        reachable={a}
        for depth in range(n):
            if b in reachable:return str(depth)+"\n"
            reachable|={v for u,v in edges if u in reachable}|{u for u,v in edges if v in reachable}
        return "-1\n"
    raise ValueError("Unknown family")

def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
