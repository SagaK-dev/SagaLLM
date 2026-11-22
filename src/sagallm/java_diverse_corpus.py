"""Reproducible original Java 17 code-instruction curriculum, v2.

Every example is *synthetic, template-derived*. It is not a repository crawl,
not independent human-written code, and does not establish coding competence.
Whole task families are partitioned before generating any examples.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .java_ai import _model_user_message
from .java_coding_eval import BASE_SOURCE
from .java_large_corpus import (_case as old_case, _java, TRAIN_FAMILIES,
                                VALIDATION_FAMILIES, TEST_FAMILIES)

SCHEMA = 'sagallm-java-diverse-v2'
SEED = 20261122
NEW_TRAIN = (
    'count_divisors','perfect_square','reverse_digits','collatz_steps','sum_squares',
    'triangular_number','binomial_coefficient','hex_format','binary_format','bit_count',
    'reverse_words','word_count','remove_vowels','unique_character_count',
    'last_index','left_pad','title_case','numeric_string',
    'array_rotate_left','array_second_max','array_range','array_median',
    'array_unique_sorted','array_filter_negative','array_count_greater',
    'array_running_max','matrix_trace','matrix_transpose','matrix_row_sums',
    'map_frequency','list_filter_even','stream_even_sum',
)
NEW_VALIDATION = ('armstrong_number','missing_array_value','run_length_encode','set_union')
NEW_TEST = ('collatz_max','array_two_sum','longest_common_prefix','matrix_spiral')
SPLITS = {
    'train': (tuple(TRAIN_FAMILIES), 140, tuple(NEW_TRAIN), 280),
    'validation': (tuple(VALIDATION_FAMILIES), 18, tuple(NEW_VALIDATION), 30),
    'test': (tuple(TEST_FAMILIES), 18, tuple(NEW_TEST), 30),
}
# 58*140 + 32*280 = 17,080 train; 6*18+4*30=228 in each heldout split.
MAX_SPLIT_BYTES = 90_000_000


def _json(data: object) -> str:
    return json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def _digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def new_case(family: str, i: int) -> tuple[str,str,str]:
    """Pure original Java example. Variation changes task inputs, not just comments."""
    if family not in NEW_TRAIN+NEW_VALIDATION+NEW_TEST or i<0 or i>5000:
        raise ValueError('Invalid family or variant')
    a=31+7*i
    b=3+(11*i)%79
    n=3+i%10
    c=1+i%8
    array=[a,b,2+i%9,12+i%11,5+i%7]
    arr=', '.join(str(x) for x in array)
    s=f'JavaCode{i:04d}'
    words=f'alpha beta code{i:04d} java'
    # A bounded set of original exercises. Use the Java standard library only.
    if family=='count_divisors':
        jp,en=f'{a}の正の約数の個数を求めて',f'Count the positive divisors of {a}'
        body=f'int v={a},count=0;for(int k=1;k<=v;k++)if(v%k==0)count++;System.out.println(count);'
    elif family=='perfect_square':
        jp,en=f'{a}が平方数かどうかを判定して',f'Check whether {a} is a perfect square'
        body=f'int v={a},r=(int)Math.sqrt(v);System.out.println(r*r==v);'
    elif family=='reverse_digits':
        jp,en=f'{a}の十進数字を逆順にして',f'Reverse the decimal digits of {a}'
        body=f'int v={a},out=0;while(v>0){{out=10*out+v%10;v/=10;}}System.out.println(out);'
    elif family=='collatz_steps':
        jp,en=f'{17+i}からコラッツ操作で1になるまでの回数を表示',f'Count Collatz steps from {17+i} to 1'
        body=f'long v={17+i};int steps=0;while(v!=1){{v=v%2==0?v/2:3*v+1;steps++;}}System.out.println(steps);'
    elif family=='sum_squares':
        jp,en=f'1から{17+i}までの平方の和を求めて',f'Sum the squares from 1 to {17+i}'
        body=f'int total=0;for(int k=1;k<={17+i};k++)total+=k*k;System.out.println(total);'
    elif family=='triangular_number':
        jp,en=f'{a}番目の三角数を表示',f'Compute triangular number {a}'
        body=f'long v={a};System.out.println(v*(v+1)/2);'
    elif family=='binomial_coefficient':
        jp,en=f'{n+c}個から{c}個選ぶ組合せの数に{a}を掛けて',f'Multiply {n+c} choose {c} by {a}'
        body=f'long result=1;for(int k=1;k<={c};k++)result=result*({n+c}-k+1)/k;System.out.println(result*{a});'
    elif family=='hex_format':
        jp,en=f'{a}を16進数で表示',f'Print {a} in hexadecimal'
        body=f'System.out.println(Integer.toHexString({a}));'
    elif family=='binary_format':
        jp,en=f'{a}を2進数で表示',f'Print {a} in binary'
        body=f'System.out.println(Integer.toBinaryString({a}));'
    elif family=='bit_count':
        jp,en=f'{a}の2進表現に含まれる1の個数を求めて',f'Count set bits of {a}'
        body=f'System.out.println(Integer.bitCount({a}));'
    elif family=='reverse_words':
        jp,en=f'{words}の単語の順番を逆に表示',f'Reverse the word order of {words}'
        body=f'String[] p="{words}".split(" ");for(int k=p.length-1;k>=0;k--)System.out.println(p[k]);'
    elif family=='word_count':
        jp,en=f'文章{words}の単語数を数えて',f'Count words in {words}'
        body=f'String s="{words}";System.out.println(s.trim().split("\\\\s+").length);'
    elif family=='remove_vowels':
        jp,en=f'{s}から英語の母音を取り除いて',f'Remove English vowels from {s}'
        body=f'String s="{s}";System.out.println(s.replaceAll("(?i)[aeiou]", ""));'
    elif family=='unique_character_count':
        jp,en=f'{s}にある異なる文字数を出して',f'Count unique characters in {s}'
        body=f'String s="{s}";System.out.println(s.chars().distinct().count());'
    elif family=='last_index':
        jp,en=f'{s}の中で文字aが最後に現れる位置',f'Find last index of a in {s}'
        body=f'System.out.println("{s}".lastIndexOf("a"));'
    elif family=='left_pad':
        jp,en=f'{s}を幅{len(s)+c}まで左側を空白で埋めて',f'Pad {s} to width {len(s)+c}'
        body=f'String s="{s}";System.out.println(" ".repeat(Math.max(0,{len(s)+c}-s.length()))+s);'
    elif family=='title_case':
        jp,en=f'{words}の単語先頭を大文字にして',f'Title-case words in {words}'
        body=f'String[] parts="{words}".split(" ");for(int k=0;k<parts.length;k++)parts[k]=parts[k].substring(0,1).toUpperCase()+parts[k].substring(1);System.out.println(String.join(" ",parts));'
    elif family=='numeric_string':
        jp,en=f'文字列{a}が数字だけか判定して',f'Check whether string {a} has only digits'
        body=f'String s="{a}";System.out.println(s.matches("[0-9]+"));'
    elif family=='array_rotate_left':
        jp,en=f'配列[{arr}]を左へ1つ回転して',f'Rotate array [{arr}] left by one'
        body=f'int[] a={{{arr}}};int first=a[0];for(int k=1;k<a.length;k++)a[k-1]=a[k];a[a.length-1]=first;System.out.println(java.util.Arrays.toString(a));'
    elif family=='array_second_max':
        jp,en=f'配列[{arr}]の2番目に大きい値を求めて',f'Find the second largest distinct value of [{arr}]'
        body=f'int[] a={{{arr}}};java.util.Arrays.sort(a);int p=a.length-2;while(p>=0&&a[p]==a[a.length-1])p--;System.out.println(p<0?"none":Integer.toString(a[p]));'
    elif family=='array_range':
        jp,en=f'配列[{arr}]の最大値と最小値の差を求めて',f'Find range max minus min of [{arr}]'
        body=f'int[] a={{{arr}}};int lo=a[0],hi=a[0];for(int x:a){{lo=Math.min(lo,x);hi=Math.max(hi,x);}}System.out.println(hi-lo);'
    elif family=='array_median':
        jp,en=f'5要素配列[{arr}]の中央値を求めて',f'Find the median of five values [{arr}]'
        body=f'int[] a={{{arr}}};java.util.Arrays.sort(a);System.out.println(a[a.length/2]);'
    elif family=='array_unique_sorted':
        jp,en=f'配列[{arr}]を重複なしで昇順にして',f'Print unique sorted values of [{arr}]'
        body=f'int[] a={{{arr}}};java.util.Set<Integer> s=new java.util.TreeSet<>();for(int x:a)s.add(x);System.out.println(s);'
    elif family=='array_filter_negative':
        signed=[-a,b,-c,n,5+i%7];astr=', '.join(map(str,signed))
        jp,en=f'配列[{astr}]から負の値を取り除いて',f'Remove negative values from [{astr}]'
        body=f'int[] a={{{astr}}};for(int x:a)if(x>=0)System.out.println(x);'
    elif family=='array_count_greater':
        jp,en=f'配列[{arr}]で{b}より大きい要素数',f'Count values greater than {b} in [{arr}]'
        body=f'int[] a={{{arr}}};int count=0;for(int x:a)if(x>{b})count++;System.out.println(count);'
    elif family=='array_running_max':
        jp,en=f'配列[{arr}]の累積最大値を出して',f'Print running maxima of [{arr}]'
        body=f'int[] a={{{arr}}};int best=Integer.MIN_VALUE;for(int x:a){{best=Math.max(best,x);System.out.println(best);}}'
    elif family=='matrix_trace':
        jp,en=f'対角成分が{a}、{b}、{n}の行列のトレース',f'Compute trace of diagonal matrix {a}, {b}, {n}'
        body=f'int[][] a={{{{{a},0,0}},{{0,{b},0}},{{0,0,{n}}}}};int total=0;for(int k=0;k<3;k++)total+=a[k][k];System.out.println(total);'
    elif family=='matrix_transpose':
        jp,en=f'行列[{a},{b};{n},{c}]を転置して',f'Transpose 2x2 matrix [{a}, {b}; {n}, {c}]'
        body=f'int[][] m={{{{{a},{b}}},{{{n},{c}}}}};for(int col=0;col<2;col++)for(int row=0;row<2;row++)System.out.println(m[row][col]);'
    elif family=='matrix_row_sums':
        jp,en=f'2行の行列[{a},{b};{n},{c}]の行ごとの和',f'Print row sums of matrix [{a}, {b}; {n}, {c}]'
        body=f'int[][] m={{{{{a},{b}}},{{{n},{c}}}}};for(int[] row:m){{int total=0;for(int v:row)total+=v;System.out.println(total);}}'
    elif family=='map_frequency':
        jp,en=f'配列[{arr}]に含まれる{a}の個数をMapで数えて',f'Count {a} in [{arr}] using a Map'
        body=f'int[] a={{{arr}}};java.util.Map<Integer,Integer> m=new java.util.HashMap<>();for(int x:a)m.merge(x,1,Integer::sum);System.out.println(m.getOrDefault({a},0));'
    elif family=='list_filter_even':
        jp,en=f'整数リスト[{arr}]から偶数だけ表示',f'Print even elements from list [{arr}]'
        body=f'java.util.List<Integer> a=java.util.List.of({arr});for(int x:a)if(x%2==0)System.out.println(x);'
    elif family=='stream_even_sum':
        jp,en=f'配列[{arr}]の偶数の合計をStreamで',f'Sum even numbers in [{arr}] with a stream'
        body=f'int[] a={{{arr}}};System.out.println(java.util.Arrays.stream(a).filter(x->x%2==0).sum());'
    elif family=='armstrong_number':
        jp,en=f'{a}がアームストロング数か判定',f'Check whether {a} is an Armstrong number'
        body=f'int n={a};String s=Integer.toString(n);int sum=0;for(char d:s.toCharArray())sum+=(int)Math.pow(d-\'0\',s.length());System.out.println(sum==n);'
    elif family=='missing_array_value':
        jp,en=f'1から{n+1}で{c}が欠けた配列の欠損値',f'Find missing {c} from values 1..{n+1}'
        body=f'int missing={c},max={n+1};int sum=0;for(int x=1;x<=max;x++)if(x!=missing)sum+=x;System.out.println(max*(max+1)/2-sum);'
    elif family=='run_length_encode':
        jp,en=f'同じ文字が{c}回続く{chr(65+i%20)}を圧縮して',f'Run-length encode {chr(65+i%20)} repeated {c} times'
        body=f'String s="{chr(65+i%20)}".repeat({c});StringBuilder out=new StringBuilder();for(int k=0;k<s.length();){{int j=k;while(j<s.length()&&s.charAt(j)==s.charAt(k))j++;out.append(s.charAt(k)).append(j-k);k=j;}}System.out.println(out);'
    elif family=='set_union':
        jp,en=f'{{{a},{b}}}と{{{b},{n}}}を集合の和にして',f'Union sets {{{a},{b}}} and {{{b},{n}}}'
        body=f'java.util.Set<Integer> s=new java.util.TreeSet<>(java.util.List.of({a},{b}));s.addAll(java.util.List.of({b},{n}));System.out.println(s);'
    elif family=='collatz_max':
        jp,en=f'{17+i}から1になるまでのコラッツ数列の最大値',f'Maximum Collatz value starting at {17+i}'
        body=f'long x={17+i},best=x;while(x!=1){{x=x%2==0?x/2:3*x+1;best=Math.max(best,x);}}System.out.println(best);'
    elif family=='array_two_sum':
        jp,en=f'配列[{arr}]の中で合計が{a+b}になる2要素があるか',f'Whether two elements of [{arr}] sum to {a+b}'
        body=f'int[] a={{{arr}}};boolean ok=false;for(int p=0;p<a.length;p++)for(int q=p+1;q<a.length;q++)if(a[p]+a[q]=={a+b})ok=true;System.out.println(ok);'
    elif family=='longest_common_prefix':
        jp,en=f'{s}と{s}Suffixの最長共通接頭辞を表示',f'Find longest common prefix of {s} and {s}Suffix'
        body=f'String a="{s}",b="{s}Suffix";int k=0;while(k<a.length()&&k<b.length()&&a.charAt(k)==b.charAt(k))k++;System.out.println(a.substring(0,k));'
    elif family=='matrix_spiral':
        jp,en=f'2x2行列[{a},{b};{n},{c}]を時計回りで表示',f'Print clockwise spiral of matrix [{a},{b};{n},{c}]'
        body=f'int[][] m={{{{{a},{b}}},{{{n},{c}}}}};System.out.println(m[0][0]);System.out.println(m[0][1]);System.out.println(m[1][1]);System.out.println(m[1][0]);'
    else:
        raise AssertionError(family)
    return jp,en,_java(body)


def _expected_rows() -> dict[str,list[dict]]:
    result={}
    for split,(prior_families,prior_count,new_families,new_count) in SPLITS.items():
        out=[]
        for family in prior_families:
            for i in range(prior_count):
                ja,en,source=old_case(family,i)
                task=ja if i%2==0 else en
                out.append({'id':f'{family}-{i:04d}','family':family,'messages':[
                    {'role':'user','content':_model_user_message(task,BASE_SOURCE,'Main')},
                    {'role':'assistant','content':source}]})
        for family in new_families:
            for i in range(new_count):
                ja,en,source=new_case(family,i)
                task=ja if i%2==0 else en
                out.append({'id':f'{family}-{i:04d}','family':family,'messages':[
                    {'role':'user','content':_model_user_message(task,BASE_SOURCE,'Main')},
                    {'role':'assistant','content':source}]})
        random.Random(SEED+len(split)).shuffle(out)
        result[split]=out
    return result


def _expected_counts() -> dict[str,int]:
    return {k:len(old)*n+len(new)*m for k,(old,n,new,m) in SPLITS.items()}


def _quality(rows: dict[str,list[dict]]) -> dict:
    all_ids=set();all_hashes=set();families=[];by_split={}
    for split,entries in rows.items():
        ss=set()
        for row in entries:
            identifier=row['id']
            if identifier in all_ids:raise ValueError('Duplicate ID across splits')
            all_ids.add(identifier);ss.add(row['family'])
            source=row['messages'][1]['content']
            key=_digest(source.encode('utf-8'))
            if key in all_hashes:raise ValueError('Duplicate Java source across splits or within split')
            all_hashes.add(key)
        by_split[split]={'examples':len(entries),'families':len(ss),'duplicate_source_files':0}
        families.append(ss)
    if any(families[i]&families[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('Task family overlap across train and evaluation splits')
    return {'splits':by_split,'unique_sources':len(all_hashes),'unique_ids':len(all_ids),
            'family_disjoint':True,'independent_algorithms':False}


def build_java_diverse_corpus(output: str|Path) -> dict:
    root=Path(output).expanduser().absolute()
    if root.exists():raise FileExistsError('Never overwrite a dataset directory')
    data=_expected_rows()
    quality=_quality(data)
    root.mkdir(parents=True)
    try:
        files={}
        for split,rows in data.items():
            raw=('\n'.join(_json(x) for x in rows)+'\n').encode('utf8')
            if len(raw)>MAX_SPLIT_BYTES:raise ValueError('Corpus too large')
            path=root/(split+'.jsonl');path.write_bytes(raw)
            files[split]={'filename':path.name,'sha256':_digest(raw),'count':len(rows),
                          'families':list(SPLITS[split][0])+list(SPLITS[split][2])}
        manifest={'schema':SCHEMA,'seed':SEED,'files':files,'quality':quality,
                  'creation':'original code generation only; template-derived bilingual tasks',
                  'license_attestation':'NOT_LEGALLY_REVIEWED',
                  'third_party_sources':[],'model_weights_source':'NONE',
                  'data_split':'disjoint_by_whole_task_family'}
        (root/'manifest.json').write_text(_json(manifest)+'\n',encoding='utf8')
        return {'status':'CREATED','counts':_expected_counts(),**quality}
    except BaseException:
        import shutil
        shutil.rmtree(root)
        raise


def verify_java_diverse_corpus(folder: str|Path) -> dict:
    root=Path(folder)
    manifest=root/'manifest.json'
    if not manifest.is_file() or manifest.is_symlink() or manifest.stat().st_size>65536:
        raise ValueError('Missing or unsafe manifest')
    obj=json.loads(manifest.read_text(encoding='utf8'))
    if (obj.get('schema')!=SCHEMA or obj.get('seed')!=SEED or
        obj.get('license_attestation')!='NOT_LEGALLY_REVIEWED' or
        obj.get('third_party_sources')!=[] or obj.get('data_split')!='disjoint_by_whole_task_family' or
        set(obj.get('files',{}))!=set(SPLITS)):
        raise ValueError('Changed or untrusted dataset provenance')
    data=_expected_rows()
    reference=_quality(data)
    if obj.get('quality')!=reference:raise ValueError('Noncanonical quality manifest')
    for split,expected in data.items():
        info=obj['files'][split];file=root/(split+'.jsonl')
        if (not file.is_file() or file.is_symlink() or file.stat().st_size>MAX_SPLIT_BYTES
            or info.get('filename')!=file.name or info.get('count')!=len(expected)
            or info.get('families')!=list(SPLITS[split][0])+list(SPLITS[split][2])):
            raise ValueError('Invalid split metadata')
        raw=file.read_bytes()
        if _digest(raw)!=info.get('sha256'):raise ValueError('Source file SHA-256 mismatch')
        actual=[json.loads(line) for line in raw.decode('utf8').splitlines()]
        if actual!=expected:raise ValueError('Noncanonical generated Java source')
    return {'status':'PASS','counts':_expected_counts(),**reference}