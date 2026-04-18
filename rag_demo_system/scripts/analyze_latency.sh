#!/usr/bin/env bash
# Analyze LATENCY log lines. Usage: bash scripts/analyze_latency.sh [LOGFILE]
# Default logfile: .state/backend.log

LOG="${1:-.state/backend.log}"

echo "=== Latency distribution (total_e2e_ms) ==="
grep -oE "total_e2e_ms=[0-9]+" "$LOG" | awk -F= '{print $2}' | sort -n | awk '
  BEGIN { c=0 }
  { a[c++]=$1; sum+=$1 }
  END {
    if (c==0) { print "no samples"; exit }
    n=c
    printf "count=%d mean=%.0f min=%d max=%d\n", n, sum/n, a[0], a[n-1]
    printf "p50=%d p90=%d p95=%d p99=%d\n",
      a[int(n*0.50)], a[int(n*0.90)], a[int(n*0.95)], a[int(n*0.99)]
  }
'

echo
echo "=== Classifier latency distribution ==="
grep -oE "classifier_ms=[0-9]+" "$LOG" | awk -F= '{print $2}' | awk '$1 >= 0' | sort -n | awk '
  BEGIN { c=0 }
  { a[c++]=$1; sum+=$1 }
  END {
    if (c==0) { print "no samples"; exit }
    n=c
    printf "count=%d mean=%.0f min=%d max=%d\n", n, sum/n, a[0], a[n-1]
    printf "p50=%d p90=%d p95=%d p99=%d\n",
      a[int(n*0.50)], a[int(n*0.90)], a[int(n*0.95)], a[int(n*0.99)]
  }
'

echo
echo "=== Top 10 slowest total_e2e turns (to hunt outliers) ==="
grep "\[LATENCY:" "$LOG" | awk -F'total_e2e_ms=' 'NF>1 {
  n=$2; split(n, p, " "); print p[1] "\t" $0
}' | sort -n | tail -10

echo
echo "=== Correlation: does user_len predict classifier_ms? (slowest 10 classifier calls) ==="
grep "\[LATENCY:" "$LOG" | awk -F'classifier_ms=' 'NF>1 {
  split($2, p, " "); c=p[1]; line=$0
  match(line, /user_len=[0-9]+/); ul = substr(line, RSTART+9, RLENGTH-9)
  if (c+0 > 0) printf "%d\t%d\n", c+0, ul+0
}' | sort -n | tail -10
