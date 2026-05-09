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
echo "=== Classifier latency (existing markers) ==="
grep -oE "\[Classifier\] result: \([0-9]+ms\)" "$LOG" | grep -oE "[0-9]+" | sort -n | awk '
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
echo "=== STT latency (existing markers) ==="
grep -oE "STT\([0-9]+ms\)" "$LOG" | grep -oE "[0-9]+" | sort -n | awk '
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
echo "=== LLM brain first-token (FireLLMFallback only) ==="
grep -oE "llm_first_ms=[0-9]+" "$LOG" | awk -F= '{print $2}' | awk '$1 >= 0' | sort -n | awk '
  BEGIN { c=0 }
  { a[c++]=$1; sum+=$1 }
  END {
    if (c==0) { print "no samples (deploy instrumentation, run turns)"; exit }
    n=c
    printf "count=%d mean=%.0f min=%d max=%d\n", n, sum/n, a[0], a[n-1]
    printf "p50=%d p90=%d p95=%d p99=%d\n",
      a[int(n*0.50)], a[int(n*0.90)], a[int(n*0.95)], a[int(n*0.99)]
  }
'

echo
echo "=== LLM brain total stream (FireLLMFallback only) ==="
grep -oE "llm_total_ms=[0-9]+" "$LOG" | awk -F= '{print $2}' | sort -n | awk '
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
echo "=== Action distribution (P2 over-routing check) ==="
echo "Counts every apply_turn dispatch by action kind. The 'over-routing'"
echo "claim is: FireLLMFallback share is much higher than necessary."
grep -oE "\[LATENCY:[^]]+\] action_kind=[A-Za-z]+" "$LOG" | awk '{print $NF}' | awk -F= '{print $2}' | sort | uniq -c | sort -rn

echo
echo "=== Action share (% of total dispatches) ==="
grep -oE "action_kind=[A-Za-z]+" "$LOG" | awk -F= '{print $2}' | awk '
  { total++; counts[$1]++ }
  END {
    if (total==0) { print "no samples (per-turn LATENCY marker not yet deployed)"; exit }
    for (k in counts) printf "%-22s %5d  %5.1f%%\n", k, counts[k], 100.0*counts[k]/total
  }
' | sort -k2 -rn

echo
echo "=== Dispatch overhead (apply_turn body time, ms) ==="
grep -oE "dispatch_ms=[0-9]+" "$LOG" | awk -F= '{print $2}' | sort -n | awk '
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
