# Run a command; kill its whole process tree if the tree's total RSS exceeds $LIMIT_GB.
LIMIT_KB=$(( ${LIMIT_GB:-16} * 1024 * 1024 ))
"$@" & root=$!
tree() { local p; for p in $(pgrep -P $1); do echo $p; tree $p; done; }
peak=0
while kill -0 $root 2>/dev/null; do
  pids="$root $(tree $root)"
  rss=$(ps -o rss= -p $(echo $pids | tr ' ' ',') 2>/dev/null | awk '{s+=$1} END {print s+0}')
  [ "$rss" -gt "$peak" ] && peak=$rss && echo "$peak" > ${PEAK_FILE:-/dev/null}
  if [ "$rss" -gt "$LIMIT_KB" ]; then
    echo "KILLED: tree RSS ${rss} KB > ${LIMIT_GB:-16} GB"; kill -9 $pids 2>/dev/null; exit 99
  fi
  sleep 1
done
wait $root
