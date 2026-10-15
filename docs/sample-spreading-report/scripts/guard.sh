# Run one process; kill it if its RSS exceeds $LIMIT_GB (default 8).
LIMIT_KB=$(( ${LIMIT_GB:-8} * 1024 * 1024 ))
"$@" & pid=$!
while kill -0 $pid 2>/dev/null; do
  rss=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' ')
  if [ -n "$rss" ] && [ "$rss" -gt "$LIMIT_KB" ]; then echo "KILLED: RSS ${rss} KB > limit"; kill -9 $pid; exit 99; fi
  sleep 0.5
done
wait $pid
