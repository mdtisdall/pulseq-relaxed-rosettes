set -u
S=/private/tmp/claude-501/-Users-dylan-dev-pulseq-relaxed-rosettes/a8ca02b4-9385-42df-9fbc-2bd8229d967a/scratchpad/sweep
cd /Users/dylan/dev/pulseq-relaxed-rosettes/.worktrees/repulsion
n=$1; shift
s=$(date +%s)
.venv/bin/python -W ignore $S/repel.py $S/base16.shapes.npz $S/$n.shapes.npz "$@" > $S/$n.log 2>&1
echo "$n repel $(( $(date +%s)-s ))s exit $?" >> $S/status
.venv/bin/relaxed-rosettes sequence $S/$n.shapes.npz -o $S/$n.seq > $S/$n.seq.log 2>&1
s=$(date +%s)
.venv/bin/relaxed-rosettes coverage $S/$n.seq > $S/$n.cov.txt 2>&1
echo "$n coverage $(( $(date +%s)-s ))s" >> $S/status
