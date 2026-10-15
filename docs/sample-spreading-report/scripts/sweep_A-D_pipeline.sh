set -u
S=/private/tmp/claude-501/-Users-dylan-dev-pulseq-relaxed-rosettes/a8ca02b4-9385-42df-9fbc-2bd8229d967a/scratchpad/sweep
cd /Users/dylan/dev/pulseq-relaxed-rosettes/.worktrees/repulsion
R=.venv/bin/relaxed-rosettes
s=$(date +%s); $R shapes --set shapes.num_shapes=16 --set shapes.crossing_tolerance_cells=1 --set shapes.tip_width=0.5 --jobs 5 -o $S/base16.shapes.npz > $S/base16.log 2>&1
echo "base16 shapes $(( $(date +%s)-s ))s exit $?" >> $S/status
$R sequence $S/base16.shapes.npz -o $S/base16.seq > $S/base16.seq.log 2>&1
s=$(date +%s); $R coverage $S/base16.seq > $S/base16.cov.txt 2>&1; echo "base16 coverage $(( $(date +%s)-s ))s" >> $S/status
for lam in 0.5 1 3; do
  zsh $S/run.sh lam$lam shapes.repulsion_iterations=20 shapes.repulsion_weight=$lam
done
echo DONE >> $S/status
