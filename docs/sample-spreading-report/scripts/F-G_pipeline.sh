set -u
S=/private/tmp/claude-501/-Users-dylan-dev-pulseq-relaxed-rosettes/6d224db4-ae46-4fa9-8933-eb00150e8494/scratchpad/hairpin
cd /Users/dylan/dev/pulseq-relaxed-rosettes
R=.venv/bin/relaxed-rosettes
s=$(date +%s); $R shapes --set shapes.num_shapes=16 --jobs 5 -o $S/hp16.shapes.npz > $S/hp16.log 2>&1; echo "hp16 shapes $(( $(date +%s)-s ))s exit $?" >> $S/status
$R sequence $S/hp16.shapes.npz -o $S/hp16.seq > $S/hp16.seq.log 2>&1
s=$(date +%s); $R coverage $S/hp16.seq > $S/hp16.cov.txt 2>&1; echo "hp16 coverage $(( $(date +%s)-s ))s" >> $S/status
s=$(date +%s); .venv/bin/python -W ignore $S/repel.py $S/hp16.shapes.npz $S/hp_lam3.shapes.npz shapes.repulsion_iterations=20 shapes.repulsion_weight=3 > $S/hp_lam3.log 2>&1; echo "hp_lam3 repel $(( $(date +%s)-s ))s exit $?" >> $S/status
$R sequence $S/hp_lam3.shapes.npz -o $S/hp_lam3.seq > $S/hp_lam3.seq.log 2>&1
s=$(date +%s); $R coverage $S/hp_lam3.seq > $S/hp_lam3.cov.txt 2>&1; echo "hp_lam3 coverage $(( $(date +%s)-s ))s" >> $S/status
echo DONE >> $S/status
