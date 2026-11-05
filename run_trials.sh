#!/bin/bash
# Repeated-trial runner for the FRONT checkpoint baseline.
# Usage (from the repo root):  bash run_trials.sh 10
N=${1:-10}
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA="$ROOT/data/checkpoint/cells/"
LOGS="$ROOT/logs"
RES="$ROOT/defenses/results"
mkdir -p "$LOGS" "$RES" "$ROOT/attacks/kfingerprinting/results"
CSV="$LOGS/trials.csv"
echo "condition,trial,kfp_accuracy,overhead" > "$CSV"

COUNT=$(ls "$DATA" | wc -l | tr -d ' ')
echo "Input traces: $COUNT (expected 1010)" | tee "$LOGS/trials_summary.txt"

cd "$ROOT/attacks/kfingerprinting"
python3 extract.py "$DATA" > /dev/null 2>&1

for t in 1 2 3; do
  ACC=$(python3 new_main.py results/cells.npy 2>&1 | tail -1)
  echo "undefended,$t,$ACC," >> "$CSV"
  echo "undefended run $t: accuracy=$ACC"
done

for t in $(seq 1 "$N"); do
  cd "$ROOT/defenses/front"
  python3 main.py "$DATA" -format .cell -c checkpoint > "$LOGS/front_trial_$t.log" 2>&1
  NEW=$(ls -td "$RES"/ranpad2_* | grep -v _norm | head -1)
  OUT="$RES/front_trial_$t"
  rm -rf "$OUT" "${OUT}_norm"
  mv "$NEW" "$OUT"
  NOUT=$(ls "$OUT" | wc -l | tr -d ' ')

  cd "$ROOT/utils"
  OVH=$(python3 overhead.py "$OUT/" -format .cell 2>&1 | grep "Random Padding" | awk '{print $NF}')
  python3 norm.py "$OUT/" > /dev/null 2>&1

  cd "$ROOT/attacks/kfingerprinting"
  python3 extract.py "${OUT}_norm/" > /dev/null 2>&1
  ACC=$(python3 new_main.py "results/front_trial_${t}_norm.npy" 2>&1 | tail -1)
  echo "front,$t,$ACC,$OVH" >> "$CSV"
  echo "FRONT run $t: output files=$NOUT accuracy=$ACC overhead=$OVH"
  sleep 1
done

python3 - "$CSV" <<'EOF' | tee -a "$LOGS/trials_summary.txt"
import csv, sys, statistics as s
rows = list(csv.DictReader(open(sys.argv[1])))
for cond in ("undefended", "front"):
    acc = [float(r["kfp_accuracy"]) for r in rows if r["condition"] == cond]
    print(f"{cond}: n={len(acc)}  kFP accuracy mean={s.mean(acc):.4f}  std={s.stdev(acc) if len(acc)>1 else 0:.4f}")
    if cond == "front":
        ov = [float(r["overhead"]) for r in rows if r["condition"] == cond]
        print(f"front: overhead mean={s.mean(ov):.4f}  std={s.stdev(ov) if len(ov)>1 else 0:.4f}")
EOF
echo "Saved: logs/trials.csv and logs/trials_summary.txt"
