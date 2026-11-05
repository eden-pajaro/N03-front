Instructions to run:
First Setup:

cd ~/Documents/GitHub/N03-front
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn matplotlib joblib

Run FRONT:
cd defenses/front
python main.py ../../data/checkpoint/cells/ -format .cell -c checkpoint

Normalize:
cd ../..
python utils/overhead.py defenses/results/ranpad2_1105_1633/ -format .cell
python utils/norm.py defenses/results/ranpad2_1105_1633/

Run CUMUL:
cd attacks/cumul
mkdir -p results
python extract.py ../../data/checkpoint/cells/
python extract.py ../../defenses/results/ranpad2_1105_1633_norm/
python main.py results/cells.npy
python main.py results/ranpad2_1105_1633_norm.npy

Sanity Check:
head -20 data/checkpoint/cells/0-0.cell
head -20 defenses/results/ranpad2_1105_1633/0-0.cell
