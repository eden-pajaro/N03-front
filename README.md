
**Instructions to run:**

First Setup:
Clone the repo and move to the project folder (like in the first command).
Run the following commands to create a virtual environment and install dependencies.
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
  cd ../..
  head -20 data/checkpoint/cells/0-0.cell
  head -20 defenses/results/ranpad2_1105_1633/0-0.cell
  

Results of Sanity Check:

Run without FRONT:

0.0     1

0.2516  1

0.4695  -1

0.4718  1

0.4719  1

0.7277  -1

0.7313  1

0.7314  1

0.975   -1

0.9751  -1

0.9751  -1

0.9751  -1

0.9752  -1

0.9752  -1

0.9752  -1

1.1849  -1

1.1915  1

1.1917  1

1.4026  -1

1.4027  -1

Run with FRONT:

0.0000  1

0.1847  888

0.2516  1

0.2572  888

0.3036  888

0.3375  888

0.4150  888

0.4695  -1

0.4718  1

0.4719  1

0.5142  888

0.5153  888

0.5629  888

0.6118  888

0.6390  888

0.6780  888

0.7277  -1

0.7313  1

0.7314  1

0.7653  888

What it Means:

First number: when the packet was sent, in seconds since the page started loading.

Second number: which way it went.

1 = your computer → website (e.g. “give me the page”)

-1 = website → your computer (e.g. the page’s content)

888 = a fake packet FRONT added; real ones are only ever 1 or -1

## Team Contributions

| Team member | Bounded contribution | Work completed | Evidence to link |
|---|---|---|---|
| Eden Pajaro | Dataset preparation and documentation | Found the dataset, uploaded the checkpoint traces to `data/checkpoint/cells`, and organized the README | Dataset upload commit and README revision |
| Riya Ramakrishnan | Code compatibility | Updated the code to use the checkpoint data paths and work with the newer Python environment. | Compatibility commit or pull request |
| Reah Robinson | FRONT runs and results | Ran the FRONT defense several times and averaged the results across runs. | Individual run logs and the averaged results |

