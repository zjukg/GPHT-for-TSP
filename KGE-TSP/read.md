python runs.py -train -test -data=DATASET -gpu=GPU -perfix=PERFIX --valid_steps=STEP
```

`PERFIX`: set the integrity of the dataset in the format of `percent_`, like `0.6_`

`STEP`: do valid every `STEP` steps

Family
#######
python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.2_' --model='HAKE' --
python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.4_' --model='HAKE' --
python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR'
python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='HAKE' --

python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='1' -perfix='0.2_' --model='PairRE'--
python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='1' -perfix='0.4_' --model='PairRE'--
python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='1' -perfix='0.6_' --model='PairRE'--
python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='1' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR'

python HAKE-TSP/run_close.py -test -data='Family' -gpu='1' -perfix='0.8_' --model='PairRE' --init_checkpoint='EXPS/Family/PairRE-0.8_transe-20230501_21:18'



Wiki
python  HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='2' -perfix='0.2_' --model='HAKE'
python  HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.4_' --model='HAKE'
python  HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.6_' --model='HAKE'
python  HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='HAKE'
python  HAKE-TSP/run_open.py -train -test -data='wiki' -gpu='1' -perfix='0.8_' --model='HAKE'

python  HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.2_' --model='PairRE'
python  HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.4_' --model='PairRE'
python  HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.6_' --model='PairRE'
python  HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='PairRE'
python  HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.6_' --model='HAKE'

wiki_select_new_3:
HAKE#800000才是最好值
PairRE# 40000、50000、50000最好值

wiki_select_new:
HAKE #1100000左右是最好值
PairRE #40000左右
