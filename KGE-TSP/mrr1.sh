#python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' --rank=500
#python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' --rank=1000
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' --max_steps=200000
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' --max_steps=200000
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='1' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' -seed=3407 --init_steps=400000 --max_step=600000
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='1' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' -seed=3407 --init_steps=5000 --max_step=100000
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='1' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' -seed=3407 --init_steps=5000 --max_step=100000
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='1' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' -seed=3407 --init_steps=5000 --max_step=100000
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' -seed=3407 --init_steps=400000 --max_step=650000
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' -seed=3407 --init_steps=400000 --max_step=650000
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='0' -perfix='0.8_' --model='HAKE' --best_evaluate='F1' -seed=3407 --init_steps=10000 --max_step=350000 -testGNN="EXPS/wiki_select_new_3/model_wiki_select_new_3_hake__minconf0.4_toKGE_SP0.8_V_f0.7355_r729_p0-T_r735_p0_l22779460.pt"
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='0' -perfix='0.8_' --model='HAKE' --best_evaluate='F1' -seed=41504 -testGNN="EXPS/wiki_select_new_3/model_wiki_select_new_3_hake__minconf0.4_toKGE_SP0.8_V_f0.7979_r682_p0-T_r688_p1_l19640684.pt" --init_checkpoint='EXPS/wiki_select_new_3/HAKE-0.8_transe-20230804_00:58'
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='0' -perfix='0.8_' --model='HAKE' --best_evaluate='F1' -seed=41504 -testGNN="EXPS/wiki_select_new_3/model_wiki_select_new_3_hake__minconf0.4_toKGE_SP0.8_V_f0.7979_r682_p0-T_r688_p1_l19640684.pt" --init_checkpoint='EXPS/wiki_select_new_3/HAKE-0.8_transe-20230803_00:09'
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='0' -perfix='0.8_' --model='HAKE' --best_evaluate='F1' -seed=41504 -testGNN="EXPS/wiki_select_new_3/model_wiki_select_new_3_hake__minconf0.4_toKGE_SP0.8_V_f0.7979_r682_p0-T_r688_p1_l19640684.pt" --init_checkpoint='EXPS/wiki_select_new_3/HAKE-0.8_transe-20230803_08:51'
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' -seed=3407 --init_steps=600000 --max_step=1200000 -testGNN="EXPS/wiki_select_new/model_wiki_select_new_hake__minconf0.35_toKGE_SP0.8_V_f2.2939_r549_p1-T_r577_p3_l3032156.pt"

#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' -seed=3407 --init_steps=100000 --max_step=200000  --init_checkpoint="EXPS/wiki_select_new/PairRE-0.8_transe-20230715_11:01"
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' -seed=3407 --init_steps=400000 --max_step=650000 --init_checkpoint="EXPS/wiki_select_new/HAKE-0.8_transe-20230714_23:50"
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' -seed=3407 --init_steps=400000 --max_step=650000 --init_checkpoint="EXPS/wiki_select_new/HAKE-0.8_transe-20230716_08:20"
# #python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='1' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' -seed=3407
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' --init_steps=400000 --max_step=1200000 -seed=41504 -testGNN='EXPS/wiki_select_new/model_wiki_select_new_hake__minconf0.35_toKGE_SP0.8_V_f2.2939_r549_p1-T_r577_p3_l3032156.pt'

# python HAKE-TSP/run_open.py -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' -seed=3407 --init_checkpoint="EXPS/wiki_select_new/HAKE-0.8_transe-20230718_15:35"
python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  -testGNN "EXPS/wiki_select_new_3/model_wiki_select_new_3_pairre__minconf0.4_toKGE_SP0.8_V_f4.2710_r578_p2-T_r593_p5_l3103475.pt" --model="PairRE"  --init_steps=10000 --max_steps=150000 --valid_steps=10000 --bs="noraml"
python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  -testGNN "EXPS/wiki_select_new_3/model_wiki_select_new_3_pairre__minconf0.4_toKGE_SP0.8_V_f4.2710_r578_p2-T_r593_p5_l3103475.pt" --model="PairRE"  --init_steps=10000 --max_steps=150000 --valid_steps=10000 --bs="normal"
python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  -testGNN "EXPS/wiki_select_new_3/model_wiki_select_new_3_pairre__minconf0.4_toKGE_SP0.8_V_f4.2710_r578_p2-T_r593_p5_l3103475.pt" --model="PairRE"  --init_steps=10000 --max_steps=150000 --valid_steps=10000 --bs="normal"


