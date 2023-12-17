#python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR'
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' --init_steps=600000 --max_step=1200000 -testGNN="EXPS/wiki_select_new/model_wiki_select_new_hake__minconf0.35_toKGE_SP0.8_V_f2.2939_r549_p1-T_r577_p3_l3032156.pt" 
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' --init_steps=600000 --max_step=1200000 -testGNN="EXPS/wiki_select_new/model_wiki_select_new_hake__minconf0.35_toKGE_SP0.8_V_f2.2939_r549_p1-T_r577_p3_l3032156.pt" 
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' --init_steps=600000 --max_step=1200000 -testGNN="EXPS/wiki_select_new/model_wiki_select_new_hake__minconf0.35_toKGE_SP0.8_V_f2.2939_r549_p1-T_r577_p3_l3032156.pt" 

#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' --init_steps=10000 --max_step=600000 -testGNN="EXPS/wiki_select_new/model_wiki_select_new_hake__minconf0.05_toKGE_SP0.8_V_f2.3226_r238_p1-T_r243_p2_l1295675.pt" 
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' --init_steps=10000 --max_step=600000 -testGNN="EXPS/wiki_select_new/model_wiki_select_new_hake__minconf0.005_toKGE_SP0.8_V_f2.0009_r224_p1-T_r216_p2_l1413040.pt"


#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR'  -testGNN="EXPS/wiki_select_new/model_wiki_select_new_pairre__minconf0.2_toKGE_SP0.8_V_f14.2060_r608_p7-T_r635_p18_l536391.pt"
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='1' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR'  -testGNN="EXPS/wiki_select_new_3/model_wiki_select_new_3_pairre__minconf0.4_toKGE_SP0.8_V_f4.2710_r578_p2-T_r593_p5_l3103475.pt"
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='HAKE' --best_evaluate='F1' -seed=3407 -testGNN="EXPS/wiki_select_new_3/model_wiki_select_new_3_hake__minconf0.4_toKGE_SP0.8_V_f0.7979_r682_p0-T_r688_p1_l19640684.pt"
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' --init_steps=200000 --max_step=800000 -seed=41504

# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='1' -perfix='0.8_' --model='PairRE' --best_evaluate='F1' --init_steps=20000 --max_step=200000
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='1' -perfix='0.8_' --model='PairRE' --best_evaluate='F1' --init_steps=20000 --max_step=200000

#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='0' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' -seed=3407 --max_step=200000 -testGNN="EXPS/wiki_select_new_3/model_wiki_select_new_3_pairre__minconf0.4_toKGE_SP0.8_V_f4.2710_r578_p2-T_r593_p5_l3103475.pt"
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='0' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' -seed=3407 --max_step=200000 -testGNN="EXPS/wiki_select_new_3/model_wiki_select_new_3_pairre__minconf0.4_toKGE_SP0.8_V_f4.2710_r578_p2-T_r593_p5_l3103475.pt"
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='0' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' -seed=3407 --max_step=200000 -testGNN="EXPS/wiki_select_new_3/model_wiki_select_new_3_pairre__minconf0.4_toKGE_SP0.8_V_f4.2710_r578_p2-T_r593_p5_l3103475.pt"
# wiki_select_new
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' -seed=3407 --init_steps=100000 --max_step=200000  --init_checkpoint="EXPS/wiki_select_new/PairRE-0.8_transe-20230715_11:01"
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' -seed=3407 --init_steps=600000 --max_step=1200000 -testGNN="EXPS/wiki_select_new/model_wiki_select_new_hake__minconf0.35_toKGE_SP0.8_V_f2.6979_r386_p1-T_r399_p3_l1807645.pt"
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' -seed=3407 --init_steps=600000 --max_step=1200000 -testGNN="EXPS/wiki_select_new/model_wiki_select_new_hake__minconf0.35_toKGE_SP0.8_V_f2.1077_r700_p1-T_r728_p2_l4204876.pt"

# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='HAKE' --best_evaluate='MRR' -seed=3407 --init_steps=600000 --max_step=1200000 -testGNN="EXPS/wiki_select_new/model_wiki_select_new_hake__minconf0.35_toKGE_SP0.8_V_f2.1077_r700_p1-T_r728_p2_l4204876.pt" --init_checkpoint="EXPS/wiki_select_new/HAKE-0.8_transe-20230713_08:57"

#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' -seed=3407 --init_steps=5000 --max_step=200000 -testGNN="EXPS/wiki_select_new/model_wiki_select_new_pairre__minconf0.2_toKGE_SP0.8_V_f13.1014_r607_p6-T_r639_p17_l581541.pt"
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' -seed=3407 --init_steps=5000 --max_step=200000 -testGNN="EXPS/wiki_select_new/model_wiki_select_new_pairre__minconf0.2_toKGE_SP0.8_V_f13.1014_r607_p6-T_r639_p17_l581541.pt" 
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' -seed=3407 --init_steps=5000 --max_step=200000 -testGNN="EXPS/wiki_select_new/model_wiki_select_new_pairre__minconf0.2_toKGE_SP0.8_V_f13.1014_r607_p6-T_r639_p17_l581541.pt"
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' -seed=3407 --init_steps=5000 --max_step=100000
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' -seed=3407 --init_steps=5000 --max_step=100000
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='PairRE' --best_evaluate='MRR' -seed=3407 --init_steps=5000 --max_step=100000
python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  -testGNN "EXPS/Family/attr_model_pairre_toKGE_SP0.8__vr549_vp9-T_r565_p24_l107676.pt" --model="PairRE"  --init_steps=40000 --max_steps=150000 --valid_steps=10000 --bs="atte_attr"
