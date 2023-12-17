#! /bin/bash
# python  HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='1' -perfix='0.8_' --model='HAKE'
# python  HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu='0' -perfix='0.8_' --model='PairRE'
# python  HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='0' -perfix='0.8_' --model='HAKE'
#python  HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu='0' -perfix='0.8_' --model='PairRE'
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  -testGNN "EXPS/wiki_select_new_3/model_wiki_select_new_3_pairre__minconf0.4_toKGE_SP0.8_V_f4.2710_r578_p2-T_r593_p5_l3103475.pt" --model="PairRE"
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  -testGNN "EXPS/wiki_select_new_3/model_wiki_select_new_3_pairre__minconf0.4_toKGE_SP0.8_V_f4.2710_r578_p2-T_r593_p5_l3103475.pt" --model="PairRE" --best_evaluate='MRR'

#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu=0 -perfix='0.8_'  -testGNN "EXPS/wiki_select_new/model_wiki_select_new_pairre__minconf0.2_toKGE_SP0.8_V_f14.1873_r610_p7-T_r637_p18_l538798.pt" --model="PairRE"
#python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  -testGNN "EXPS/wiki_select_new_3/model_wiki_select_new_3_pairre__minconf0.4_toKGE_SP0.8_V_f6.5867_r569_p3-T_r579_p8_l1976159.pt" --model="PairRE"
#python GPHT/run.py -dataset='wiki_select_new' -perfix='0.8_' -lr=0.00003 -restore='EXPS/wiki_select_new/0.8_wiki_new-pairre-0.8_rotate-20230527-14_00' -batch=1 -epoch=3000 -valid_epochs=2 -score_func='pairre' -minconf=0.4
#python GPHT/run.py -dataset='wiki_select_new' -perfix='0.8_' -lr=0.00003 -restore='EXPS/wiki_select_new/0.8_wiki_new-pairre-0.8_rotate-20230527-14_00' -batch=1 -epoch=3000 -valid_epochs=2 -score_func='pairre' -minconf=0.15
# python HAKE-TSP/run_open.py -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  --model="HAKE" --best_evaluate='F1' --init_checkpoint="EXPS/wiki_select_new_3/HAKE-0.8_transe-20230804_00:58" -testGNN="EXPS/wiki_select_new_3/model_wiki_select_new_3_hake__minconf0.4_toKGE_SP0.8_V_f0.7979_r682_p0-T_r688_p1_l19640684.pt"
# python HAKE-TSP/run_open.py -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  --model="HAKE" --best_evaluate='F1' --init_checkpoint="EXPS/wiki_select_new_3/HAKE-0.8_transe-20230803_08:51" -testGNN="EXPS/wiki_select_new_3/model_wiki_select_new_3_hake__minconf0.4_toKGE_SP0.8_V_f0.7979_r682_p0-T_r688_p1_l19640684.pt"
# python HAKE-TSP/run_open.py -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  --model="HAKE" --best_evaluate='F1' --init_checkpoint="EXPS/wiki_select_new_3/HAKE-0.8_transe-20230803_00:09" -testGNN="EXPS/wiki_select_new_3/model_wiki_select_new_3_hake__minconf0.4_toKGE_SP0.8_V_f0.7979_r682_p0-T_r688_p1_l19640684.pt"
python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  -testGNN "EXPS/wiki_select_new_3/atte_attr_model_pairre_toKGE_SP0.8__vr553_vp1-T_r562_p4_l3382510.pt" --model="PairRE"  --init_steps=40000 --max_steps=150000 --valid_steps=10000 --bs="atte_attr"
python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  -testGNN "EXPS/wiki_select_new_3/atte_attr_model_pairre_toKGE_SP0.8__vr553_vp1-T_r562_p4_l3382510.pt" --model="PairRE"  --init_steps=40000 --max_steps=150000 --valid_steps=10000 --bs="atte_attr"