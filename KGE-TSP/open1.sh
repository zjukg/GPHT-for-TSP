#! /bin/bash
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_' --model='PairRE'
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_' --model='PairRE' --best_evaluate='MRR'
python HAKE-TSP/run_open.py -train -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  -testGNN "EXPS/wiki_select_new_3/atte_attr_model_hake_toKGE_SP0.8__vr514_vp1-T_r521_p3_l4271179.pt" --model="HAKE"  --init_steps=400000 --max_steps=1000000 --valid_steps=10000 --bs="atte_attr"
# python HAKE-TSP/run_open.py -train -test -data='wiki_select_new' -gpu=0 -perfix='0.8_'  -testGNN "EXPS/wiki_select_new/model_wiki_select_new_pairre__minconf0.2_toKGE_SP0.8_V_f14.2060_r608_p7-T_r635_p18_l536391.pt" --model="PairRE" --best_evaluate='MRR'



# python HAKE-TSP/run_open.py -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  --model="HAKE" --best_evaluate='F1' --init_checkpoint="EXPS/wiki_select_new_3/HAKE-0.8_transe-20230712_23:20"
# python HAKE-TSP/run_open.py -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  --model="HAKE" --best_evaluate='F1' --init_checkpoint="EXPS/wiki_select_new_3/HAKE-0.8_transe-20230711_23:05"
# python HAKE-TSP/run_open.py -test -data='wiki_select_new_3' -gpu=0 -perfix='0.8_'  --model="HAKE" --best_evaluate='F1' --init_checkpoint="EXPS/wiki_select_new_3/HAKE-0.8_transe-20230710_23:39"

# python HAKE-TSP/run_open.py -test -data='wiki_select_new_3' -gpu=1 -perfix='0.8_'  --model="PairRE" --best_evaluate='F1' --init_checkpoint="EXPS/wiki_select_new_3/PairRE-0.8_transe-20230803_08:29"


# python GPHT/run.py -dataset='wiki_select_new_3' -perfix='0.8_' -lr=0.00003 -restore='EXPS/wiki_select_new_3/0.8_wiki_new_3-pairre-0.8_rotate-20230527-23_26' -batch=1 -epoch=3000 -valid_epochs=2 -score_func='pairre' -minconf=0.15
# python GPHT/run.py -dataset='wiki_select_new_3' -perfix='0.8_' -lr=0.00003 -restore='EXPS/wiki_select_new_3/0.8_wiki_new_3-pairre-0.8_rotate-20230527-23_26' -batch=1 -epoch=3000 -valid_epochs=2 -score_func='pairre' -minconf=0.35
# python GPHT/run.py -dataset='wiki_select_new_3' -perfix='0.8_' -lr=0.00003 -restore='EXPS/wiki_select_new_3/0.8_wiki_new_3-pairre-0.8_rotate-20230527-23_26' -batch=1 -epoch=3000 -valid_epochs=2 -score_func='pairre' -minconf=0.4