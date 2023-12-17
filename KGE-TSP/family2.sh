# python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='1' -perfix='0.8_' --model='PairRE'  
# python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='1' -perfix='0.8_' --model='PairRE'  
# python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='1' -perfix='0.8_' --model='PairRE'  
#python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='PairRE'  -testGNN "EXPS/Family/atte_model_pairre_toKGE_SP0.8__vr497_vp11-T_r511_p29_l78664.pt" --max_steps=100000 --valid_steps=10000 --bs="atte"
#python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='PairRE'  -testGNN "EXPS/Family/atte_model_pairre_toKGE_SP0.8__vr497_vp11-T_r511_p29_l78664.pt" --max_steps=100000 --valid_steps=10000 --bs="atte"
#python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='PairRE'  -testGNN "EXPS/Family/atte_model_pairre_toKGE_SP0.8__vr497_vp11-T_r511_p29_l78664.pt" --max_steps=100000 --valid_steps=10000 --bs="atte"
python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='PairRE'  -testGNN "EXPS/Family/attr_model_pairre_toKGE_SP0.8__vr549_vp9-T_r565_p24_l107676.pt" --max_steps=250000 --valid_steps=10000 --bs="attr"
