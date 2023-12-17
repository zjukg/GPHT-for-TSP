# python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='HAKE'
# python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='PairRE'
# python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='PairRE'  -testGNN "EXPS/Family/atte_attr_model_pairre_toKGE_SP0.8__vr799_vp11-T_r823_p30_l126025.pt" --max_steps=100000 --valid_steps=10000 --bs="atte_attr"
# python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='PairRE'  -testGNN "EXPS/Family/atte_attr_model_pairre_toKGE_SP0.8__vr799_vp11-T_r823_p30_l126025.pt" --max_steps=100000 --valid_steps=10000 --bs="atte_attr"
# python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='PairRE'  -testGNN "EXPS/Family/atte_attr_model_pairre_toKGE_SP0.8__vr799_vp11-T_r823_p30_l126025.pt" --max_steps=100000 --valid_steps=10000 --bs="atte_attr"
# python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='PairRE'  -testGNN "EXPS/Family/attr_model_pairre_toKGE_SP0.8__vr549_vp9-T_r565_p24_l107676.pt" --max_steps=640000 --valid_steps=10000 --bs="attr" --db=True --best_evaluate='MRR'
# python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='PairRE'  -testGNN "EXPS/Family/attr_model_pairre_toKGE_SP0.8__vr549_vp9-T_r565_p24_l107676.pt" --max_steps=640000 --valid_steps=10000 --bs="attr" --db=True --best_evaluate='MRR'
# python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='0' -perfix='0.8_' --model='PairRE'  -testGNN "EXPS/Family/attr_model_pairre_toKGE_SP0.8__vr549_vp9-T_r565_p24_l107676.pt" --max_steps=640000 --valid_steps=10000 --bs="attr" --db=True --best_evaluate='MRR'
python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='1' -perfix='0.8_' --model='HAKE'  -testGNN="EXPS/Family/model_hake__minconf0.3_toKGE_SP0.8_V_f15.4860_r889_p7-T_r908_p19_l209320.pt" --max_steps=200000 --valid_steps=5000 --best_evaluate="F1" --bs="normal"
#python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='1' -perfix='0.8_' --model='HAKE'  -testGNN="EXPS/Family/model_hake__minconf0.3_toKGE_SP0.8_V_f15.4860_r889_p7-T_r908_p19_l209320.pt" --max_steps=200000 --valid_steps=5000 --best_evaluate="F1" --bs="normal"

#python HAKE-TSP/run_close.py -train -test -data='Family' -gpu='1' -perfix='0.8_' --model='HAKE' -testGNN "EXPS/Family/model_hake__minconf0.3_toKGE_SP0.8_V_f15.4860_r889_p7-T_r908_p19_l209320.pt"