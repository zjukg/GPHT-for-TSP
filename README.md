
# Preprocess

bash init.sh

# RuleTensor-TSP

- CFamily  
  python RuleTensor-TSP/GraphRule_close.py -dataset=DATASET  -rule_len=LEN -hc_thr=HC -sc_thr=SC -percent=0.8 -gpu=GPU

- Wiki79k and Wiki143k  
  python RuleTensor-TSP/GraphRule_open.py -dataset=DATASET  -rule_len=LEN -hc_thr=HC -sc_thr=SC -percent=0.8 -gpu=GPU
      
  -DATASET: choose the dataset in `DATA/`  
  -LEN: set the length of rule  
  -HC: set the head coverage threshold of rule  
  -SC: set the standard confidence threshold of rule  
  -PER: set the integrity of the dataset  
  -GPU: -1 for cpu, otherwise the gpu id  
# KGE-TSP

- CFamily  
  python HAKE-TSP/run_close.py -train -test -data=DATASET -gpu=GPU -perfix='0.8_' --model=MODEL  --valid_steps=STEP

- Wiki79k and Wiki143k   
  python  HAKE-TSP/run_open.py -train -test -data=DATASET -gpu=gpu -perfix='0.8_' --model=MODEL --valid_steps=STEP
     
  -MODEL: the choice of KGE model, ['HAKE', 'PairRE']  
  -PERFIX: set the integrity of the dataset in the format of `percent_`  
  -STEP: do valid every `STEP` steps  
# GPHT

1. generate subgraphs

    python GPHT/run.py -dataset=DATASET -subgraph=SUBLEN -perfix=PERFIX  
    
    -SUBLEN: set max hops of subgraph from center to edge  
  
2. pre-train embeddings

    python GPHT/run.py -dataset=DATASET -subgraph=SUBLEN -perfix=PERFIX -batch=BATCH -pretrain -desc=DESC

3. train the model

    python GPHT/run.py -dataset=DATASET -perfix=PERFIX -lr=LR -restore=RESTORE -batch=1 -epoch=EPOCH -valid_epochs=STEP -score_func=MODEL -minconf=MINCONF  
      
    -LR: a little scale number for learning rate, like 0.00003 or less  
    -MINCONF: selecting the final predicted triples  
  
4. predict triples(in `KGE-TSP`)

    - CFamily  
      python HAKE-TSP/run_close.py -train -test -data=DATASET -gpu=0 -perfix='0.8_'  -testGNN "EXPS/CFamily/toKGE_XXX.pt" -model=MODEL

    -  Wiki143k and Wiki79k  
      python HAKE-TSP/run_open.py -train -test -data=DATASET -gpu=0 -perfix='0.8_'  -testGNN "EXPS/DATASET/toKGE_XXX.pt" -model=MODEL -valid_steps=STEP


# Acknowledgement
We refer to the code of [HAKE](https://github.com/MIRALab-USTC/KGE-HAKE)、[PairRE](https://github.com/ant-research/KnowledgeGraphEmbeddingsViaPairedRelationVectors_PairRE) and [CompGCN](https://github.com/malllabiisc/CompGCN). Thanks for their contributions.
