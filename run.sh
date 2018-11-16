cd data
unzip ai_challenger_fsauor2018_testa_20180816.zip
unzip ai_challenger_fsauor2018_validationset_20180816.zip
unzip ai_challenger_fsauor2018_trainingset_20180816.zip


cd ../code
python word2vec.py
python feature.py
python train.py GCAE
python train.py SYN_ATT