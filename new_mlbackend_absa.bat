label-studio-ml init my-ml-backend-absa --script models/bert_absa.py:BertABSA
cp models/bert_absa.py my-ml-backend-absa/bert_absa.py
cp models/absa_data_utils.py my-ml-backend-absa/absa_data_utils.py
cp models/run_config.py my-ml-backend-absa/run_config.py
cp models/modelconfig.py my-ml-backend-absa/modelconfig.py
robocopy models/pt_model my-ml-backend-absa/pt_model /E
robocopy models/berts my-ml-backend-absa/berts /E

label-studio-ml start .\my-ml-backend-absa
