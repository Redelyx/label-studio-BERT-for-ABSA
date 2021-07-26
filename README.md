# Label Studio for Transformers

[Website](https://labelstud.io/) • [Docs](https://labelstud.io/guide) • [Twitter](https://twitter.com/heartexlabs) • [Join Slack Community <img src="https://go.heartex.net/docs/images/slack-mini.png" width="18px"/>](https://docs.google.com/forms/d/e/1FAIpQLSdLHZx5EeT1J350JPwnY2xLanfmvplJi6VZk65C2R4XSsRBHg/viewform?usp=sf_link)

<br/>

**Transfer learning for NLP models by annotating your textual data without any additional coding.**

This package provides a ready-to-use container that links together:

- [Label Studio](https://github.com/heartexlabs/label-studio) as annotation frontend
- [Hugging Face's transformers](https://github.com/huggingface/transformers) as machine learning backend for NLP

<br/>

[<img src="https://raw.githubusercontent.com/heartexlabs/label-studio-transformers/master/images/codeless.png" height="500">](https://github.com/heartexlabs/label-studio-transformers)

### Quick Usage

##### Install Label Studio and other dependencies

```bash
pip install -r requirements.txt
```
##### Download Bert models
Place laptop and restaurant post-trained BERTs into ```pt_model/laptop_pt``` and ```pt_model/rest_pt```, respectively. The post-trained Laptop weights can be download [here](https://drive.google.com/file/d/1io-_zVW3sE6AbKgHZND4Snwh-wi32L4K/view?usp=sharing) and restaurant [here](https://drive.google.com/file/d/1TYk7zOoVEO8Isa6iP0cNtdDFAUlpnTyz/view?usp=sharing).

##### Create ML backend for ABSA Aspect Extraction
```bash
label-studio-ml init my-ml-backend-ae --script models/bert_ae.py
cp models/absa_data_utils.py my-ml-backend-ae/absa_data_utils.py
cp models/pick_bert.py my-ml-backend-ae/pick_bert.py
cp models/run_config.py my-ml-backend-ae/run_config.py
cp models/modelconfig.py my-ml-backend-ae/modelconfig.py
robocopy models/pt_model my-ml-backend-ae/pt_model /E
robocopy models/berts my-ml-backend-ae/berts /E
```

##### Create ML backend for ABSA Aspect Sentiment Classification
```bash
label-studio-ml init my-ml-backend-asc --script models/bert_asc.py
cp models/absa_data_utils.py my-ml-backend-asc/absa_data_utils.py
cp models/pick_bert.py my-ml-backend-asc/pick_bert.py
cp models/run_config.py my-ml-backend-asc/run_config.py
cp models/modelconfig.py my-ml-backend-asc/modelconfig.py
robocopy models/pt_model my-ml-backend-asc/pt_model /E
robocopy models/berts my-ml-backend-asc/berts /E
```

##### Start ML backend at http://localhost:9090
```bash
label-studio-ml start my-ml-backend-ae
```
or
```bash
label-studio-ml start my-ml-backend-asc
```

##### Start Label Studio with ML backend connection
```bash
label-studio start my-annotation-project --init --ml-backend http://localhost:9090
```

The browser opens at `http://localhost:8080`. Upload your data on **Import** page ~then annotate by selecting **Labeling** page.
Once you've annotate sufficient amount of data, go to **Model** page and press **Start Training** button. Once training is finished, model automatically starts serving for inference from Label Studio, and you'll find all model checkpoints inside `my-ml-backend/<ml-backend-id>/` directory.~
(Not available yet)


[Click here](https://labelstud.io/guide/ml.html) to read more about how to use Machine Learning backend and build Human-in-the-Loop pipelines with Label Studio

## License

This software is licensed under the [Apache 2.0 LICENSE](/LICENSE) © [Heartex](https://www.heartex.ai/). 2020

<img src="https://github.com/heartexlabs/label-studio/blob/master/images/opossum_looking.png?raw=true" title="Hey everyone!" height="140" width="140" />
