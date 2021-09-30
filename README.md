# Label Studio for BERTforABSA

Code for my degree thesis (in italian) [A dataset labeling support system for Aspect-Based Sentiment Analysis
](https://github.com/Redelyx/BERT-for-ABSA/files/7260749/tesi_Alice_Cipriani.pdf)
<br/>

**Transfer learning for NLP models by annotating your textual data without any additional coding.**

This package provides a ready-to-use container that links together:

- [Label Studio](https://github.com/heartexlabs/label-studio) as annotation frontend
- [BERTforABSA models](https://github.com/Redelyx/BERT-for-ABSA) as machine learning backend for NLP
- BERTforABSA is based on [Hugging Face's transformers](https://github.com/huggingface/transformers)

<br/>

![labelstudio-bertforabsa](https://user-images.githubusercontent.com/32637807/135485737-b8d3d540-bf66-410b-b817-26590aa63e57.png)

### Quick Usage

##### Install Label Studio and other dependencies

```bash
pip install -r requirements.txt
```
##### Download Bert models
Place laptop and restaurant post-trained BERTs into ```pt_model/laptop_pt``` and ```pt_model/rest_pt```, respectively. The post-trained Laptop weights can be download [here](https://drive.google.com/file/d/1io-_zVW3sE6AbKgHZND4Snwh-wi32L4K/view?usp=sharing) and restaurant [here](https://drive.google.com/file/d/1TYk7zOoVEO8Isa6iP0cNtdDFAUlpnTyz/view?usp=sharing).

See [BERTforABSA](https://github.com/Redelyx/BERT-for-ABSA) to obtain the fine-tuned models from training. 
The two \*.pt files (one for Aspect Extraction, one for Aspect Sentiment Classification), have to be renamed by the following standard: 
```model_task_domain.pt```
where, model can be hsum, psum or at, task is ae or asc, domain can be rest or laptop.
for example:
```
hsum_ae_laptop.pt
hsum_asc_laptop.pt
```

##### Create ML backend for ABSA 
```bash
label-studio-ml init my-ml-backend-ae --script models/bert_absa.py
cp models/absa_data_utils.py my-ml-backend-ae/absa_data_utils.py
cp models/pick_bert.py my-ml-backend-ae/pick_bert.py
cp models/modelconfig.py my-ml-backend-ae/modelconfig.py
robocopy models/pt_model my-ml-backend-ae/pt_model /E
robocopy models/berts my-ml-backend-ae/berts /E
```

You can also do the two tasks separately:
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
label-studio-ml start my-ml-backend-name
```
If you want to do the two tasks separately you have to create two projects on Label Studio and two backends, one for AE, one for ASC.

##### Start Label Studio with ML backend connection
```bash
label-studio start my-project-name --init --ml-backend http://localhost:9090
```

The browser opens at `http://localhost:8080`. Upload your data on **Import** page and retrieve your predictions.<br/>
See my other repo [BERTforABSA](https://github.com/Redelyx/BERT-for-ABSA) to train bert-based models.


[Click here](https://labelstud.io/guide/ml.html) to read more about how to use Machine Learning backend and build Human-in-the-Loop pipelines with Label Studio


## Other useful instructions
You can find explanations for these instructions here:

[Set up your labeling interface](https://labelstud.io/guide/setup.html)

[Get data into Label Studio](https://labelstud.io/guide/tasks.html)


### Set up your labeling interface on Label Studio

This is the code for the ABSA labeling interface:
```html
<View>
  <Labels name="label" toName="text">
    <Label value="positive" background="#00ff33"/>
    <Label value="negative" background="#ff0000"/>
    <Label value="neutral" background="#FFC069"/>
  </Labels>
  <Text name="text" value="$ner"/>
</View>
```

This is the code for the AE labeling interface:
```html
<View>
  <Labels name="label" toName="text">
    <Label value="B" background="#fd8326"/>
    <Label value="I" background="#fff570"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>

```
This is the code for the ASC labeling interface:
```html
<View>
  <Labels name="label" toName="text">
    <Label value="positive" background="green"/>
    <Label value="negative" background="#ff0d00"/>
    <Label value="neutral" background="#f79b55"/>
  </Labels>
  <Header value="Sentence"/>
  <Text name="text" value="$sentence"/>
  <Header value="Aspect"/>
  <Text name="text1" value="$term"/>
</View>
```

### Import dataset
For ABSA you can import a \*.txt file with one sentence per line.<br/>
For AE task you can import a \*.txt file with one sentence per line.<br/>
For ASC task you can import a \*.tsv file similiar to this:<br/>
```
sentence	term
The screen is nice, side view angles are pretty good	screen
Applications respond immediately (not like the tired MS applications).	Applications
i also love having the extra calculator number set up on the keyboard.	calculator number
```
