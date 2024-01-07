# Neural-Symbolic Visual Question Answering with Contrastive Explanation (NS-VQA-CE)

Implémetation Python du Github : (NSVQASP : https://github.com/pudumagico/nsvqasp) qui propose une technique de NS-VQA avec CE en ASP (Answer Set Programming) sur le **[CLEVR Dataset : https://cs.stanford.edu/people/jcjohns/clevr/]**

### Publication NSVQASP 2023
**[A Logic-based Approach to Contrastive Explainability for Neurosymbolic Visual Question Answering](https://www.ijcai.org/proceedings/2023/0408.pdf)**
https://www.ijcai.org/proceedings/2023/0408.pdf

<div align="center">
  <img src="img/ce.png" width="750px">
</div>

Basé lui-même sur les travaux de : https://github.com/kexinyi/ns-vqa (qui nous a servi de point de départ)

### Publication NSVQA 2018
**[Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding](https://arxiv.org/abs/1810.02338)**

<div align="center">
  <img src="img/model.png" width="750px">
</div>

<br>
Kexin Yi&ast;, 
[Jiajun Wu](https://jiajunwu.com/)&ast;, 
[Chuang Gan](http://people.csail.mit.edu/ganchuang/), 
[Pushmeet Kohli](https://sites.google.com/site/pushmeet/), 
[Antonio Torralba](http://web.mit.edu/torralba/www/), and
[Joshua B. Tenenbaum](https://web.mit.edu/cocosci/josh.html)
<br>
(* indicates equal contributions)
<br>
In Neural Information Processing Systems (*NeurIPS*) 2018.
<br>

```
@inproceedings{yi2018neural,
  title={Neural-symbolic vqa: Disentangling reasoning from vision and language understanding},
  author={Yi, Kexin and Wu, Jiajun and Gan, Chuang and Torralba, Antonio and Kohli, Pushmeet and Tenenbaum, Joshua B.},
  booktitle={Advances in Neural Information Processing Systems},
  pages={1039--1050},
  year={2018}
}
```

## Prérequis
* Python 3.10.13

## Pour faire tourner le code

Clonez ce répertoire
```
git clone https://github.com/vyktron/ns-vqa-ce.git
```

Créer un environnement et installez tous les packages listés dans `requirements.txt`
```
python3.10 -m venv /path/to/new/virtual/environment
pip install -r requirements.txt
```

Lancez la commande suivante dans votre terminal avec l'environnement activé pour faire tourner l'algorithme sur 10000 questions portant sur 1000 images.
```
python run.py --load_checkpoint_path data/pretrained/question_parser.pt --save_result_path data/reason/results --clevr_val_scene_path scene_parse/images/val_scene.json --clevr_vocab_path data/reason/clevr_h5/clevr_vocab.json
```

## Fonctionnement

### Etape 1 : Détection d'objets

Le détecteur d'objet est un modèle YoloV8n (Yolo version 8, nano = plus petit modèle) développé par ultralytics : (https://github.com/ultralytics/ultralytics)
Il s'agit donc d'une version plus performante et compacte que YoloV5, utilisé dans la publication "NSVQASP 2023"
De plus il est possible de faire tourner ce modèle sur CPU uniquement, contrairement à l'implémentation Mask-R-CNN utilisé dans la publication initiale de 2018.

Le modèle n'est pas présent dans ce répertoire, seuls les résultats des segmentations des 1000 images sont présentes (dans le fichier ```scene_parse/images/val_scene.json```)

#### Exemple

<div align="center">
  <img src="img/train_batch0.jpg" width="750px">
</div>



### Step 2: attribute extraction

The next step is to feed the detected objects into an attribute network to extract their attributes and form abstract representations of the input scenes. First, go to directory
```
cd {repo_root}/scene_parse/attr_net
```
and process the detection result
```
python tools/process_proposals.py \
    --dataset clevr \
    --proposal_path ../../data/mask_rcnn/results/clevr_val_pretrained/detections.pkl \
    --output_path ../../data/attr_net/objects/clevr_val_objs_pretrained.json
```
This will generate an object file at `{repo_root}/data/attr_net/objects/clevr_val_objs_pretrained.json`(17.5MB) which can be loaded by the attribute network.

Then, run attribute extraction
```
python tools/run_test.py \
    --run_dir ../../data/attr_net/results \
    --dataset clevr \
    --load_checkpoint_path ../../data/pretrained/attribute_net.pt \
    --clevr_val_ann_path ../../data/attr_net/objects/clevr_val_objs_pretrained.json \
    --output_path ../../data/attr_net/results/clevr_val_scenes_parsed_pretrained.json
```
The output file `{repo_root}/data/attr_net/results/clevr_val_scenes_parsed_pretrained.json`(15.2MB) stores the parsed scenes that are going to be used for reasoning.

### Step 3: reasoning

We are now ready to perform reasoning. The model first parses the questions into programs, and then run the logic of the programs on the abstract scene representations.
```
cd {repo_root}/reason
```
```
python tools/run_test.py \
    --run_dir ../data/reason/results \
    --load_checkpoint_path ../data/pretrained/question_parser.pt \
    --clevr_val_scene_path ../data/attr_net/results/clevr_val_scenes_parsed_pretrained.json \
    --save_result_path ../data/reason/results/result_pretrained.json
```
The result statistics can be found in the output file `{repo_root}/data/reason/results/result_pretrained.json`. The pretrained model will yield an overall question answering accuracy of 99.8%, same as reported in the paper.

## Train you own model

### Scene parsing

Our scene parser is trained on 4000 rendered CLEVR images. The only difference between the rendered images and the original ones is that the rendered images come with object masks. We refer to this dataset as `CLEVR-mini`, which is downloadable via the `download.sh` script. No images from the original training set are used throughout training. 

1, Train a Mask-RCNN for object detection. We adopt the implementation from [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). Please go to the link for more details.
```
cd {repo_root}/scene_parse/mask_rcnn
```
```
python tools/train_net_step.py \
    --dataset clevr-mini \
    --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    --bs 8 \
    --set OUTPUT_DIR ../../data/mask_rcnn/outputs
```
The program will determine the training schedule based on the number of GPU used. Our code is tested on 4 NVIDIA TITAN Xp GPUs.

2, Run detection on the CLEVR-mini dataset. This step obtains the *proposed* masks of all objects in the dataset, which will be used for training the attribute network. 
```
python tools/test_net.py \
    --dataset clevr_mini \
    --cfg configs/baselines/e2e_mask_rcnn_R-50-FPN_1x.yaml \
    --output_dir ../../data/mask_rcnn/results/clevr_mini \
    --load_ckpt ../../data/mask_rcnn/outputs/ckpt/{checkpoint .pth file}
```

3, Extract the *proposed* CLEVR-mini object masks and pair them to the ground-truth objects via mask IoU
```
cd {repo_root}/scene_parse/attr_net
```
```
python tools/process_proposals.py \
    --dataset clevr \
    --proposal_path ../../data/mask_rcnn/results/clevr_mini/detections.pkl \
    --gt_scene_path ../../data/raw/CLEVR_mini/CLEVR_mini_coco_anns.json \
    --output_path ../../data/attr_net/objects/clevr_mini_objs.json
```

4, Train the attribute network on the CLEVR-mini dataset, using the proposed masks plus ground-truth labels
```
python tools/run_train.py \
    --run_dir ../../data/attr_net/outputs/trained_model \
    --clevr_mini_ann_path ../../data/attr_net/objects/clevr_mini_objs.json \
    --dataset clevr
```

### Reasoning

Go to the "reason" directory
```
cd {repo_root}/reason
```

1, Make sure the raw questions are preprocessed. If you want to pre-train on a subset of questions uniformly sampled over the 90 question families, run
```
python tools/sample_questions.py \
    --n_questions_per_family 3 \
    --input_question_h5 ../data/reason/clevr_h5/clevr_train_questions.h5 \
    --output_dir ../data/reason/clevr_h5
```

2, Pretrain question parser
```
python tools/run_train.py \
    --checkpoint_every 200 \
    --num_iters 5000 \
    --run_dir ../data/reason/outputs/model_pretrain_uniform_270pg \
    --clevr_train_question_path ../data/reason/clevr_h5/clevr_train_3questions_per_family.h5
```

3, Fine-tune question parser
```
python tools/run_train.py \
    --reinforce 1 \
    --learning_rate 1e-5 \
    --checkpoint_every 2000 \
    --num_iters 1000000 \
    --run_dir ../data/reason/outputs/model_reinforce_uniform_270pg \
    --load_checkpoint_path ../data/reason/outputs/model_pretrain_uniform_270pg/checkpoint.pt
```
The output models are stored in the folder `{repo_root}/data/reason/outputs/model_reinforce_uniform_270pg`.
