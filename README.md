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

Le modèle a été entrainé sur un dataset nommé "CLEVR-Mini" qui contient 4000 images et les coordonnées de la segmentation de chaque objet.
Le modèle est capable de reconnaître 96 classes : (Taille_Couleur_Texture_Forme)
* "Large_Red_Metal_Sphere": 0,
* "Large_Red_Metal_Cube": 1,
* "Large_Red_Metal_Cylinder": 2,
* "Large_Red_Rubber_Sphere": 3,
* ...

Ensuite à partir des objets détectés et leurs coordonnées nous sommes capables d'obtenir le tableau des attributs de la scène.

#### Performance

<div align="center">
  <img src="img/results.png" width="750px">
</div>

Le modèle converge effectivement avec notamment une **précision de 99,6%** pour les classes attribuées

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

### Contrastive Explanation

La dernière partie consiste à trouver une modification de la scène (bouger un objet, changer sa couleur, sa forme...) pour engendrer une modification de la réponse

#### Exemple 

How many red objects are small shiny blocks or small rubber balls ?  

<div align="center">
  <img src="scene_parse/images/val/CLEVR_val_000998.png" width="750px">
</div>

* Image : 998
* Predicted answer: 1
* Ground truth answer: 1
* Modified answer: 0
* Cost: 11
* Description: change_size 4 : small -> large
* Modified scene: [{'id': '998-0', 'position': [1.2073033273220062, 1.4945631957054137, 0.7], 'color': 'brown', 'material': 'rubber', 'shape': 'cube', 'size': 'large'}, {'id': '998-1', 'position': [1.6233381736278534, -2.141068323254585, 0.35], 'color': 'gray', 'material': 'rubber', 'shape': 'cylinder', 'size': 'small'}, {'id': '998-2', 'position': [-1.154744883775711, -1.1294089430570604, 0.35], 'color': 'blue', 'material': 'rubber', 'shape': 'cylinder', 'size': 'small'}, {'id': '998-3', 'position': [-1.693931791782379, 1.8313643437623979, 0.7], 'color': 'cyan', 'material': 'rubber', 'shape': 'cylinder', 'size': 'large'}, {'id': '998-4', 'position': [0.6711750781536102, -2.412887197732925, 0.7], 'color': 'red', 'material': 'metal', 'shape': 'cube', 'size': **'large'**}, {'id': '998-5', 'position': [0.3581746220588684, -0.5633923751115799, 0.35], 'color': 'blue', 'material': 'rubber', 'shape': 'cylinder', 'size': 'small'}, {'id': '998-6', 'position': [-3.7221895015239714, -0.08891469061374657, 0.35], 'color': 'brown', 'material': 'metal', 'shape': 'cylinder', 'size': 'small'}, {'id': '998-7', 'position': [-2.910327101945877, 0.874869332909584, 0.35], 'color': 'gray', 'material': 'rubber', 'shape': 'cylinder', 'size': 'small'}, {'id': '998-8', 'position': [-0.5011349391937258, -2.678761799931526, 0.35], 'color': 'gray', 'material': 'rubber', 'shape': 'cylinder', 'size': 'small'}]
