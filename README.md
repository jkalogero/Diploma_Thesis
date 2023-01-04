# Graph Neural Networks with External Knowledge for Visual Dialog

Author: Ioannis Kalogeropoulos

Supervisor: [Prof. Alexandros Potamianos](https://www.ece.ntua.gr/en/staff/188)

[Thesis page](http://artemis.cslab.ece.ntua.gr:8080/jspui/handle/123456789/18425?locale=en)

We study the effectiveness of Graph Neural Networks on the task of Visual Dialog. Towards achieving interesting architectures and great results, we experiment on two axes. Firstly, we study various Fusion Methods. In a wide range of Machine Learning problems, we encounter the problem of combining different types of information extracted from various sources. The fusion method used to combine the different modalities is a fundamental design choice of the model and a crucial factor towards the achievement of better results. We experimented on a few sets of different methods and selected the best one for our model. Subsequently, we introduce External Knowledge. The task of Visual Dialog doesn’t require by itself the use of external knowledge. Nevertheless, introducing external knowledge has been proved effective in many tasks of Machine Learning and especially in the field of Natural Language Processing. As a result it has drawn a lot of research interest through the last years and has been applied to a wide variety of similar tasks. Hence, we attempt to introduce external knowledge to our approach and experiment with a few ways of exploiting the extra information. In our experiments we adapt the fusion methods of our baseline and utilize them for fusing the three modalities of our model. We further experiment on the encoding of the External Knowledge. Specifically, we examine the use of one or multiple types of relations of the knowledge graph as well as different methods of aggregating the external information. By conducting a number of experiments, we are able to draw interesting conclusions about the impact of introducing External Knowledge to our model. Specifically, by surpassing the implemented baseline using two different methods, we conclude that it is beneficial for the overall performance. Moreover, we demonstrate this impact by using two types of decoders. The consistency of the results using both decoders highlights the impact of the different encoders. Finally, from our results, we come to the conclusion that the simplest models with less parameters were able to perform better towards encoding the External Knowledge Graph.
<p align="center">
  <img src="images/proposed.png" alt="Model architecture." width="1000" />
</p>

  * [Requirements](#Requirements)
  * [Data](#Data)
  * [Training](#training)
  * [Evaluation](#evaluation)
  * [Acknowledgements](#acknowledgements)



Requirements
----------------------
This code is implemented using PyTorch. To install the necessary dependencies execute:
```sh
conda create -n visdialch python=3.6
conda activate visdialch  # activate the environment and install all dependencies
cd Diploma_Thesis/
pip install -r requirements.txt
```


Data
----------------------

1. Download the VisDial v1.0 dialog json files and images from [here][1].
2. Download the word counts file for VisDial v1.0 train split from [here][2]. 
They are used to build the vocabulary.
3. Use Faster-RCNN to extract image features from [here][3].
4. Use Large-Scale-VRD to extract visual relation embedding from [here][4].
6. Generate ELMo word vectors from [here][5].
7. Download pre-trained GloVe word vectors from [here][6].

Training
--------


Train the model by executing:

```sh
python train.py --config-yml configs/gcn.yml --gpu-ids 0 1
```

Some useful flags of training script are:

| Syntax      | Description |
| :---        |    ----:   |
| `--config-yml`      | Path to a config file listing reader, model and solver parameters.|
| `--gpu-ids`      | List of ids of GPUs to use. |
| `--overfit`      | Overfit model on 5 examples, meant for debugging.   |
| `--validate`   | Whether to validate on val split after every epoch.|
| `--save-dirpath` | Path of directory to create checkpoint directory and save checkpoints. |
| `--load-pthpath` | To continue training, path to .pth file of saved checkpoint. |

Please refer to the training script [training.py][8] to see the all the available flags.

### Saving model checkpoints

This script will save model checkpoints at every epoch as per path specified by `--save-dirpath`.

### Logging

Use [Tensorboard][8] for logging training progress. 

In order to see the resulted logs:
1. Activate tensorboard:

```sh
tensorboard --logdir /path/to/save_dir --port 8008
``` 
2. Visit `localhost:8008` using the browser.



Evaluation
----------

Evaluation of a trained model checkpoint can be done as follows:
To evaluate a trained model, locate the desired checkpoint, e.g. `my_checkpoint.pth` and execute:
```sh
python evaluate.py --config-yml /path/to/config.yml --load-pthpath /path/to/my_checkpoint.pth --split val --gpu-ids 0
```

The flag `--split` specifies which split to evaluate upon.

Please refer to the evaluation script [evaluate.py][11] to see the all the available flags.





[1]: https://visualdialog.org/data
[2]: https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/visdial_1.0_word_counts_train.json
[3]: https://github.com/peteanderson80/bottom-up-attention
[4]: https://github.com/jz462/Large-Scale-VRD.pytorch
[5]: https://allennlp.org/elmo
[7]: https://github.com/stanfordnlp/GloVe
[8]: https://github.com/jkalogero/Diploma_Thesis/blob/main/train.py
[9]: https://www.github.com/lanpa/tensorboardX
[10]: https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
[11]: https://github.com/jkalogero/Diploma_Thesis/blob/main/evaluate.py