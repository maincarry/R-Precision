# R-Precision
A R-Precision evaluation module for AttnGAN based model, using the procedure outlined in the paper [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) by Tao Xu and others. The module calculates the R-Precision score using a special data directory called RPData. This dataset can be generated from evaluated checkpoint directory using [build_RPdata.py](https://github.com/maincarry/R-Precision/blob/master/build_RPData.py).

## Requirement
The script is written in Python 3. However, with little modification, running with Python 2 is also possible.

The image encoder and text encoder follows the same format as AttnGAN. Modify the model structure as necessary.

## How to use
Paste **eval_RP.py** into your model code directory. It requires model.py and miscc/config.py.

To evaluate the RP score for a certain checkpoint:
- Evaluate the checkpoint. AttnGAN example:
`python3 train.py --gpu 0 --cfg cfg/eval_bird.yml`

- Use build_RPData.py to build RPData directory from evaluated images
`python3 build_RPData.py /netG_epoch_xxx/valid/single -t /dataset/birds/text`

*You may need to change build_RPData.py to fit your naming pattern.*
- Use eval_RP to evaluate the RP score
`python3 eval_RP.py ./RP_DATA -c /dataset/birds/captions.pickle`


## Reference
AttnGAN: [https://github.com/taoxugit/AttnGAN](https://github.com/taoxugit/AttnGAN)
