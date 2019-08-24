from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os
import pprint
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tqdm import tqdm

import torch.utils.data as data
from PIL import Image
from nltk.tokenize import RegexpTokenizer

import torchvision.transforms as transforms
from miscc.config import cfg, cfg_from_file
from model import RNN_ENCODER, CNN_ENCODER

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

WORDS_NUM = 18

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RP. Use build_RPData.py to prepare RP_DATA directory.')
    parser.add_argument('data_dir', type=str, help='Specify path to RP_DATA directory.')
    parser.add_argument('caption_pkl', type=str, help='Specify path to captions.pickle, from AttnGan')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='config file used by DAMSM',
                        default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

# used to feed the R-Precision Test
class EvalRPDataset(data.Dataset):
    def __init__(self, data_dir, caption_pkl_path):
        assert os.path.isdir(data_dir)
        assert os.path.isfile(caption_pkl_path)
        self.length = len([name for name in os.listdir(data_dir)
                           if (os.path.isfile(os.path.join(data_dir, name))) and (name.endswith("png"))])
        print("Total of {} sample found.".format(self.length))

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.data_dir = data_dir
        self.caption_pkl = caption_pkl_path
        # self.captions = [[[s0],[s1],...], [file1] ...]

        self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir)



    def load_captions(self, data_dir):
        def tokenize(in_captions):
            res = []
            for cap in in_captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())
                # print('tokens', tokens)
                if len(tokens) == 0:
                    print('cap', cap)
                    continue

                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                res.append(tokens_new)
            return res  # [(tokens0...), (tokens1...),...]

        all_captions = []
        for i in range(self.length):
            cap_path = os.path.join(data_dir, "{}.txt".format(i))
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                # tokenize
                captions = tokenize(captions)
                all_captions.append(captions)

        return all_captions




    def load_text_data(self, data_dir):
        # entry from init
        filepath = self.caption_pkl
        # filepath = os.path.join(cfg.DATA_DIR, 'captions.pickle')

        if not os.path.isfile(filepath):
            print("Needs original captions.pickle file for indexing.")
            print(filepath)
            quit()
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f, encoding='latin1')
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)

        all_captions = self.load_captions(data_dir)
        all_captions_new = []
        for file in all_captions:
            new_file_texts = []
            for sent in file:
                rev = []
                for w in sent:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                # rev.append(0)  # do not need '<end>' token
                new_file_texts.append(rev)
            all_captions_new.append(new_file_texts)

        return all_captions_new, ixtoword, wordtoix, n_words

    def get_caption(self, fileno):
        # a list of indices for a sentence
        all_caps = self.captions[fileno]
        ret_cap = []
        ret_caplen = []

        for cap in all_caps:
            sent_caption = np.asarray(cap).astype('int64')
            if (sent_caption == 0).sum() > 0:
                print('ERROR: do not need END (0) token', sent_caption)
            num_words = len(sent_caption)
            # pad with 0s (i.e., '<end>')
            x = np.zeros((WORDS_NUM, 1), dtype='int64')
            x_len = num_words
            if num_words <= WORDS_NUM:
                x[:num_words, 0] = sent_caption
            else:
                ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
                np.random.shuffle(ix)
                ix = ix[:WORDS_NUM]
                ix = np.sort(ix)
                x[:, 0] = sent_caption[ix]
                x_len = WORDS_NUM
            ret_cap.append(x)
            ret_caplen.append(x_len)


        return ret_cap, ret_caplen

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, "{}.png".format(index))
        image = self.norm(Image.open(image_path).convert('RGB'))
        image = image.unsqueeze(0)
        captions, cap_lens = self.get_caption(index)
        return image, captions, cap_lens


    def __len__(self):
        return self.length


def prepare_data(data):
    imgs, captions, captions_lens = data

    # convert list to tensor
    captions = torch.from_numpy(np.array(captions))
    captions_lens = torch.from_numpy(np.array(captions_lens))

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    # get where the real sentence got sorted to
    real_index = (sorted_cap_indices == 0).nonzero()

    real_imgs = []
    for i in range(len(imgs)):
        # imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    # sent_indices = sent_indices[sorted_cap_indices]
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens, real_index]


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def evaluateSimilarity(dataset, cnn_model, rnn_model):
    # data: image, captions, cap_lens
    cnn_model.eval()
    rnn_model.eval()

    c_success = 0
    c_total = 0
    for i in tqdm(range(len(dataset))):
        img, captions, cap_lens, real_index = prepare_data(dataset[i])

        _, sent_code = cnn_model(img[-1].unsqueeze(0))
        hidden = rnn_model.init_hidden(100)
        _, sent_emb = rnn_model(captions, cap_lens, hidden)

        sim = cosine_similarity(sent_code, sent_emb)

        success = (sim.max() == sim[real_index])
        if bool(success):
            c_success += 1
        c_total += 1

    rp = c_success / c_total
    return rp


def build_models():
    # build model ############################################################
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 0
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)

        istart = cfg.TRAIN.NET_E.rfind('_') + 8
        iend = cfg.TRAIN.NET_E.rfind('.')
        start_epoch = cfg.TRAIN.NET_E[istart:iend]
        start_epoch = int(start_epoch) + 1
        print('start_epoch', start_epoch)
    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    # now = datetime.datetime.now(dateutil.tz.tzlocal())
    # timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    # output_dir = '../output/%s_%s_%s' % \
    #     (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader
    batch_size = cfg.TRAIN.BATCH_SIZE
    assert os.path.isfile(args.caption_pkl), "Specify captions.pickle file"
    dataset = EvalRPDataset(args.data_dir, args.caption_pkl)

    assert dataset

    # Train
    text_encoder, image_encoder, labels, start_epoch = build_models()
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)

    print("RP: {}".format(evaluateSimilarity(dataset, image_encoder, text_encoder)))
