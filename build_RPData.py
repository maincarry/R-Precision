import argparse
import os
import random
import shutil
import sys

import numpy as np
from PIL import Image
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

NUM_DIR = 50
NUM_IMG_PER_DIR = 600


def generateSample(dir, rand=False):
    count = 0
    res = []

    files = os.listdir(dir)
    if rand:
        random.shuffle(files)

    for each in files:
        if count < NUM_IMG_PER_DIR:
            fname, fexp = os.path.splitext(each)
            fake = Image.open(os.path.join(dir, each)).convert("RGB")

            text = ""
            if args.text != "":
                # TODO: CHANGE THIS TO FIT YOUR NAMING
                txt_file = os.path.join(args.text, os.path.basename(dir), "{}.txt".format(fname.split("-")[0][:-2]))
                assert os.path.isfile(txt_file), txt_file
                sentence_num = int(fname[-1])  # TODO: CHANGE THIS TO FIT YOUR NAMING
                with open(txt_file) as f:
                    all_texts = f.readlines()
                    text = all_texts[sentence_num]

            res.append((fake, text, all_texts))
            count += 1
    if count < NUM_IMG_PER_DIR:
        print("Warning: [{}/{}] Not enought sample at {}.".format(count, NUM_IMG_PER_DIR, dir))
    return res  # return [(img0, txt0, [ts]), (img1, txt1, [ts])...]


def sampleFakeSentence(all_text, exclude_text):
    exclude_text = [x.strip() for x in exclude_text]
    res = []
    while len(res) < 99:
        tmp = random.choice(all_text)
        if tmp not in exclude_text:
            res.append(tmp)
            exclude_text.append(tmp)
    assert len(res) == 99
    return np.array(res)


# Make Directory "RP_DATA"
def saveTestFiles(data, out, all_caps):
    out_path = os.path.join(out, "RP_DATA")
    if os.path.isdir(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)

    count = 0
    for d in tqdm(data):
        img, real, all_txts = d

        # sample 99 fake sentences
        # fakes = np.random.choice(all_caps, 99, replace=False)
        fakes = sampleFakeSentence(all_caps, all_txts)

        fakes = [l + "\n" for l in fakes]
        texts = [real] + fakes
        # save files
        img.save(os.path.join(out_path, "{}.png".format(count)))
        with open(os.path.join(out_path, "{}.txt".format(count)), mode="w") as f:
            f.writelines(texts)
        count += 1
    print("RP Test Folder created at " + out_path)


def read_all_txt(directory):
    all_s = []
    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        if not file.endswith(".txt"):
            continue
        with open(full_path) as f:
            captions = f.read().split('\n')
            for cap in captions:
                if len(cap) == 0 or len(cap) == 1:
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
                all_s.append(" ".join(tokens_new) + "\n")
    return all_s


def main(dirs, out, rand=False, all_caps_file=""):
    global NUM_DIR
    NUM_DIR = min(len(dirs), NUM_DIR)
    if rand:
        random.shuffle(dirs)
    # select first <NUM_DIR> DIR
    dirs = dirs[:NUM_DIR]
    data = []
    for directory in tqdm(dirs):
        data += generateSample(directory)

    print("Making R Precision Test Directory.")
    with open(all_caps_file) as f:
        all_caps = f.readlines()
    all_caps = [l.strip() for l in all_caps]
    saveTestFiles(data, out, all_caps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate RP test data from evaluation output.')
    parser.add_argument('path', type=str, help='Path to image directory')
    parser.add_argument('-c', dest="cap", default='all_texts.txt',
                        help='Optional: specify all texts file. Default to all texts in dataset.')
    parser.add_argument('-d', dest='out', type=str, help='Output directory', default='./')
    parser.add_argument('-t', dest='text', type=str, help='Directory to text data',
                        default='/dataset/birds/text')
    parser.add_argument('-r, --random', dest='rand', action='store_true',
                        help='If set, sample is selected randomly instead of sequentially.')

    args = parser.parse_args()

    # load text file
    if not os.path.isfile(args.cap):
        if args.cap == 'all_texts.txt':
            all_sentences = []
            # get total text
            for subDir in os.listdir(args.text):
                all_sentences += read_all_txt(os.path.join(args.text, subDir))

            with open("all_texts.txt", mode='w') as f:
                f.writelines(all_sentences)
        else:
            print("Specify a valid text file containing all captions, or leave blank to use all texts.")
            quit()

    print(f"All text file: {args.cap}")

    print("Processing {}.\nSample from its subfolders.".format(args.path))
    subDirs = []
    for each in os.listdir(sys.argv[1]):
        full_path = os.path.join(sys.argv[1], each)
        if os.path.isdir(full_path):
            subDirs.append(full_path)

    main(subDirs, args.out, args.rand, args.cap)
