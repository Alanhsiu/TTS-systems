import os
import numpy as np
import torch
import json
from scipy.io import wavfile
import re
from string import punctuation
from g2p_en import G2p
from pypinyin import pinyin, Style

from dlhlp_lib.audio import AUDIO_CONFIG
from dlhlp_lib.vocoders import get_vocoder, BaseVocoder

import Define
from global_setup import setup_data
from tts.systems import get_system
from text import text_to_sequence, symbols

import pyopenjtalk
from prepare_tg_accent import pp_symbols
from convert_label import openjtalk2julius


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon("lexicon/librispeech-lexicon.txt")

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    print("words: ", words)
    for w in words:
        print("W: ", w)
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
            print("50 ", g2p(w))
            print("51 ", filter(lambda p: p != " ", g2p(w)))
            print("52 ", list(filter(lambda p: p != " ", g2p(w))))
        print("phones: ", phones)
    
    print("49 phones ", phones)
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")
    print("53 phones ", phones)

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    print("57 phones", phones)
    sequence = np.array(
        text_to_sequence(
            phones, ['english_cleaners'], lang_id="en"
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text):
    lexicon = read_lexicon("lexicon/pinyin-lexicon-r.txt")

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, [], lang_id="zh"
        )
    )

    return np.array(sequence)

def preprocess_japanese(text:str):
    fullcontext_labels = pyopenjtalk.extract_fullcontext(text)
    phonemes , accents = pp_symbols(fullcontext_labels)
    phonemes = [openjtalk2julius(p) for p in phonemes if p != '']

    print(phonemes)

    phonemes = "{" + "}{".join(phonemes) + "}"
    phonemes = re.sub(r"\{[^\w\s]?\}", "{sp}", phonemes)
    phonemes = phonemes.replace("}{", " ")
    print(phonemes)
    
    sequence = np.array(
        text_to_sequence(
            phonemes, [], lang_id="jp"
        )
    )

    return np.array(sequence)

def build_fastspeech2(ckpt_path: str, data_configs):
    system = get_system("fastspeech2")
    model = system.load_from_checkpoint(data_configs=data_configs, checkpoint_path=ckpt_path)
    model.eval()

    return model


def inference(system, text, spk, lang_id, mel_path, p_control=1.0, e_control=1.0, d_control=1.0):
    text_lens = np.array([len(text)])

    text = torch.from_numpy(text).unsqueeze(0).long()
    text_lens = torch.from_numpy(text_lens)
    max_text_len = max(text_lens)
    spk = torch.from_numpy(spk).long()

    with torch.no_grad():
        emb_texts = system.embedding_model(text.cuda(), lang_id)
        mel = system.model(
            spk.cuda(), emb_texts, text_lens.cuda(), max_text_len.cuda(), 
            p_control=p_control, e_control=e_control, d_control=d_control
        )[1]
        mel = mel[0].detach().cpu().numpy()
     
    with open(mel_path, 'wb') as f:
        np.save(f, mel)


def mel2wav(vocoder: BaseVocoder, mel_path, wav_path):
    mel = torch.from_numpy(np.load(mel_path).T).float().cuda()
    with torch.no_grad():
        wav = vocoder.infer(mel.unsqueeze(0))[0]
    wavfile.write(wav_path, AUDIO_CONFIG["audio"]["sampling_rate"], wav)


if __name__ == "__main__":
    # ==================parameters==================
    ckpt_path = "output/exp_2_FastSpeech2/ckpt/epoch=19-step=50000.ckpt"
    # ckpt_path = "output/exp_2_FastSpeech2_multispeaker/ckpt/epoch=19-step=50000.ckpt"
    data_config = "data_config/LJSpeech-1.1"
    # input = "Is this the real life Is this just fantasy Caught in a landslide no escape from reality Open your eyes look up to the skies and see"
    # # input = "Whether the weather be fine or whether the weather be not"
    input = "Supercalifragilisticexpialidocious."
    # # input = "Ground Control to Major Tom."
    # # # input = "Deep learning is fun."
    spk = "LJSpeech"  # "LJSpeech", "103", "SSB0005", "jsut", "kss"...
    
    # ckpt_path = "output/exp_3_FastSpeech2-JSUT/ckpt/epoch=19-step=50000.ckpt"
    # data_config = "data_config/JSUT"
    # # # input = "おはようごぢいます"
    # spk = "jsut"  # "LJSpeech", "103", "SSB0005", "jsut", "kss"...
    # input = "するとツルはおじいさんの頭の上を三ベん回って"
    # input = "痛みを感じろ、神羅天征！"
    # input = "痛みを感じろ、痛みを考えろ、痛みを受け取れ、痛みを知れ。痛みを知らぬ者に、本当の平和は分からん。俺はヤヒコの痛みを忘れない。ここより、世界に痛みを！神羅天征！"
    # input = "するとツルは、おじいさんの 頭の上を三ベん回って"
    # input = "Is this the real life? Is this just fantasy? Caught in a landslide, no escape from reality. Open your eyes, look up to the skies and see"
    control = {  # Control FastSpeech2
        "p_control": 1.0,
        "e_control": 1.0,
        "d_control": 1.0,
    }
    output_mel_path = "_temp/test4.npy"
    output_wav_path = "_temp/test4.wav"
    vocoder = "HifiGAN"
    # ==================parameters==================
    
    os.makedirs(os.path.dirname(output_mel_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)

    # inference
    from Objects.config import DataConfigReader
    reader = DataConfigReader()
    data_config = reader.read(data_config)
    setup_data([data_config])

    # build model
    vocoder = get_vocoder(vocoder)().cuda()
    system = build_fastspeech2(ckpt_path, [data_config]).cuda()
    system.eval()

    # parser input to model's input format
    if data_config["lang_id"] == "en":
        text = preprocess_english(input)
        print("input: ", input)
        print("text: ", text)
    elif data_config["lang_id"] == "zh":
        text = preprocess_mandarin(input)
    elif data_config["lang_id"] == "jp":
        text = preprocess_japanese(input)
    else:
        raise NotImplementedError
    with open(Define.DATAPARSERS[data_config["name"]].speakers_path, 'r') as f:
        speakers = json.load(f)
    spk = np.array([speakers.index(spk)])

    inference(system, text, spk, data_config["lang_id"], output_mel_path, **control)
    mel2wav(vocoder, output_mel_path, output_wav_path)
