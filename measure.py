
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('waveglow/')

from itertools import cycle
import numpy as np
import scipy as sp
from scipy.io.wavfile import write, read
import pandas as pd
import librosa
import torch

from hparams import create_hparams
from model import Tacotron2, load_model
from waveglow.denoiser import Denoiser
from layers import TacotronSTFT
from data_utils import TextMelLoader, TextMelCollate
from text import cmudict, text_to_sequence
from mellotron_utils import get_data_from_musicxml

from train import *
from mcd.dtw import dtw
from mcd.metrics import eucCepDist

def panner(signal, angle):
    angle = np.radians(angle)
    left = np.sqrt(2)/2.0 * (np.cos(angle) - np.sin(angle)) * signal
    right = np.sqrt(2)/2.0 * (np.cos(angle) + np.sin(angle)) * signal
    return np.dstack((left, right))[0]



def load_mel(path, stft):
    audio, sampling_rate = librosa.core.load(path, sr=hparams.sampling_rate)
    audio = torch.from_numpy(audio)
    if sampling_rate != hparams.sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(
            sampling_rate, stft.sampling_rate))
    audio_norm = audio / hparams.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = melspec.cuda()
    return melspec


def measure(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Handles all the validation scoring and printing"""
    stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length,
                        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                        hparams.mel_fmax)

    mellotron = load_model(hparams).cuda().eval()
    mellotron.load_state_dict(torch.load(checkpoint_path)['state_dict'])

    waveglow_path = '/media/arsh/New Volume/Models/speech/waveglow_256channels_v4.pt'
    waveglow = torch.load(waveglow_path)['model'].cuda().eval()
    denoiser = Denoiser(waveglow).cuda().eval()

    arpabet_dict = cmudict.CMUDict('data/cmu_dictionary')
    audio_paths = 'filelists/libritts_train_clean_100_audiopath_text_sid_atleast5min_val_filelist.txt'
    dataloader = TextMelLoader(audio_paths, hparams)
    datacollate = TextMelCollate(1)

    speaker_ids = TextMelLoader("filelists/libritts_train_clean_100_audiopath_text_sid_shorterthan10s_atleast5min_train_filelist.txt", hparams).speaker_ids
    speakers = pd.read_csv('filelists/libritts_speakerinfo.txt', engine='python',header=None, comment=';', sep=' *\| *', 
                           names=['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME'])
    speakers['MELLOTRON_ID'] = speakers['ID'].apply(lambda x: speaker_ids[x] if x in speaker_ids else -1)
    female_speakers = cycle(
        speakers.query("SEX == 'F' and MINUTES > 20 and MELLOTRON_ID >= 0")['MELLOTRON_ID'].sample(frac=1).tolist())
    male_speakers = cycle(
        speakers.query("SEX == 'M' and MINUTES > 20 and MELLOTRON_ID >= 0")['MELLOTRON_ID'].sample(frac=1).tolist())

    file_idx = 0
    MEL_DTW = []
    TPP_DTW = []
    RAND_DTW = []
    logSpecDbConst = 10.0 / math.log(10.0) * math.sqrt(2.0)
    while file_idx < len(dataloader):
        audio_path, text, sid = dataloader.audiopaths_and_text[file_idx]

        # get audio path, encoded text, pitch contour and mel for gst
        text_encoded = torch.LongTensor(text_to_sequence(text, hparams.text_cleaners, arpabet_dict))[None, :].cuda()    
        pitch_contour = dataloader[file_idx][3][None].cuda()
        mel = load_mel(audio_path, stft)
        fs, audio = read(audio_path)

        # load source data to obtain rhythm using tacotron 2 as a forced aligner
        x, y = mellotron.parse_batch(datacollate([dataloader[file_idx]]))

        with torch.no_grad():
            # get rhythm (alignment map) using tacotron 2
            mel_outputs, mel_outputs_postnet, gate_outputs, rhythm, gst, tpse_gst = mellotron.forward(x)
            rhythm = rhythm.permute(1, 0, 2)
        speaker_id = next(female_speakers) if np.random.randint(2) else next(male_speakers)
        speaker_id = torch.LongTensor([speaker_id]).cuda()

        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, _ = mellotron.inference_noattention(
                (text_encoded, mel, speaker_id, pitch_contour, rhythm), with_tpse=False)
        with torch.no_grad():
            audio_mel = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:, 0]

        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, _ = mellotron.inference_noattention(
                (text_encoded, mel, speaker_id, pitch_contour, rhythm), with_tpse=True)
        with torch.no_grad():
            audio_tpp = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:, 0]
        
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, _ = mellotron.inference_noattention(
                (text_encoded, np.random.randint(0, 9), speaker_id, pitch_contour, rhythm), with_tpse=False)
        with torch.no_grad():
            audio_rand = denoiser(waveglow.infer(mel_outputs_postnet, sigma=0.8), 0.01)[:, 0]
        audio = np.pad(audio, 128)

        MEL_DTW.append(logSpecDbConst * np.log(dtw(audio_mel.data.cpu().numpy(), audio, eucCepDist)[0]))
        TPP_DTW.append(logSpecDbConst * np.log(dtw(audio_tpp.data.cpu().numpy(), audio, eucCepDist)[0]))
        RAND_DTW.append(logSpecDbConst * np.log(dtw(audio_rand.data.cpu().numpy(), audio, eucCepDist)[0]))
        print(MEL_DTW[-1], TPP_DTW[-1], RAND_DTW[-1])
        print("MEL DTW, Mean: ", np.mean(MEL_DTW), " SD: ", np.std(MEL_DTW))
        print("TPP DTW, Mean: ", np.mean(TPP_DTW), " SD: ", np.std(TPP_DTW))
        print("RAND DTW, Mean: ", np.mean(RAND_DTW), " SD: ", np.std(RAND_DTW))
        file_idx += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    measure(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
