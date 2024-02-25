import os
import pdb
import librosa
import soundfile as sf

from funasr import AutoModel

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def split_all(dir):
    # Split all the files in the directory
    for subdir in os.listdir(dir):
        if subdir == ".DS_Store":
            continue
        for file in os.listdir(dir + subdir):
            if file.endswith(".wav"):
                try:
                    res = model.generate(input=dir + subdir + "/" + file)
                    audio, sr = librosa.load(dir + subdir + "/" + file, sr=None)
                    if len(res[0]['value']) == 19 or len(res[0]['value']) == 20:
                        for idx, val in enumerate(res[0]['value']):
                            if int(val[1]) - int(val[0]) < 3.0:
                                # raise warning
                                print('WARNING: less than 3:', subdir, file, int(val[1]) - int(val[0]))
                            else:
                                if not os.path.exists(outdir + subdir):
                                    os.makedirs(outdir + subdir)

                                sf.write(outdir + subdir + "/"  + f"all_split_{idx + 1}.wav",
                                         audio[int(val[0] * sr / 1000):int(val[1] * sr / 1000)], sr)
                    else:
                        print('not 20:', subdir, file, len(res[0]['value']))
                except:
                    print(subdir, file, "failed")

model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")
# wav_file = f"/Users/yizhong/Documents/projects/artivoice/gtr_145_chestall/251/all.wav"
# audio, sr = librosa.load(wav_file, sr=None)
# res = model.generate(input=wav_file)

outdir = "/Users/yizhong/Documents/projects/artivoice/splits_batch1/"
wavdir = "/Users/yizhong/Documents/projects/artivoice/gtr_145_chestall/"
split_all(wavdir)
