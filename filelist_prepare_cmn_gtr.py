# coding: utf-8

# This is for converting Pinyins in filelist to IPAs
# SSB00090212|kai1 che1 % man4 man4 % qian2 xing2 $|1       (1 at the end is the speaker id)
# ---------->
# SB00090212|$kʰˈaɪʈʂʰˈɤ mˈanmˈan tɕʰˈjɛnɕˈiŋ$|X1111111111 44444444 22222222222X|1

import random
from pathlib import Path
from pypinyin import lazy_pinyin, Style
from filelist_prepare_cmn_aishell import pinyin_mapper

gtr_spkid = "0"


def pinyin2IPA(text):
    ipa_items = []
    tmp_strs = text.split(' ')
    for tmp_str in tmp_strs:
        if tmp_str == '$':
            continue
        if tmp_str == '%':
            ipa_items.append([' ', ' '])
            continue
        if not tmp_str[-1].isdigit():
            tmp_str = tmp_str + '5'  # add tone 5 for soft tone if missing in pinyin
        tone = tmp_str[-1]
        pinyin = tmp_str[:-1]
        ipa = pinyin_mapper[pinyin]
        ipa_items.append([ipa, tone * len(ipa)])  # repeated tones for ipa
    return ipa_items


def main():
    indir = Path("/storageNVME/melissa/gtr_125")
    subdirs = list(indir.glob("*"))

    new_lines = []
    with open("Data/gtr_trans.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_s = line.strip().split('|')
            wavfile = line_s[0]
            text = line_s[1]
            pinyin = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)

            ipa_items = pinyin2IPA(" ".join(pinyin))
            ipa_str = ''
            tone_str = ''
            for ipa, tone in ipa_items:
                ipa_str += ipa
                tone_str += tone

            for subdir in subdirs:
                if (subdir / wavfile).exists():
                    new_line = str(subdir/wavfile) + '|$' + ipa_str + '$|X' + tone_str + 'X|' + gtr_spkid
                    new_lines.append(new_line)
                else:
                    print(f"File {wavfile} not found in {subdir}")

    with open("Data/gtr_train.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(new_lines))
    with open("Data/gtr_test.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(new_lines[:100]))


if __name__ == '__main__':
    main()
