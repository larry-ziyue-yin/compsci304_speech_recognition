from os.path import isabs, join
from pathlib import Path
from torch.utils.data import Dataset


def _read_kaldi_table(path):
    entries = {}
    with open(path, "r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt, value = parts
            entries[utt] = value
    return entries


class AIShellDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size, ascending=False):
        self.path = path
        self.bucket_size = bucket_size

        pairs = []
        for s in split:
            split_dir = Path(join(path, s))
            wav_scp = split_dir / "wav.scp"
            text_path = split_dir / "text"

            assert wav_scp.exists(), "No wav.scp found @ {}".format(wav_scp)
            assert text_path.exists(), "No text found @ {}".format(text_path)

            wav_dict = _read_kaldi_table(str(wav_scp))
            txt_dict = _read_kaldi_table(str(text_path))
            utt_ids = [utt for utt in wav_dict if utt in txt_dict]
            assert len(utt_ids) > 0, "No paired wav/text found @ {}".format(split_dir)

            for utt in utt_ids:
                wav_path = wav_dict[utt]
                if not isabs(wav_path):
                    wav_path = str((split_dir / wav_path).resolve())
                txt = tokenizer.encode(txt_dict[utt])
                pairs.append((wav_path, txt))

        self.file_list, self.text = zip(*[
            (wav_path, txt) for wav_path, txt in sorted(
                pairs, reverse=not ascending, key=lambda x: len(x[1]))
        ])

    def __getitem__(self, index):
        if self.bucket_size > 1:
            index = min(len(self.file_list) - self.bucket_size, index)
            return [
                (f_path, txt)
                for f_path, txt in zip(
                    self.file_list[index:index + self.bucket_size],
                    self.text[index:index + self.bucket_size],
                )
            ]
        return self.file_list[index], self.text[index]

    def __len__(self):
        return len(self.file_list)


class AIShellTextDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size):
        self.path = path
        self.bucket_size = bucket_size

        all_text = []
        for s in split:
            split_dir = Path(join(path, s))
            text_path = split_dir / "text"
            assert text_path.exists(), "No text found @ {}".format(text_path)
            txt_dict = _read_kaldi_table(str(text_path))
            all_text.extend(tokenizer.encode(txt) for txt in txt_dict.values())

        assert len(all_text) > 0, "No text found @ {}".format(path)
        self.text = sorted(all_text, reverse=True, key=lambda x: len(x))

    def __getitem__(self, index):
        if self.bucket_size > 1:
            index = min(len(self.text) - self.bucket_size, index)
            return self.text[index:index + self.bucket_size]
        return self.text[index]

    def __len__(self):
        return len(self.text)
