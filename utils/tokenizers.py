import unicodedata
from utils.vocab import Vocab


def tokenier(args, text):
    vocab = Vocab(args)

    input_token = ['[cls]']
    input_ids = [2]
    token_types = [0]
    input_mask = [1]

    for i, sentence in enumerate(text):
        sen_token = text_split(args, sentence)
        sen_token.append('[sep]')

        input_token.extend(sen_token)
        input_ids.extend([vocab.word_2_idx(token) for token in sen_token])
        token_types.extend([i]*len(sen_token))
        input_mask.extend([1]*len(sen_token))
    return input_token, input_ids, token_types, input_mask


# split the text into tokens
def text_split(args, text):
    text_token = []
    text = text.lower()
    for char in text:
        if args.rm_punc:
            if is_punctuation(char):
                text_token.append(' ')
            elif is_whitespace(char) or is_control(char):
                text_token.append(' ')
            elif is_chinese_char(char):
                text_token.extend([' ', char, ' '])
            else:
                text_token.append(char)
        else:
            if is_punctuation(char):
                text_token.extend([' ', char, ' '])
            elif is_whitespace(char) or is_control(char):
                text_token.append(' ')
            elif is_chinese_char(char):
                text_token.extend([' ', char, ' '])
            else:
                text_token.append(char)
    text_token = ''.join(text_token).split()
    return text_token


# check whether the char is the chinese char
def is_chinese_char(char):
    cp = ord(char)
    if (
            (cp >= 0x4E00 and cp <= 0x9FFF) or
            (cp >= 0x3400 and cp <= 0x4DBF) or
            (cp >= 0x20000 and cp <= 0x2A6DF) or
            (cp >= 0x2A700 and cp <= 0x2B73F) or
            (cp >= 0x2B740 and cp <= 0x2B81F) or
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or
            (cp >= 0x2F800 and cp <= 0x2FA1F)
    ):
      return True


# Checks whether char is a punctuation character.
def is_punctuation(char):
  cp = ord(char)
  if (
          (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
          (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)
  ):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False


# Checks whether the char is a whitespace character.
def is_whitespace(char):
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


# Checks whether the char is a control character.
def is_control(char):
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True
  return False


if __name__ == '__main__':
    from args import init_arg_parser
    args = init_arg_parser()

    text = ['kagb科技馆ksjhgFHJS J DLO, . L ; KDHFBSKGBHjb较好iuiiu吧hj后狂']
    input_token, input_ids, token_types, input_mask = tokenier(args, text)















