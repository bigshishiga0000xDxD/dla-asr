import editdistance

def calc_cer(target_text, predicted_text) -> float:
    if len(predicted_text) == 0:
        return 1
    return editdistance.eval(target_text, predicted_text) / len(predicted_text)


def calc_wer(target_text, predicted_text) -> float:
    if len(predicted_text) == 0:
        return 1
    return editdistance.eval(target_text.split(), predicted_text.split()) / len(predicted_text.split())
