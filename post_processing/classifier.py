def spam_ham_suspicions(predictions):
    """
    This function is used to classify the emails as spam, ham or suspicious.
    """
    if predictions > 0.6:
        return '정상 메일'
    elif predictions < 0.4:
        return '스팸 메일'
    else:
        return '의심스러운 이메일'
    