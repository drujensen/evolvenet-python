import logging


def confusion_matrix(model, data):
    tn = tp = fn = fp = ct = 0

    for inputs, outputs in data:
        actual = model.run(inputs)
        for idx, act in enumerate(actual):
            ct += 1
            if outputs[idx] > 0.5:
                if act > 0.5:
                    tp += 1
                else:
                    fp += 1
            else:
                if act < 0.5:
                    tn += 1
                else:
                    fn += 1

    test_size = len(data)
    logging.info(f"Test size: {test_size}")
    logging.info("----------------------")
    logging.info(f"TN: {tn} | FP: {fp}")
    logging.info("----------------------")
    logging.info(f"FN: {fn} | TP: {tp}")
    logging.info("----------------------")
    accuracy = (tn + tp) / ct
    logging.info(f"Accuracy: {accuracy}")
