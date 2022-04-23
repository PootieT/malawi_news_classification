import numpy as np 

def mixup(text, classes, alpha=1.0):
    """
    Randomly mixes the given list of text embeddings with each other
    
    :param text: The text embeddings to be mixed up
    :param classes: 
    """

    # Generate random indices to shuffle the text embeddings
    indices = np.random.permutation(len(text))
    shuffled_text = text[indices]
    shuffled_classes = classes[indices]
    
    # Generate text embedding weight (minimum 0.4 and maximum 0.6)
    lam = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)
    print(f'lambda: {lam}')
    
    # Weighted Mixup
    mixedup_text = lam*text + (1 - lam)*shuffled_text
    
    mixedup_classes = []
    for text, classes, s_classes in zip(text, classes, shuffled_classes):
        mixedup_classes.append(classes + s_classes)
    
    return mixedup_text, mixedup_classes, indices.numpy()