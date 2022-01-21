import pandas as pd
import numpy as np
import nlpaug.augmenter.word as nlpaw
from tqdm import tqdm



def augment_sentence(sentence, aug, num_threads):
    """""""""
    Constructs a new sentence via text augmentation.
    
    Input:
        - sentence:     A string of text
        - aug:          An augmentation object defined by the nlpaug library
        - num_threads:  Integer controlling the number of threads to use if
                        augmenting text via CPU
    Output:
        - A string of text that been augmented
    """""""""
    return aug.augment(sentence, num_thread=num_threads)
    


def augment_text(df, aug, num_threads, num_times):
    """""""""
    Takes a pandas DataFrame and augments its text data.
    
    Input:
        - df:            A pandas DataFrame containing the columns:
                                - 'comment_text' containing strings of text to augment.
                                - 'isToxic' binary target variable containing 0's and 1's.
        - aug:           Augmentation object defined by the nlpaug library.
        - num_threads:   Integer controlling number of threads to use if augmenting
                         text via CPU
        - num_times:     Integer representing the number of times to augment text.
    Output:
        - df:            Copy of the same pandas DataFrame with augmented data 
                         appended to it and with rows randomly shuffled.
    """""""""
    
    # Get rows of data to augment
    to_augment = df[df['isToxic']==1]
    to_augmentX = to_augment['comment_text']
    to_augmentY = np.ones(len(to_augmentX.index) * num_times, dtype=np.int8)
    
    # Build up dictionary containing augmented data
    aug_dict = {'comment_text':[], 'isToxic':to_augmentY}
    for i in tqdm(range(num_times)):
        augX = [augment_sentence(x, aug, num_threads) for x in to_augmentX]
        aug_dict['comment_text'].extend(augX)
    
    # Build DataFrame containing augmented data
    aug_df = pd.DataFrame.from_dict(aug_dict)
    
    return df.append(aug_df, ignore_index=True).sample(frac=1, random_state=42)
    

    
# Define nlpaug augmentation object 
aug10p = nlpaw.ContextualWordEmbsAug(model_path='distilbert-base-german-cased', aug_min=1, aug_p=0.1, action="substitute")

# Upsample minority class ('isToxic' == 1) to create a roughly 50-50 class distribution
# balanced_df = augment_text(downsampled_df, aug10p, num_threads=8, num_times=3)