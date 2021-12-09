# Data Augmentation
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.char as nac

import math
import pandas as pd


def get_augments(raw_text,aug_choice,max_word_augs_per_record,exclusion_word_list):

    if aug_choice == 'word_synonym_wordnet':
        aug_text=naw.SynonymAug(aug_src="wordnet",aug_max=max_word_augs_per_record,aug_p=1,stopwords=exclusion_word_list).augment(raw_text)   

    elif aug_choice == 'word_antonym':
        aug_text = naw.AntonymAug().augment(raw_text)  

    elif aug_choice == 'word_random':
        aug_text = naw.RandomWordAug(action="swap").augment(raw_text)

    elif aug_choice =='character_keyboard_error':
        aug_text = nac.KeyboardAug().augment(raw_text)


    return(aug_text)

# Function to generate augmentations
def generate_text_augmentations(data,target_column,text_column,index_column,aug_choice='word_synonym_wordnet',max_word_augs_per_record = 10,exclusion_word_list=['wells','fargo']):
    
    #Checking the count of the largest class. All other labels will be augmented to this number
    largest_label_count = data[target_column].value_counts()[0]
    augmenter_info_dict={'Augment_counter':{},'Augments_per_record':{}}
    print(f'Augmenting each label to {largest_label_count} records')

    # Creating dictionary of augments to create for each label and the number of augments per record for each label (This is used
    # to check the number of iterations)
    for index,values in zip(data[target_column].value_counts().index,data[target_column].value_counts().values):
        augmenter_info_dict['Augment_counter'][index] = int(largest_label_count - values)
        augmenter_info_dict['Augments_per_record'][index] = math.ceil(largest_label_count / values)
    
    index_column_list = list()
    aug_text_list = list()
    # Iterate throught each on the labels
    for label in augmenter_info_dict['Augment_counter'].keys():
        print(f'Augmenting text for {label}')
        #Filter label data
        data_labelled = data[data[target_column]==label]
        # Number for iterations for that label
        for iters in range(augmenter_info_dict['Augments_per_record'][label]):
            # Iterate through the filtered dataframe
            for row_items in data_labelled.iterrows():
                # Check counter flag
                if augmenter_info_dict['Augment_counter'][label] > 0:
                    raw_text=row_items[1][text_column]
                    # Generate augmentations
                    augmented_text = get_augments(raw_text,aug_choice,max_word_augs_per_record,exclusion_word_list)
                    # Reduce counter flag
                    augmenter_info_dict['Augment_counter'][label] = augmenter_info_dict['Augment_counter'][label] - 1
                    index_column_list.append(row_items[1][index_column])
                    aug_text_list.append(augmented_text)
                    

    augmented_data = pd.DataFrame({index_column: index_column_list,text_column: aug_text_list})                
    # Match augmented data with other datapoints
    augmented_data_matched=pd.merge(augmented_data,data.drop(text_column,axis=1),how='left',on=index_column)
    augmented_data_matched=augmented_data_matched[data.columns]
    # Concatenate main and augmented datframe
    data_augmented = pd.concat([data,augmented_data_matched],axis=0)
    print(f'Numbe of records before augmentation {data.shape[0]}')
    print(f'Numbe of records after augmentation {data_augmented.shape[0]}')
    return(data_augmented)

