# Functions to perform EDA on the data

import numpy as np
import pandas as pd
from IPython.display import display
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
  

class data_explorer:
    def __init__(self,data,text_column,target_column=None):
        self.data=data
        self.text_column=text_column
        self.target_column=target_column

    def get_basic_stats(self):
        word_counts=np.array(self.data[self.text_column].apply(lambda x:len(x.split(' '))))
        chars_counts=np.array(self.data[self.text_column].apply(len))

        out_df=pd.DataFrame(index=['Word count','Character count'],columns=['Min','Percentile 10','Percentile 25','Percentile 50','Percentile 75','Percentile 90','Max','Mean','Std Dev'])
        out_df.iloc[0,0]=min(word_counts)
        out_df.iloc[0,1]=np.percentile(word_counts, 10)
        out_df.iloc[0,2]=np.percentile(word_counts, 25)
        out_df.iloc[0,3]=np.percentile(word_counts, 50)
        out_df.iloc[0,4]=np.percentile(word_counts, 75)
        out_df.iloc[0,5]=np.percentile(word_counts, 90)
        out_df.iloc[0,6]=max(word_counts)
        out_df.iloc[0,7]=word_counts.mean()
        out_df.iloc[0,8]=word_counts.std()

        out_df.iloc[1,0]=min(chars_counts)
        out_df.iloc[1,1]=np.percentile(chars_counts, 10)
        out_df.iloc[1,2]=np.percentile(chars_counts, 25)
        out_df.iloc[1,3]=np.percentile(chars_counts, 50)
        out_df.iloc[1,4]=np.percentile(chars_counts, 75)
        out_df.iloc[1,5]=np.percentile(chars_counts, 90)
        out_df.iloc[1,6]=max(chars_counts)
        out_df.iloc[1,7]=chars_counts.mean()
        out_df.iloc[1,8]=chars_counts.std()

        print(f'Number of text records : {self.data.shape[0]}')
        print(f'Number of unique texts : {self.data[self.text_column].nunique()}')
        print(f'Number of missing texts : {self.data[self.text_column].isna().sum()}')
        print()
        display(out_df)


    def get_word_cloud(self):

        comment_words = ''
        stopwords = set(STOPWORDS)
  
        # iterate through the csv file
        for val in self.data[self.text_column]:
      
            # typecaste each val to string
            val = str(val)
  
            # split the value
            tokens = val.lower().split()
      
            comment_words += " ".join(tokens)+" "

        wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = stopwords,
                    min_font_size = 10).generate(comment_words)
  
        # plot the WordCloud image                       
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()

