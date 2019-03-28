# Front End processing for Chinese Corpus
### Input files and information
#### text files are raw text files, the format should be following:
```
s10_out_0 夢想創意呈獻，心心媽媽講故事，非洲長頸鹿，作者，心心媽媽。
s10_out_1 喺非洲嘅大草原入面有好多唔同野生動物架。
s10_out_10 於是爸爸步步好溫柔咁舐一舐花花嗰肚，就話喇。
```
left side is utterance index and right side is utterance\
**some notes:**
1. we haven't designed prosody prediction(prosodic phrase and prosodic word) moudles for mandarin or cantonese.
2. we also assume we donot have any prosody information
3. But we can consider punctuation as prosody boundary 
# phone segmentation
format of timestamp file:s10_out_0
```angular2
    
```
## kaldi alignment 
## Using Montreal to force-alignment as alignment 
Here it will generate Textgrid file. We need to transform this kind of file to 
our standard input timestamp file.

#### mandarin_front_end.py