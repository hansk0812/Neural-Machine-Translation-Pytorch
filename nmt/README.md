#### Download EnTamV2 dataset

```
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1454{/en-ta-parallel-v2.tar.gz}
tar xvzf en-ta-parallel-v2.tar.gz
mv en-ta-parallel-v2/corpus.bcn.* .
rmdir en-ta-parallel-v2/
rm en-ta-parallel-v2.tar.gz
```

Follow the instructions to install indic_nlp_library and indic_nlp_resources from github: https://github.com/anoopkunchukuttan

## EnTamV2 stats:

### Without symbols

train set:

English vocabulary size for train set: 58441
Tamil vocabulary size for train set: 330116
Using train set with 172966 sentence pairs

dev set:

English vocabulary size for dev set: 5359
Tamil vocabulary size for dev set: 8912
Using dev set with 1000 sentence pairs

test set:

English vocabulary size for test set: 8166
Tamil vocabulary size for test set: 10178
Using test set with 2000 sentence pairs

### With symbols and english-tamil token disambiguation

train set:

English vocabulary size for train set: 58066
Tamil vocabulary size for train set: 325012
Using train set with 172631 sentence pairs

dev set:

English vocabulary size for dev set: 5375
Tamil vocabulary size for dev set: 8921
Using dev set with 1000 sentence pairs

test set:

English vocabulary size for test set: 8166
Tamil vocabulary size for test set: 15546
Using test set with 2000 sentence pairs

### With symbols and morphemes for tamil

train set:

English vocabulary size for train set: 58066
Tamil vocabulary size for train set: 41134
Using train set with 172631 sentence pairs

dev set:

English vocabulary size for dev set: 5375
Tamil vocabulary size for dev set: 6976
Using dev set with 1000 sentence pairs

test set:

English vocabulary size for test set: 8166
Tamil vocabulary size for test set: 10178
Using test set with 2000 sentence pairs

![tamil_unicode](https://github.com/hansk0812/NMT_repetitions/blob/main/pytorch/morphologically_rich_MT/misc/tamil_unicode.png?raw=true)
![kannada_unicode](https://github.com/hansk0812/NMT_repetitions/blob/main/pytorch/morphologically_rich_MT/misc/kannada_unicode.png?raw=true)


### Other morphologically rich languages: 

1. Kujamaat Jóola - 
  No bilingual corpus data available, found a dictionary here: https://uva.theopenscholar.com/files/kujamaat-joola/files/en-idx.pdf 
  A book on Diola-Fogny grammar available which only studies the morphological properties of the language
2. Hungarian - agglutinative
3. Turkish - morphologically rich
4. Telugu - morphologically rich

#TODO:
Curriculum sampling
