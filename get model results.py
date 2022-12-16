import json
import os 
import numpy as np
import pandas as pd
from minicons import scorer

path_to_model_stimuli=os.getenv("path_to_model_stimuli")

with open(path_to_model_stimuli, 'r') as file:
    stimuli = json.load(file)

all_stim = stimuli['data']
indices = [all_stim[i]['idx'] for i in range(len(all_stim))]
sentences = [all_stim[i]['sentence'][:-3] for i in range(len(all_stim))]
followups = [all_stim[i]['followup'] for i in range(len(all_stim))]
ftypes = [all_stim[i]['ftype'] for i in range(len(all_stim))]
stypes = [all_stim[i]['stype'] for i in range(len(all_stim))]

gpt2small_minicons = scorer.IncrementalLMScorer('gpt2')
gpt2small_responses = [sum(x) for x in gpt2small_minicons.compute_stats(gpt2small_minicons.prime_text(sentences, followups))]
print("Results for GPT2-Small Compiled")
del(gpt2small_minicons)

gpt2med_minicons = scorer.IncrementalLMScorer('gpt2-medium')
gpt2med_responses = [sum(x) for x in gpt2med_minicons.compute_stats(gpt2med_minicons.prime_text(sentences, followups))]
print("Results for GPT2-Medium Compiled")
del(gpt2med_minicons)

gpt2large_minicons = scorer.IncrementalLMScorer('gpt2-large')
gpt2large_responses = [sum(x) for x in gpt2large_minicons.compute_stats(gpt2large_minicons.prime_text(sentences, followups))]
print("Results for GPT2-Large Compiled")
del(gpt2large_minicons)

gpt2xl_minicons = scorer.IncrementalLMScorer('gpt2-xl')
gpt2xl_responses = [sum(x) for x in gpt2xl_minicons.compute_stats(gpt2xl_minicons.prime_text(sentences, followups))]
print("Results for GPT2-XL Compiled")
del(gpt2xl_minicons)

gpt2_responses = pd.DataFrame(list(zip(indices, sentences, followups, ftypes, stypes, gpt2small_responses, gpt2med_responses, gpt2large_responses, gpt2xl_responses)), columns =['idx', 'sentence', 'followup', 'ftype', 'stype', 
                                                                                                                                                               'gpt2small_response', 'gpt2med_response', 'gpt2large_response', 'gpt2xl_response'])
gpt2_responses.to_csv("gpt2_small_to_xl_responses.csv")
print("CSV Ready")

opt125m_minicons = scorer.IncrementalLMScorer('facebook/opt-125m')
opt125m_responses = [sum(x) for x in opt125m_minicons.compute_stats(opt125m_minicons.prime_text(sentences, followups))]
print("Results for OPT-125m Compiled")
del(opt125m_minicons)


opt350m_minicons = scorer.IncrementalLMScorer('facebook/opt-350m')
opt350m_responses = [sum(x) for x in opt350m_minicons.compute_stats(opt350m_minicons.prime_text(sentences, followups))]
print("Results for OPT-350m Compiled")
del(opt350m_minicons)

opt1b_minicons = scorer.IncrementalLMScorer('facebook/opt-1.3b')
opt1b_responses = [sum(x) for x in opt1b_minicons.compute_stats(opt1b_minicons.prime_text(sentences, followups))]
print("Results for OPT-1.3B Compiled")
del(opt1b_minicons)

opt3b_minicons = scorer.IncrementalLMScorer('facebook/opt-2.7b')
opt3b_responses = [sum(x) for x in opt3b_minicons.compute_stats(opt3b_minicons.prime_text(sentences, followups))]
print("Results for OPT-2.7B Compiled")
del(opt3b_minicons)

opt7b_minicons = scorer.IncrementalLMScorer('facebook/opt-6.7b')
opt7b_responses = [sum(x) for x in opt7b_minicons.compute_stats(opt7b_minicons.prime_text(sentences, followups))]
print("Results for OPT-6.7b Compiled")
del(opt7b_minicons)

opt13b_minicons = scorer.IncrementalLMScorer('facebook/opt-13b')
opt13b_responses = [sum(x) for x in opt13b_minicons.compute_stats(opt13b_minicons.prime_text(sentences, followups))]
print("Results for OPT-13b Compiled")
del(opt13b_minicons)

opt30b_minicons = scorer.IncrementalLMScorer('facebook/opt-30b')
opt30b_responses = [sum(x) for x in opt30b_minicons.compute_stats(opt30b_minicons.prime_text(sentences, followups))]
print("Results for OPT-30b Compiled")
del(opt30b_minicons)

opt_responses = pd.DataFrame(list(zip(indices, sentences, followups, ftypes, stypes, opt125m_responses, opt350m_responses, opt1b_responses, opt3b_responses, opt7b_responses, opt13b_responses, opt30b_responses)), columns =['idx', 'sentence', 'followup', 'ftype', 'stype', 
                                                                                                                                                           'opt125m_response','opt350m_response', 'opt1b_response', 'opt3b_response', 'opt7b_response', 'opt13b_response', 'opt30b_response'])
opt_responses.to_csv("opt125m_to_30b_responses.csv")


## For GPT-3, Slightly Different Method:
import openai 
openai.api_key=os.getenv("OPENAI_API_KEY")
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

SFs = [sentence + " " + followup for sentence, followup in zip(sentences, followups)]
GPT3_response_full = openai.Completion.create(engine="davinci",
             prompt=SFs,
             max_tokens=0,
             temperature=0.0,
             logprobs=0,
             echo=True,
         )
sentences_tokenized = tokenizer(sentences)['input_ids']
preamble_lengths = [len(tokenized_sentence) for tokenized_sentence in sentences_tokenized]
tokenwise_logprobs = [GPT3_response_full['choices'][i]['logprobs']['token_logprobs'][preamble_lengths[i]:] for i in range(len(preamble_lengths))]
gpt3_responses = [sum(x) for x in tokenwise_logprobs]

gpt3_responses = pd.DataFrame(list(zip(indices, sentences, followups, ftypes, stypes, gpt3_responses)), columns =['idx', 'sentence', 'followup', 'ftype', 'stype', 
                                                                                                                                                           'gpt3_response'])
gpt3_responses.to_csv("gpt3_responses.csv")
