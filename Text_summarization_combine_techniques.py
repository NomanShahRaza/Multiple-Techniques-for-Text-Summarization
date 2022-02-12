"""
@author: Noman Raza Shah
"""
#%% 
# ============================= #
# TEXT SUMMARIZATION TECHNIQUES #
# ============================= #

#%% Define all the text summarization function

def gensim_text_summarization(original_text):
    import gensim
    from gensim.summarization import summarize 
    short_summary = summarize(original_text)
    return short_summary

def lex_summary(original_text):
    import sumy
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.kl import KLSummarizer
    import nltk; nltk.download('punkt')
    my_parser = PlaintextParser.from_string(original_text,Tokenizer('english'))
    lex_rank_summarizer = LexRankSummarizer()
    lexrank_summary = lex_rank_summarizer(my_parser.document,sentences_count=3)
    kl_summarizer=KLSummarizer()
    kl_summary=kl_summarizer(my_parser.document,sentences_count=3)
    return lexrank_summary

def KL_summary(original_text):
    import sumy
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.kl import KLSummarizer
    import nltk; nltk.download('punkt')
    my_parser = PlaintextParser.from_string(original_text,Tokenizer('english'))
    kl_summarizer=KLSummarizer()
    kl_summary=kl_summarizer(my_parser.document,sentences_count=3)
    return kl_summary

def transformer_T5_summary(original_text):
    from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
    my_model = T5ForConditionalGeneration.from_pretrained('t5-base')      
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    input_ids=tokenizer.encode(original_text, return_tensors='pt', max_length=512)
    summary_ids = my_model.generate(input_ids)
    t5_summary = tokenizer.decode(summary_ids[0])
    return t5_summary

def transformer_T5_large_summary(original_text):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    model = T5ForConditionalGeneration.from_pretrained("t5-large")
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    inputs = tokenizer.encode("summarize: " + original_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs, 
        max_length=150, 
        min_length=40, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True)
    summary = tokenizer.decode(outputs[0])
    return summary

def bert_summary(original_text):
    from summarizer import Summarizer,TransformerSummarizer
    bert_model = Summarizer()
    summary_bert = ''.join(bert_model(original_text, min_length=60))
    return summary_bert

def GPT2_summary(original_text):
    from summarizer import Summarizer,TransformerSummarizer    
    GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
    summary_GPT2 = ''.join(GPT2_model(original_text, min_length=60))
    return summary_GPT2
    
def XLNet_summary(original_text):
    from summarizer import Summarizer,TransformerSummarizer    
    model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
    summary_XLNet = ''.join(model(original_text, min_length=60))
    return summary_XLNet

def BART_summary(original_text):
    from transformers import pipeline
    # use bart in pytorch
    # using pipeline API for summarization task
    summarization = pipeline("summarization")
    summary_text = summarization(original_text)[0]['summary_text']
    return summary_text

#%%

original_text= """we are working in Artificial intelligence lab. we have few groups who are focus on the area of machine learning and artificial intelligence.
Google tackles the most challenging problems in computer science. Our teams aspire to make discoveries that impact everyone, and core to our approach is sharing 
our research and tools to fuel progress in the field. Our researchers publish regularly in academic journals, release projects as open source, and apply research 
to Google products."""


#%%
original_text = '''
       Scientists say they have discovered a new species of orangutans on Indonesia’s island of Sumatra.
The population differs in several ways from the two existing orangutan species found in Sumatra and the neighboring island of Borneo.
The orangutans were found inside North Sumatra’s Batang Toru forest, the science publication Current Biology reported.
Researchers named the new species the Tapanuli orangutan. They say the animals are considered a new species because of genetic, skeletal and tooth differences.
Michael Kruetzen is a geneticist with the University of Zurich who has studied the orangutans for several years. 
He said he was excited to be part of the unusual discovery of a new great ape in the present day. He noted that most great apes are currently considered endangered or severely endangered.
Gorillas, chimpanzees and bonobos also belong to the great ape species.
Orangutan – which means person of the forest in the Indonesian and Malay languages - is the world’s biggest tree-living mammal. 
The orange-haired animals can move easily among the trees because their arms are longer than their legs. 
They live more lonely lives than other great apes, spending a lot of time sleeping and eating fruit in the forest.
The new study said fewer than 800 of the newly-described orangutans exist. Their low numbers make the group the most endangered of all the great ape species.
They live within an area covering about 1,000 square kilometers. The population is considered highly vulnerable. That is because the environment
 which they depend on is greatly threatened by development.
Researchers say if steps are not taken quickly to reduce the current and future threats, the new species could become extinct “within our lifetime.”
Research into the new species began in 2013, when an orangutan protection group in Sumatra found an injured orangutan in an area far away from the 
other species. The adult male orangutan had been beaten by local villagers and died of his injuries. The complete skull was examined by researchers.
Among the physical differences of the new species are a notably smaller head and frizzier hair. The Tapanuli orangutans also have a different diet and
 are found only in higher forest areas.
There is no unified international system for recognizing new species. But to be considered, discovery claims at least require publication in a major 
scientific publication.
Russell Mittermeier is head of the primate specialist group at the International Union for the Conservation of Nature. He called the finding a 
“remarkable discovery.” He said it puts responsibility on the Indonesian government to help the species survive.
Matthew Nowak is one of the writers of the study. He told the Associated Press that there are three groups of the Tapanuli orangutans that are 
separated by non-protected land.He said forest land needs to connect the separated groups.
In addition, the writers of the study are recommending that plans for a hydropower center in the area be stopped by the government.
It also recommended that remaining forest in the Sumatran area where the orangutans live be protected.
I’m Bryan Lynn. '''

#%%
print("="*70)
print('The Summarized text by using Gensim Technique is "',gensim_text_summarization(original_text), '"')
print("="*70)

#%%
print("="*70)
print('The Summarized text by using LexRank Technique is "', lex_summary(original_text), '"')
print("="*70)

#%%
print("="*70)
print('The Summarized text by using KL Technique is "', KL_summary(original_text), '"')
print("="*70)

#%%
print("="*70)
print('The Summarized text by using Transformer T5 model is "', transformer_T5_summary(original_text), '"')
print("="*70)

#%%
print("="*70)
print('The Summarized text by using Transformer T5 large model is "', transformer_T5_large_summary(original_text), '"')
print("="*70)

#%%
print("="*70)
print('The Summarized text by using BERT model is "',bert_summary(original_text), '"')
print("="*70)

#%%
print("="*70)
print('The Summarized text by using GPT2 model is "',GPT2_summary(original_text), '"')
print("="*70)

#%%
print("="*70)
print('The Summarized text by using XLNet model is "',XLNet_summary(original_text), '"')
print("="*70)

#%%
print("="*70)
print('The Summarized text by using BART model is "',BART_summary(original_text), '"')
print("="*70)










