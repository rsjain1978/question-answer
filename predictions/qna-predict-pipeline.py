from transformers import pipeline, QuestionAnsweringPipeline
import pandas as pd

nlp = pipeline('question-answering', model='C:\\MachineLearning\\repos\\personel\\qna-wrkspace\\question-answer\\outputs\\distilbert-base-uncased-distilled-squad-finetuned', tokenizer='bert-base-cased')
q1 = "how did normans grew"
ctx1 = "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."

#q ='how has Aman contributed'
#ctx = "A conditional generation script is also included to generate text from a prompt. The generation script includes the tricks proposed by Aman Rusia to get high-quality generation with memory models like Transformer-XL and XLNet (include a predefined text to make short inputs longer)."

response = nlp(question=q1, context=ctx1, topk=100, max_answer_len=200)

scores = []
starts = []
ends = []
answers = []
for r in response:
    scores.append(r['score'])
    starts.append(r['start'])
    ends.append(r['end'])
    answers.append(r['answer'])

answer_df = pd.DataFrame(data=zip(scores, answers, starts, ends), columns=['score','answer','start','end'])
#print (answer_df)

least_start = min(starts)
max_end = max(ends)

answer = ctx1[least_start:max_end]
print (answer)