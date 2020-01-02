from simpletransformers.question_answering import QuestionAnsweringModel
import os
import json

# create QuestionAnsweringMode
print ('Loading DistilBERT language model...')
qnaModel = QuestionAnsweringModel('distilbert','distilbert-base-uncased-distilled-squad', args={'reprocess_input_data': True, 'overwrite_output_dir': True},use_cuda=False)
print ('Loaded DistilBERT language model...')

# train the model
print ('Finetuning started for DistilBERT language model on custom data set...')
qnaModel.train_model('data/train.json')
print ('Finetuning completed for DistilBERT language model on custom data set...')

# evaluate the model
print ('Evaluation of finetuned model started...')
result, text = qnaModel.eval_model('data/train.json')
print (result)
print (text)
print ('Evaluation of finetuned model completed...')