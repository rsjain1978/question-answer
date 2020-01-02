from simpletransformers.question_answering import QuestionAnsweringModel
import os
import json

print ('Loading fine tuned DistilBERT language model...')
# create QuestionAnsweringMode
qnaModel = QuestionAnsweringModel('distilbert','outputs', 
                                    args={'reprocess_input_data': True, 
                                    'overwrite_output_dir': False},
                                    use_cuda=False)
print ('Loaded fine tuned DistilBERT language model...')

# prediction using the model
to_predict=[{
    'context': "lex Wright has 13 years of investment experience. He joined Fidelity in 2001 as a European equity research analyst, successively covering building materials, alcoholic beverages, leisure, emerging European and African banks and UK small-cap stocks. He became portfolio manager of the Fidelity UK Smaller Companies Fund in 2008. He continues to manage this fund alongside the Fidelity Special Situations Fund and the Fidelity Special Values PLC, which he started managing in 2012.",
    'qas': [{
        'question': "Who managed fidelity special situations fund",'id':'0', ## Right
        'question': "When did lex joined fidelity",'id':'1', ## Right
        'question': "which role did lex joined fidelity",'id':'2', ## Right
        'question': "how many funds does lex manages",'id':'3'  ##Wrong
    }]
}]
print ('Starting prediction...')
prediction = qnaModel.predict(to_predict)
print (prediction)
print ('Completed prediction...')