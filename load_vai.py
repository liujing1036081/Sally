import csv

"""
# to load the true annotated value of VAI
"""

def loadVAI(dim):
    if dim=='I':
        with open('./irony_1005/dataI_irony1005_1.5.csv','r',encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            column = [row['Irony_Mean'] for row in reader]
            print('------------- now is irony1005------------')


    if dim == 'V':
         with open('./irony_1005/dataV_irony1005_1.5.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            column = [row['Valence_Mean'] for row in reader]
            print('--------------- now is valence1005------------')

    if dim == 'A':
         with open('./irony_1005/dataA_irony1005_1.5.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            column = [row['Arousal_Mean'] for row in reader]
            print(' ----------------now is arousal1005-----------')
    return column






