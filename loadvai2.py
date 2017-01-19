import csv
def loadVAI():
     
        with open('dataI_irony1005_1.5.csv','r',encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            i_sd= [row['Irony_SD'] for row in reader]
            i = [row['Irony_Mean'] for row in reader]
            
            print('------------- now is irony------------')
            
    
        with open('dataV_irony1005_1.5.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            v = [row['Valence_Mean'] for row in reader]
            v_sd= [row['Valence_SD'] for row in reader]
            print('--------------- now is valence------------')
            
    
        with open('dataA_irony1005_1.5.csv', 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            a = [row['Arousal_Mean'] for row in reader]
            a_sd= [row['Arousal_SD'] for row in reader]
        #print(a)
        #print(a_sd)
        print(' ----------------now is A-----------')
        
        return v, a, i, i_sd, v_sd, a_sd






