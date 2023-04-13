import os
models = [ "biolm", "biolm+i2b2_2019", "clinical_bert", "clinical_bert+i2b2_2019"]
header = ['finetune'] + models
header = '\t'.join(header)
with open('commandline_fold.txt', 'w') as fileC:
    with open('finetune_res.txt', 'w') as fileW:        
        fileW.write(header + '\n')
        for fold in ["0", "1", "2", "3", "4", "data_v3"]:           
            dev_data = "corpus/" + fold + "/dev/"
            line = [fold]
            for model in models:
                command = 'python3 src/predict.py '
                result_dir = "experiments/" + fold + "/" + model + "/"
                max_score = 0.0
                best = 0
                for file in os.listdir("experiments/" + fold + "/finetune-128/" + model + "/model/"):
                    # print(file)
                    temp = file.split("_")
                    score = temp[-1][:-3]
                    # print(score)
                    if max_score < float(score):
                        max_score = float(score)
                        best = 128
                for file in os.listdir("experiments/" + fold + "/finetune-256/" + model + "/model/"):
                    # print(file)
                    temp = file.split("_")
                    score = temp[-1][:-3]
                    # print(score)
                    if max_score < float(score):
                        max_score = float(score)
                        best = 256
                # print(line)
                line += [str(max_score), str(best)]
                best_dir = "experiments/" + fold + "/finetune-" + str(best) + "/" + model 
                model_dir = best_dir + "/model/"
                print(best_dir)
                params_dir = best_dir + "/" + model + ".params_best"
                print(result_dir)
                print(model_dir)
                print(params_dir)               
                seed, max_seq = '0', '0'
                with open (best_dir + '/' + model + ".params_optimised.txt") as fParam:
                    for pline in fParam:
                        cols = pline.rstrip().split(':')
                        if cols[0] == 'seed':
                            seed = cols[1]
                            print (seed)
                        if cols[0] == 'max_seq':
                            max_seq = cols[1]
                assert seed != '0' , "Seed = 0"
                assert max_seq != '0', "Max_seq = 0"
                log_file = result_dir + 'dev_log.txt'
                command_dev = command + " --seed {} --max_seq {}  --result_dir {} \
                    --model_dir {} \
                    --params_dir {}  --test_data {} \
                    > {}".format(seed,  max_seq, result_dir + "predict-dev/", model_dir, params_dir, dev_data, log_file)
                log_file = result_dir + 'test_log.txt'
                command_test = command + " --seed {} --max_seq {}  --result_dir {} \
                    --model_dir {} \
                    --params_dir {} \
                    > {}".format(seed,  max_seq, result_dir + "predict-test/", model_dir, params_dir, log_file)
                fileC.write(command_dev + '\n')
                fileC.write(command_test + '\n')
            fileW.write('\t'.join(line))
            fileW.write('\n')
            
    
        
           
