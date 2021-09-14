
#main (these are the ones you have to touch more often)
task = "ae"              #ae (Aspect Extraction) or asc (Aspect Sentiment Classification)
btype = "hsum"           #at (Adversarial Training) or psum or hsum
domain = "laptop"          #laptop or rest

#others
bert = domain + "_pt"    #laptop_pt or rest_pt
run_dir = "pt_" + task   #pt_ae or pt_asc
eval = True              #if True the evaluation will run after the training
runs = 5
train_epochs = 2

#eval
tasks = [task]
berts=["pt"]            #pt stands for PreTrained
domains=[domain]
testing = True

#psum

#hsum

#at_rest
ae_rest_dropout = 0.0
ae_rest_epsilon = 5.0
asc_rest_dropout = 0.5
asc_rest_epsilon = 5.0
#at_laptop
ae_lap_dropout = 0.0
ae_lap_epsilon = 0.2
asc_lap_dropout = 0.4
asc_lap_epsilon = 5.0
