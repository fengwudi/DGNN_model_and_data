import re
import numpy as np

def WcsLog():
    epoch, time, val_ap, val_auc =[], [], [], []
    with open("./wikipedia.log") as f:
        for line in f.readlines():
            # print(line)
            match_epoch = re.search(r"Epoch (.*)  ,  Loss = (.*), Val AUC (.*) Test AUC (.*)", line)
            if match_epoch!=None:
                epoch.append(int(match_epoch.group(1)))
                continue
            
            match_time = re.search(r"average time (.*) s, memory (.*) MB", line)
            if match_time!=None:
                time.append(float(match_time.group(1)))
                continue
            
            match_ap_auc = re.search(r"val auc (.*), ap (.*), recall (.*), acc (.*)", line)
            if match_ap_auc!=None:
                val_ap.append(float(match_ap_auc.group(2)))
                val_auc.append(float(match_ap_auc.group(1)))
                continue
    print(epoch)
    print(time)
    print(val_auc)
    print(val_ap)
    # np.savetxt('output.txt',np.array(epoch),np.array(time),np.array(val_ap),np.array(val_auc))
 
if __name__ =="__main__":
    WcsLog()