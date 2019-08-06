import os, sys
import numpy as np
from scipy import signal
from itertools import product
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import sys
from fun import do_smd, do_dif, find_peaks, analize_peaks, fix_peaks

PMT_num=20
time_samples=1024
start_time = time.time()
path='/home/gerak/Desktop/DireXeno/pulser_190803_46211/'
file=open(path+'out.DXD', 'rb')
event=0
rec_list=[]
Dataframe=pd.DataFrame()
v=850
f=5
wind=15
while event<100000:
    if event%10==0:
        print('Event number {} voltage: {}, frequency: {} wind {}, ({} files per sec).'.format(event, v,f,wind, 100/(time.time()-start_time)))
        start_time = time.time()
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    data=Data[2:1002,:PMT_num]
    smd = do_smd(data, wind)
    trig=np.argmax(np.roll(smd[:,-1],1)-np.roll(smd[:,-1],-1))

    rec=np.recarray(PMT_num, dtype=[
        ('event', ('<i4', 1)),
        ('chn', ('<i4', 1)),
        ('trig', ('<i4', 1)),
        ('init', ('<f4', 1)),
        ('maxi', ('<i4', 1)),
        ('fin', ('<i4', 1)),
        ('height', ('<f8', 1)),
        ('area', ('<f8', 1)),
        ])
    rec['event']=event
    event+=1
    rec['chn']=chn
    rec['trig']=trig
    rec['init']=init
    rec['maxi']=maxi
    rec['fin']=fin
    rec['height']=height
    rec['area']=area
    rec_list.append(rec)

    if len(rec_list)>5000:
        data_frame=pd.DataFrame.from_records(rec_list[0])
        for i in range(1, len(rec_list)):
            if i%100==0:
                print('Creating DF in event number {}, DF number {} ({} files per sec). DF size {} GB'.format(event, i,
                            100/(time.time()-start_time), sys.getsizeof(data_frame)/1e9))
                start_time = time.time()
            data_frame=data_frame.append(pd.DataFrame.from_records(rec_list[i]), ignore_index=True)
        print('!!!!!!!!!!!!!!! Appending to big DF !!!!!!!!!!!!!!!!')
        Dataframe=Dataframe.append(data_frame, ignore_index=True)
        rec_list=[]

    if len(rec_list)>1:
        data_frame=pd.DataFrame.from_records(rec_list[0])
        for i in range(1, len(rec_list)):
            if i%100==0:
                print('Creating DF in voltage {}V friquency {}GHz event number {}, DF number {} ({} files per sec).'.format(v,
                                    f, event, i, 100/(time.time()-start_time)))
                start_time = time.time()
            data_frame=data_frame.append(pd.DataFrame.from_records(rec_list[i]), ignore_index=True)
        print('!!!!!!!!!!!!!!! Appending to big DF !!!!!!!!!!!!!!!!')
        Dataframe=Dataframe.append(data_frame, ignore_index=True)
        rec_list=[]

print('############## Pickeling ###################')
Dataframe.to_pickle(path+'df.pkl')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
