from __future__ import division
import os, random
import numpy as np
from numpy import * 
from sys import platform
import matplotlib
from scipy import interpolate
import matplotlib.pyplot as plt

#####################################
## NOTE
# Algoritmi da rivedere e aggiornare
#####################################

def FilterRR(RR,BT, winlength=50,last=13,minbpm=24,maxbpm=198):
    """Removes outliers from RR series"""
    
    maxRR=60/minbpm
    minRR=60/maxbpm

    # threshold initialization
    ulast=last #13%
    umean=1.5*ulast #19%

    index=1 # salto il primo
    varPrec=100*abs((RR[1]-RR[0])/RR[0])
    while index<len(RR)-1: #arrivo al penultiimo
        v=RR[max(index-winlength,0):index] #media degli ultimi winlengh valori
        M=np.mean(v)
        varNext=100*abs((RR[index+1]-RR[index])/RR[index+1])
        varMean=100*abs((RR[index]-M)/M)
       
        if ( ( (varPrec < ulast) | #variazione dal precedente
        (varNext < ulast) | # variazione dal successivo
        (varMean < umean) ) # variazione dalla media
        & (RR[index] > minRR) & (RR[index] < maxRR)): # valori ammessi
            index += 1 #tutto ok
        else:
            RR=np.delete(RR, index)
            BT=np.delete(BT, index)
        varPrev=varNext
    return RR,  BT

def smooth(x,window_len=11,window='hanning'):
        if x.ndim != 1:
                raise ValueError, "smooth only accepts 1 dimension arrays."
        if x.size < window_len:
                raise ValueError, "Input vector needs to be bigger than window size."
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:  
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]

def massimi2RR(timesMax, winlength=50, minbpm=24,maxbpm=198):
    # filtra vettore di massimi del segnale BVP
    
    maxRR=60/minbpm
    minRR=60/maxbpm
    
    RR=[0]
    timeRR=[0]
   
    # algoritmo alternativo
    # beat artifacts correction
    ##inizializzazioni
    RR=[(timesMax[1]-timesMax[0])]
    RRvect=RR
    
    timeRR=timesMax[0]
    tprev=timesMax[0]
       
    for i in range(1, winlength): #inizializzo la media RR
        tact=timesMax[i]
        RRcurr=tact-tprev
        
        if RRcurr>minRR and RRcurr<maxRR:
            RR=np.append(RR, RRcurr)
            timeRR=np.append(timeRR, tact)
        RRvect=np.append(RRvect, RR)
        tprev=tact
    
    meanRR=np.mean(RRvect)
    
    tfalse=tact
    soglia=0.4#last # coefficiente di variazione di ogni intervallo RR tollerato rispetto alla RRmedia
    for i in range(winlength, len(timesMax)):
        tact=timesMax[i]
        RRcurr=tact-tprev
        
        if RRcurr<meanRR*(1-soglia): # troppo breve: falso picco?
            tfalse=tact #aspetto (non aggiorno tact) e salvo il falso campione
        elif RRcurr>meanRR*(1+soglia): # troppo lungo ho perso un picco?
            RRcurr=tact-tfalse # provo con l'ultimo falso picco se esiste
            tprev=tact # dopo ripartiro' da questo picco
            
        if RRcurr>meanRR*(1-soglia) and RRcurr<meanRR*(1+soglia): # tutto OK
            RR=np.append(RR, RRcurr) #aggiungo valore RR
            timeRR=np.append(timeRR, tact) #aggiungo istante temporale
            tfalse=tact #aggiorno falso picco
            tprev=tact #aggiorno tprev
        
        RRvect=np.append(RRvect, RRcurr) #aggiungo sempre il valore RR trovato
        meanRR=np.mean(RRvect[-winlength:]) # aggiorno RRmedia
    return RR,  timeRR

def windowRR(RR, WINDSIZE=30, OVERLAP=0): #segmentation di un vettore RR
    BT=np.cumsum(RR)
    
    if OVERLAP==0:
        OVERLAP=WINDSIZE
    tStartREF=np.arange(0, BT[-1], OVERLAP)
    tEndREF=tStartREF+WINDSIZE
    numWindows=len(tEndREF)
    
    indexStart=np.zeros(numWindows) #indici del vettore RR (o BT) da cui partire
    indexEnd=np.zeros(numWindows) #indici del vettore RR (o BT) a cui finire
    
    tStart=np.zeros(numWindows) #istanti a cui partire
    tEnd=np.zeros(numWindows) #istanti a cui finire
    
    nSamples=np.zeros(numWindows)
    for i in xrange(numWindows):
        indexStart[i]=np.argmin(abs(BT-tStartREF[i]))
        indexEnd[i]=np.argmin(abs(BT-tEndREF[i]))
        tStart[i]=BT[indexStart[i]]
        tEnd[i]=BT[indexEnd[i]]
        nSamples[i]= indexEnd[i]-indexStart[i]
    
    return indexStart, indexEnd, tStart, tEnd, nSamples

def findSessions(cry, sleep, BT, winsize=20, subject=''):
    three_class=np.array(abs(sleep-1)+cry)
    three_class_smoothed=smooth(three_class, winsize, window='flat')
    
    if subject!='':
        import matplotlib.pyplot as plt
        
        fig=plt.figure()
#        plt.plot(BT, three_class, 'b')
        plt.plot(BT, three_class_smoothed, 'r')
        plt.xlabel('time')
        plt.ylabel('session')
        plt.yticks([0, 1, 2], ('Sleep', 'Calm', 'Cry'))
        plt.savefig(subject+'.png')
        
    IND_sessionStart=[]
    IND_sessionEnd=[]
    sessionType=[]
    sessionDuration=[]
    prevSession=-1
    t0=BT[0]
    for i in range(len(three_class_smoothed)):
        if three_class_smoothed[i]>=1.5:
            currentSession=2
        elif three_class_smoothed[i]>0.5 and three_class_smoothed[i]<1.5:
            currentSession=1
        else:
            currentSession=0
            
        if currentSession!=prevSession:
            IND_sessionStart=np.hstack([IND_sessionStart, i])
            IND_sessionEnd=np.hstack([IND_sessionEnd, i-1])
            sessionType=np.hstack([sessionType, currentSession])
            sessionDuration=np.hstack([sessionDuration, BT[i-1]-t0])
            
            prevSession=currentSession
            t0=BT[i]
    sessionDuration=np.hstack([sessionDuration[1:], BT[-1]-t0])
    IND_sessionEnd=np.hstack([IND_sessionEnd[1:], i])
    
    return IND_sessionStart, IND_sessionEnd, sessionType,  sessionDuration

def findSessionsCARRING(IND_sessionStart, IND_sessionEnd, sessionType, carring, BT):
    
    IND_sessionStart_CARRING=[]
    IND_sessionEnd_CARRING=[]
    sessionType_CARRING=[]
    sessionDuration_CARRING=[]
    
    IND_sessionStart=IND_sessionStart[sessionType==1]
    IND_sessionEnd=IND_sessionEnd[sessionType==1]
    sessionType=sessionType[sessionType==1]
    
    print len(sessionType)
    
    for i in range(len(IND_sessionStart)):
        prevCarring=str('')
        IND_sessionStart_CARRING_tmp=[]
        IND_sessionEnd_CARRING_tmp=[]
        sessionType_CARRING_tmp=[]
        sessionDuration_CARRING_tmp=[]        
        
        indStart=IND_sessionStart[i]
        indEnd=IND_sessionEnd[i]
        BT_session=BT[indStart:indEnd]
        carring_session=np.array(carring[indStart:indEnd])
        
        prevCarring=str('')
        t0=BT_session[0]
        for j in range(len(BT_session)):
            
            currCarring=str(carring_session[j])
            if currCarring!=prevCarring:
                IND_sessionStart_CARRING_tmp=np.hstack([IND_sessionStart_CARRING_tmp, indStart+j])
                IND_sessionEnd_CARRING_tmp=np.hstack([IND_sessionEnd_CARRING_tmp, indStart+j-1])
                sessionType_CARRING_tmp=np.hstack([sessionType_CARRING_tmp, currCarring])
                sessionDuration_CARRING_tmp=np.hstack([sessionDuration_CARRING_tmp, BT_session[j-1]-t0])
                
                prevCarring=currCarring
                t0=BT_session[j]
        sessionDuration_CARRING_tmp=np.hstack([sessionDuration_CARRING_tmp[1:], BT_session[-1]-t0])
        IND_sessionEnd_CARRING_tmp=np.hstack([IND_sessionEnd_CARRING_tmp[1:], indEnd])
        
        IND_sessionStart_CARRING=np.hstack([IND_sessionStart_CARRING, IND_sessionStart_CARRING_tmp])
        IND_sessionEnd_CARRING=np.hstack([IND_sessionEnd_CARRING, IND_sessionEnd_CARRING_tmp])
        sessionType_CARRING=np.hstack([sessionType_CARRING, sessionType_CARRING_tmp])
        sessionDuration_CARRING=np.hstack([sessionDuration_CARRING, sessionDuration_CARRING_tmp])
        
    return IND_sessionStart_CARRING, IND_sessionEnd_CARRING, sessionType_CARRING,  sessionDuration_CARRING
