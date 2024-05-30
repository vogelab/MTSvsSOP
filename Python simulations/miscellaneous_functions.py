import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import find_peaks

def fig5(f5A,f5B,f5C_stimulus,f5D_stimulus):
    print("Plotting Figure 5")
    
    fig5, ax = plt.subplots(2, 2, figsize=(12, 10),sharex=True,sharey=True)

    pkA = find_peaks(f5A.A1)[0]
    pkB = find_peaks(f5B.A1)[0]
    pkC = find_peaks(f5C_stimulus.A1)[0]
    pkD = find_peaks(f5D_stimulus.A1)[0]
    
    ax[0, 0].plot(f5A.A1, label="A1s (Response)")
    for i in [pkA[0], pkA[-2], pkA[-1]]:
        ax[0, 0].text(i,f5A.A1[i],f'{f5A.A1[i]:.2f}')
    ax[0, 0].plot(f5A.A2,"--", label="A2s (Memory)")
    ax[0,0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    ax[0, 1].plot(f5B.A1)
    for i in [pkB[0], pkB[-2], pkB[-1]]:
        ax[0, 1].text(i,f5B.A1[i],f'{f5B.A1[i]:.2f}')
    ax[0, 1].plot(f5B.A2,"--")
    
    ax[1, 0].plot(f5C_stimulus.A1)
    for i in [pkC[0], pkC[-2], pkC[-1]]:
        ax[1, 0].text(i,f5C_stimulus.A1[i],f'{f5C_stimulus.A1[i]:.2f}')
    ax[1, 0].plot(f5C_stimulus.A2,"--")
    
    ax[1, 1].plot(f5D_stimulus.A1)
    for i in [pkD[0], pkD[-2], pkD[-1]]:
        ax[1, 1].text(i,f5D_stimulus.A1[i],f'{f5D_stimulus.A1[i]:.2f}')    
    ax[1, 1].plot(f5D_stimulus.A2,"--")
    
    ax[0, 0].set_title("A) ISI=10 moments (no context-stimulus associations)")
    ax[0, 1].set_title("B) ISI=60 moments (no context-stimulus associations)")
    
    ax[1, 0].set_title("C) ISI=10 moments (context-stimulus associations)")
    ax[1, 1].set_title("D) ISI=60 moments (context-stimulus associations)")

    ax[-1, 0].set_xlabel("Simulated Time")
    ax[-1, 1].set_xlabel("Simulated Time")
    ax[0, 0].set_ylabel("Simulated Values")
    ax[1, 0].set_ylabel("Simulated Values")

    ax[0,0].set_ylim(0,1)
    ax[0,1].set_ylim(0,1)
    ax[1,0].set_ylim(0,1)
    ax[1,1].set_ylim(0,1)

    fig5.suptitle("Figure 5",fontsize=16)
    plt.tight_layout()
    plt.show()

def fig6(f6A,f6B,f6C,f6D):
    print("Plotting Figure 6")
    
    fig6, ax = plt.subplots(2, 2, figsize=(12, 10),sharex=True,sharey=True)

    pkA = find_peaks(f6A.X[1])[0]
    pkB = find_peaks(f6B.X[1])[0]
    pkC = find_peaks(f6C.X[2])[0]
    pkD = find_peaks(f6D.X[2])[0]
    
    ax[0, 0].plot(f6A.X[1], label="X2 (Response)")
    for i in [pkA[0], pkA[-2], pkA[-1]]:
        ax[0, 0].text(i,f6A.X[1][i],f'{f6A.X[1][i]:.2f}')
    ax[0, 0].plot(f6A.V[0],"--", label="V1 (Memory)")
    ax[0,0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    ax[0, 1].plot(f6B.X[1])
    for i in [pkB[0], pkB[-2], pkB[-1]]:
        ax[0, 1].text(i,f6B.X[1][i],f'{f6B.X[1][i]:.2f}')
    ax[0, 1].plot(f6B.V[0],"--")

    ax[1, 0].plot(f6C.X[2], label="X3 (Response)")
    for i in [pkC[0], pkC[-2], pkC[-1]]:
        ax[1, 0].text(i,f6C.X[2][i],f'{f6C.X[2][i]:.2f}')
    ax[1, 0].plot(f6C.V[1],"--", label="V2 (Memory)")
    ax[1,0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    
    ax[1, 1].plot(f6D.X[2])
    for i in [pkD[0], pkD[-2], pkD[-1]]:
        ax[1, 1].text(i,f6D.X[2][i],f'{f6D.X[2][i]:.2f}')    
    ax[1, 1].plot(f6D.V[1],"--")
    
    ax[0, 0].set_title("A) ISI=10 moments (one unit)")
    ax[0, 1].set_title("B) ISI=60 moments (one unit)")
    ax[1, 0].set_title("C) ISI=10 moments (two units)")
    ax[1, 1].set_title("D) ISI=60 moments (two units)")

    ax[-1, 0].set_xlabel("Simulated Time")
    ax[-1, 1].set_xlabel("Simulated Time")
    ax[0, 0].set_ylabel("Simulated Values")
    ax[1, 0].set_ylabel("Simulated Values")

    ax[0,0].set_ylim(0,1)
    ax[0,1].set_ylim(0,1)
    ax[1,0].set_ylim(0,1)
    ax[1,1].set_ylim(0,1)

    fig6.suptitle("Figure 6",fontsize=16)
    plt.tight_layout()
    plt.show()

def fig7(f7B,f7C, f7D,isi2_stimuli,isi10_stimuli,isi30_stimuli,isi60_stimuli,time):
    print("Plotting Figure 7")
    
    fig7, ax = plt.subplots(2, 2, figsize=(12, 10),sharex=True,sharey=True)

    isi = ["2", "10", "30", "60"]

    count_isi2 = trial_count(isi2_stimuli,time)
    count_isi10 = trial_count(isi10_stimuli,time)
    count_isi30 = trial_count(isi30_stimuli,time)
    count_isi60 = trial_count(isi60_stimuli,time)

    count = [count_isi2, count_isi10, count_isi30, count_isi60]

    rankin_broster = rankin_and_broster_7()

    for i,rb in enumerate(rankin_broster):
        ax[0, 0].plot(rb,".-" ,label=f"ISI = {isi[i]}")
    ax[0,0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4)

    for i in range(4):
        pkSOP = find_max_response(f7B[i][0].A1, count[i])[1]
        pkFB = find_max_response(f7C[i].X[2], count[i])[1]
        pkFF = find_max_response(f7D[i].X[2], count[i])[1]

        pkSOP = pkSOP/np.max(pkSOP)
        pkFB = pkFB/np.max(pkFB)
        pkFF = pkFF/np.max(pkFF)
        
        max_SOP = np.concatenate([pkSOP[0:30], [np.nan], pkSOP[-4:]])
        max_MTS_FB = np.concatenate([pkFB[0:30], [np.nan], pkFB[-4:]])
        max_MTS_FF = np.concatenate([pkFF[0:30], [np.nan], pkFF[-4:]])

        ax[0,1].plot(max_SOP,".-")
        ax[1,0].plot(max_MTS_FB, ".-")
        ax[1,1].plot(max_MTS_FF, ".-")

    ax[0, 0].set_title("A) Rankin & Broster (1992)")
    ax[0, 1].set_title("B) SOP")
    ax[1, 0].set_title("C) MTS-Feedback")
    ax[1, 1].set_title("D) MTS-Feedforward")    
    
    ax[-1, 0].set_xlabel("Trial")
    ax[-1, 1].set_xlabel("Trial")

    ax[-1, 0].set_xticks([i for i in range(0,35)])
    ax[-1, 0].set_xticklabels([str(i) for i in range(1,31)] + [None, "Test 30", "Test 600", "Test 1200", "Test 1800"],  rotation='vertical')
    ax[-1, 1].set_xticks([i for i in range(0,35)])
    ax[-1, 1].set_xticklabels([str(i) for i in range(1,31)] + [None, "Test 30", "Test 600", "Test 1200", "Test 1800"],  rotation='vertical')

    ax[0, 0].set_ylabel("Response")
    ax[1, 0].set_ylabel("Response")
    
    fig7.suptitle("Figure 7",fontsize=16)
    plt.tight_layout()
    plt.show()

def fig8(f7B,f7C, f7D,isi2_stimuli,isi10_stimuli,isi30_stimuli,isi60_stimuli,time):
    print("Plotting Figure 8")
    
    fig8, ax = plt.subplots(2, 2, figsize=(12, 10),sharex=True,sharey=True)

    isi = ["2", "10", "30", "60"]

    count_isi2 = trial_count(isi2_stimuli,time)
    count_isi10 = trial_count(isi10_stimuli,time)
    count_isi30 = trial_count(isi30_stimuli,time)
    count_isi60 = trial_count(isi60_stimuli,time)

    count = [count_isi2, count_isi10, count_isi30, count_isi60]

    rankin_broster = rankin_and_broster_8()

    for i,rb in enumerate(rankin_broster):
        ax[0, 0].plot(rb,".-" ,label=f"ISI = {isi[i]}")
    ax[0,0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=4)

    for i in range(4):
        pkSOP = find_max_response(f7B[i][0].A1, count[i])[1]
        pkFB = find_max_response(f7C[i].X[2], count[i])[1]
        pkFF = find_max_response(f7D[i].X[2], count[i])[1]

        xhab_SOP = np.mean(pkSOP[57:60]) -  np.mean(pkSOP[57:60])
        t1_SOP = pkSOP[60] - np.mean(pkSOP[57:60])
        t2_SOP = pkSOP[61] - np.mean(pkSOP[57:60])
        t3_SOP = pkSOP[62] - np.mean(pkSOP[57:60])
        t4_SOP = pkSOP[63] - np.mean(pkSOP[57:60])
        SOP = np.array([xhab_SOP, t1_SOP, t2_SOP, t3_SOP, t4_SOP]) 

        xhab_FB = np.mean(pkFB[57:60]) -  np.mean(pkFB[57:60])
        t1_FB = pkFB[60] - np.mean(pkFB[57:60])
        t2_FB = pkFB[61] - np.mean(pkFB[57:60])
        t3_FB = pkFB[62] - np.mean(pkFB[57:60])
        t4_FB = pkFB[63] - np.mean(pkFB[57:60])
        MTS_FB = np.array([xhab_FB, t1_FB, t2_FB, t3_FB, t4_FB]) 

        xhab_FF = np.mean(pkFF[57:60]) -  np.mean(pkFF[57:60])
        t1_FF = pkFF[60] - np.mean(pkFF[57:60])
        t2_FF = pkFF[61] - np.mean(pkFF[57:60])
        t3_FF = pkFF[62] - np.mean(pkFF[57:60])
        t4_FF = pkFF[63] - np.mean(pkFF[57:60])
        MTS_FF = np.array([xhab_FF, t1_FF, t2_FF, t3_FF, t4_FF]) 
        
        ax[0,1].plot(SOP,".-")
        ax[1,0].plot(MTS_FB, ".-")
        ax[1,1].plot(MTS_FF, ".-")

    ax[0, 0].set_title("A) Rankin & Broster (1992)")
    ax[0, 0].axhline(y=0, color='black', linestyle='--')
    
    ax[0, 1].set_title("B) SOP")
    ax[0, 1].axhline(y=0, color='black', linestyle='--')
    
    ax[1, 0].set_title("C) MTS-Feedback")
    ax[1, 0].axhline(y=0, color='black', linestyle='--')
    
    ax[1, 1].set_title("D) MTS-Feedforward")    
    ax[1, 1].axhline(y=0, color='black', linestyle='--')
    
    ax[-1, 0].set_xticks([i for i in range(0,5)])
    ax[-1, 0].set_xticklabels(["X hab", "Test 30", "Test 600", "Test 1200", "Test 1800"])
    ax[-1, 1].set_xticks([i for i in range(0,5)])
    ax[-1, 1].set_xticklabels(["X hab", "Test 30", "Test 600", "Test 1200", "Test 1800"])
   
    ax[0, 0].set_ylabel("Response")
    ax[1, 0].set_ylabel("Response")

    fig8.suptitle("Figure 8",fontsize=16)
    plt.tight_layout()
    plt.show()

def fig9(f9B,f9C,f9D,isi2_stimuli60,isi2_stimuli86400,isi16_stimuli60,isi16_stimuli86400,time):
    print("Plotting Figure 9")
    
    fig9, ax = plt.subplots(2, 2, figsize=(12, 10))

    count_isi2_60 = [trial_count(isi2_stimuli60[i],time) for i in range(6)]
    count_isi2_86400 = [trial_count(isi2_stimuli86400[i],time) for i in range(6)]

    count_isi16_60 = [trial_count(isi16_stimuli60[i],time) for i in range(6)]
    count_isi16_86400 = [trial_count(isi16_stimuli86400[i],time) for i in range(6)]

    k = 0
    SOP = {"isi2_60":[], "isi2_86400":[], "isi16_60":[], "isi16_86400":[]}
    MTS_FB = {"isi2_60":[], "isi2_86400":[], "isi16_60":[], "isi16_86400":[]}
    MTS_FF = {"isi2_60":[], "isi2_86400":[], "isi16_60":[], "isi16_86400":[]}
    for i in range(6):
        pk1SOP = find_max_response(f9B[k][0].A1, count_isi2_60[i])[1]
        pk2SOP = find_max_response(f9B[k+1][0].A1, count_isi16_60[i])[1]
        pk3SOP = find_max_response(f9B[k+2][0].A1, count_isi2_86400[i])[1]
        pk4SOP = find_max_response(f9B[k+3][0].A1, count_isi16_86400[i])[1]
        SOP["isi2_60"].append(pk1SOP)
        SOP["isi16_60"].append(pk2SOP)
        SOP["isi2_86400"].append(pk3SOP)
        SOP["isi16_86400"].append(pk4SOP)
        
        pk1FB = find_max_response(f9C[k].X[2], count_isi2_60[i])[1]
        pk2FB = find_max_response(f9C[k+1].X[2], count_isi16_60[i])[1]
        pk3FB = find_max_response(f9C[k+2].X[2], count_isi2_86400[i])[1]
        pk4FB = find_max_response(f9C[k+3].X[2], count_isi16_86400[i])[1]
        MTS_FB["isi2_60"].append(pk1FB)
        MTS_FB["isi16_60"].append(pk2FB)
        MTS_FB["isi2_86400"].append(pk3FB)
        MTS_FB["isi16_86400"].append(pk4FB)

        pk1FF = find_max_response(f9D[k].X[2], count_isi2_60[i])[1]
        pk2FF = find_max_response(f9D[k+1].X[2], count_isi16_60[i])[1]
        pk3FF = find_max_response(f9D[k+2].X[2], count_isi2_86400[i])[1]
        pk4FF = find_max_response(f9D[k+3].X[2], count_isi16_86400[i])[1]
        MTS_FF["isi2_60"].append(pk1FF)
        MTS_FF["isi16_60"].append(pk2FF)
        MTS_FF["isi2_86400"].append(pk3FF)
        MTS_FF["isi16_86400"].append(pk4FF)

        k += 4
    
    SOP_bar_data2 = np.array([np.mean(np.array(SOP["isi2_60"])[:,0:5]) , np.mean(np.array(SOP["isi2_60"])[:,25:]), np.mean(np.array(SOP["isi2_86400"])[:,25:])])
    MTS_FB_bar_data2 = np.array([np.mean(np.array(MTS_FB["isi2_60"])[:,0:5]) , np.mean(np.array(MTS_FB["isi2_60"])[:,25:]), np.mean(np.array(MTS_FB["isi2_86400"])[:,25:])])
    MTS_FF_bar_data2 = np.array([np.mean(np.array(MTS_FF["isi2_60"])[:,0:5]) , np.mean(np.array(MTS_FF["isi2_60"])[:,25:]), np.mean(np.array(MTS_FF["isi2_86400"])[:,25:])])
    
    SOP_bar_data16 = np.array([np.mean(np.array(SOP["isi16_60"])[:,0:5]) , np.mean(np.array(SOP["isi16_60"])[:,25:]), np.mean(np.array(SOP["isi16_86400"])[:,25:])])
    MTS_FB_bar_data16 = np.array([np.mean(np.array(MTS_FB["isi16_60"])[:,0:5]) , np.mean(np.array(MTS_FB["isi16_60"])[:,25:]), np.mean(np.array(MTS_FB["isi16_86400"])[:,25:])])
    MTS_FF_bar_data16 = np.array([np.mean(np.array(MTS_FF["isi16_60"])[:,0:5]) , np.mean(np.array(MTS_FF["isi16_60"])[:,25:]), np.mean(np.array(MTS_FF["isi16_86400"])[:,25:])])

    SOP_trials2_noblocks = np.mean(np.array(SOP["isi2_60"])[:,5:25],axis=0)
    SOP_trials2 = [(SOP_trials2_noblocks[i] + SOP_trials2_noblocks[i+1])/2 for i in range(0,20,2)]
    SOP_trials16_noblocks = np.mean(np.array(SOP["isi16_60"])[:,5:25],axis=0)
    SOP_trials16 = [(SOP_trials16_noblocks[i] + SOP_trials16_noblocks[i+1])/2 for i in range(0,20,2)]

    MTS_FB_trials2_noblocks = np.mean(np.array(MTS_FB["isi2_60"])[:,5:25],axis=0)
    MTS_FB_trials2 = [(MTS_FB_trials2_noblocks[i] + MTS_FB_trials2_noblocks[i+1])/2 for i in range(0,20,2)]
    MTS_FB_trials16_noblocks = np.mean(np.array(MTS_FB["isi16_60"])[:,5:25],axis=0)
    MTS_FB_trials16 = [(MTS_FB_trials16_noblocks[i] + MTS_FB_trials16_noblocks[i+1])/2 for i in range(0,20,2)]

    MTS_FF_trials2_noblocks = np.mean(np.array(MTS_FF["isi2_60"])[:,5:25],axis=0)
    MTS_FF_trials2 = [(MTS_FF_trials2_noblocks[i] + MTS_FF_trials2_noblocks[i+1])/2 for i in range(0,20,2)]
    MTS_FF_trials16_noblocks = np.mean(np.array(MTS_FF["isi16_60"])[:,5:25],axis=0)
    MTS_FF_trials16 = [(MTS_FF_trials16_noblocks[i] + MTS_FF_trials16_noblocks[i+1])/2 for i in range(0,20,2)]
    
    davis = davis_9()


    bar_data2 = [davis[0], SOP_bar_data2, MTS_FB_bar_data2, MTS_FF_bar_data2]
    bar_data16 = [davis[1], SOP_bar_data16, MTS_FB_bar_data16, MTS_FF_bar_data16]
    trials_2 = [davis[2][0], SOP_trials2, MTS_FB_trials2, MTS_FF_trials2]
    trials_16 = [davis[2][1], SOP_trials16, MTS_FB_trials16, MTS_FF_trials16]
    
    minor_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 60, 86400]
    major_labels = ["Pre-test", "Habituation (block of trials)", "Post-test"]

    width = 0.35

    k = 0
    for i in range(2):
        for j in range(2):
            ax[i, j].bar([1 - width / 2, 23 - width / 2, 25 - width / 2], bar_data2[k], width)
            ax[i, j].bar([1 + width / 2, 23 + width / 2, 25 + width / 2], bar_data16[k], width)
            ax[i, j].plot([3, 5, 7, 9, 11, 13, 15, 17, 19, 21], trials_2[k], "o-", label = "ISI = 2")
            ax[i, j].plot([3, 5, 7, 9, 11, 13, 15, 17, 19, 21], trials_16[k], "o-", label = "ISI = 16")
            ax[i, j].set_xticks([3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25])
            ax[i, j].set_xticklabels(minor_labels,fontsize=9)
            ax[i, j].tick_params(axis='x', which='both', bottom=False, top=False)
            ax[i, j].tick_params(axis='y', direction='in')
            
            if i == 0 and j == 0:
                ax[i, j].legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2)
        
            prim = ax[i, j].secondary_xaxis(location=0)
            prim.set_xticks([4, 6, 8, 10, 12, 14, 16, 18, 20, 24], labels=[])
            prim.tick_params('x', length=20)

            sec = ax[i, j].secondary_xaxis(location=0)
            sec.set_xticks([0, 2, 22, 26], labels=[])
            sec.tick_params('x', length=40)

            third = ax[i, j].secondary_xaxis(location=0)
            third.set_xticks([1, 12, 24], labels=major_labels)
            third.tick_params('x', length=40, which='both',  bottom=False, top=False)
            k += 1
    
    ax[0,0].set_xlim(0,26)
    ax[0,0].set_ylim(0,1)
    ax[0, 0].set_title("A) Davis (1970)")
    ax[0, 0].set_ylabel("Response")

    ax[0,1].set_xlim(0,26)
    ax[0,1].set_ylim(0,1)
    ax[0, 1].set_title("B) SOP")
    ax[0, 1].set_ylabel("Response")
    
    ax[1,0].set_xlim(0,26)
    ax[1,0].set_ylim(0,1)
    ax[1, 0].set_title("C) MTS-Feedback")
    ax[1, 0].set_ylabel("Response")
    
    ax[1,1].set_xlim(0,26)
    ax[1,1].set_ylim(0,1)
    ax[1, 1].set_title("D) MTS-Feedforward")
    ax[1, 1].set_ylabel("Response")

    fig9.suptitle("Figure 9",fontsize=16)
    
    plt.tight_layout()
    plt.show()

def fig10(f10A, f10B, f10C, f10D, stimuli, time):
    print("Plotting Figure 10")
    
    fig10, ax = plt.subplots(2, 2, figsize=(12, 10),sharex=True, sharey=True)

    count = [trial_count(stimuli[i],time) for i in range(18)]

    MTS_FB = []
    MTS_FF = []
    SOP_in = []
    SOP_out = []
    for i in range(18):
        MTS_FB.append(find_max_response(f10A[i].X[2], count[i])[1])
        MTS_FF.append(find_max_response(f10B[i].X[2], count[i])[1])
        SOP_in.append(find_max_response(f10C[i][0].A1, count[i])[1])
        SOP_out.append(find_max_response(f10D[i][0].A1, count[i])[1])

    prehabituation = 9*[0.9]
    
    habituation = [[MTS_FB[i][-2] for i in range(9)], [MTS_FF[i][-2] for i in range(9)], [SOP_in[i][-2] for i in range(9)], [SOP_out[i][-2] for i in range(9)]]

    ri1800 =[[MTS_FB[i][-1] for i in range(9)], [MTS_FF[i][-1] for i in range(9)], [SOP_in[i][-1] for i in range(9)], [SOP_out[i][-1] for i in range(9)]]

    ri3600 =[[MTS_FB[i+9][-1] for i in range(9)], [MTS_FF[i+9][-1] for i in range(9)], [SOP_in[i+9][-1] for i in range(9)], [SOP_out[i+9][-1] for i in range(9)]]

    k = 0
    for i in range(2):
        for j in range(2):
            ax[i, j].plot([1,3, 5, 7, 9, 11, 13, 15, 17], prehabituation, "o-", label = "Pre-habituation")
            ax[i, j].plot([1,3, 5, 7, 9, 11, 13, 15, 17], habituation[k], "o-", label = "Habituation")
            ax[i, j].plot([1,3, 5, 7, 9, 11, 13, 15, 17], ri1800[k], "^-", label = "Retention 1800")
            ax[i, j].plot([1,3, 5, 7, 9, 11, 13, 15, 17], ri3600[k], "x-", label = "Retention 3600")

            ax[i, j].set_xticks([1,3, 5, 7, 9, 11, 13, 15, 17])
            ax[i, j].set_xticklabels([2**i for i in range(1,10)])
            ax[i, j].tick_params(axis='x', which='both', bottom=False, top=False)
            ax[i, j].tick_params(axis='y', direction='in')
            
        
            prim = ax[i, j].secondary_xaxis(location=0)
            prim.set_xticks([0,2,4, 6, 8, 10, 12, 14, 16, 18], labels=[])
            k += 1
    
    ax[0,0].set_xlim(0,18)
    ax[0,0].set_ylim(0,1)
    ax[0,0].set_title("A) MTS-Feedback")
    ax[0,0].set_ylabel("Response")
    ax[0,0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    ax[0,1].set_xlim(0,18)
    ax[0,1].set_ylim(0,1)
    ax[0,1].set_title("B) MTS-Feedforward")
    
    ax[1,0].set_xlim(0,18)
    ax[1,0].set_ylim(0,1)
    ax[1,0].set_title("C) SOP (retention in-context)")
    ax[1,0].set_xlabel("ISI")
    ax[1,0].set_ylabel("Response")
    
    ax[1,1].set_xlim(0,18)
    ax[1,1].set_ylim(0,1)
    ax[1,1].set_title("D) SOP (retention out-context)")
    ax[1,1].set_xlabel("ISI")

    fig10.suptitle("Figure 10",fontsize=16)
    
    plt.tight_layout()
    plt.show()

def fig11(f11A_stimulus, f11B_stimulus, f11C, f11D):
    print("Plotting Figure 11")
    
    fig11, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    pkA, _ = find_peaks(f11A_stimulus.A1)
    pkB, _ = find_peaks(f11B_stimulus.A1)
    pkC, _ = find_peaks(f11C.X[2])
    pkD, _ = find_peaks(f11D.X[2])
    
    ax[0, 0].plot(f11A_stimulus.A1, label="A1s (Response")
    for i in [pkA[0], pkA[-2], pkA[-1]]:
        ax[0, 0].text(i,f11A_stimulus.A1[i],f'{f11A_stimulus.A1[i]:.2f}')
    ax[0, 0].plot(f11A_stimulus.A2,"--", label="A2s (Memory)")
    ax[0,0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    ax[0, 1].plot(f11B_stimulus.A1)
    for i in [pkB[0], pkB[-2], pkB[-1]]:
        ax[0, 1].text(i,f11B_stimulus.A1[i],f'{f11B_stimulus.A1[i]:.2f}')
    ax[0, 1].plot(f11B_stimulus.A2,"--")
    
    ax[1, 0].plot(f11C.X[2], label="X3 (Response)")
    for i in [pkC[0], pkC[-2], pkC[-1]]:
        ax[1, 0].text(i,f11C.X[2][i],f'{f11C.X[2][i]:.2f}')
    ax[1, 0].plot(f11C.V[1],"--", label="V2 (Memory)")
    ax[1,0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    
    ax[1, 1].plot(f11D.X[2])
    for i in [pkD[0], pkD[-2], pkD[-1]]:
        ax[1, 1].text(i,f11D.X[2][i],f'{f11D.X[2][i]:.2f}')    
    ax[1, 1].plot(f11D.V[1],"--")
    
    ax[0, 0].set_title("A) SOP Model: Intensity= 0.9")
    ax[0, 1].set_title("B) SOP Model: Intensity= 0.5")
    
    ax[1, 0].set_title("C) MTS-FB Model: Intensity= 0.9")
    ax[1, 1].set_title("D) MTS-FB Model: Intensity= 0.5")

    ax[-1, 0].set_xlabel("Simulated Time")
    ax[-1, 1].set_xlabel("Simulated Time")
    ax[0, 0].set_ylabel("Simulated Values")
    ax[1, 0].set_ylabel("Simulated Values")

    ax[0,0].set_ylim(0,1)
    ax[0,1].set_ylim(0,1)
    ax[1,0].set_ylim(0,1)
    ax[1,1].set_ylim(0,1)
    
    fig11.suptitle("Figure 11",fontsize=16)
    plt.tight_layout()
    plt.show()

def fig12(f12B, f12C, f12D, stimuli, time):
    print("Plotting Figure 12")
    
    fig12, ax = plt.subplots(2, 2, figsize=(12, 10), sharey=True)

    count = [trial_count(stimuli[i], time) for i in range(8)]
    
    SOP = []
    MTS_FB = []
    MTS_FF = []
    for i in range(8):
        SOP.append(find_max_response(f12B[i][0].A1, count[i])[1])
        MTS_FB.append(find_max_response(f12C[i].X[2], count[i])[1])
        MTS_FF.append(find_max_response(f12D[i].X[2], count[i])[1])

    davis_wagner = davis_wagner_12()

    SOP_high_pretest = np.array([np.mean([SOP[0][1], SOP[2][1], SOP[4][0], SOP[6][0]]), np.mean([SOP[0][0], SOP[2][0], SOP[4][1], SOP[6][1]])])
    SOP_high_post6 = np.array([np.mean([SOP[0][-1], SOP[4][-2]]), np.mean([SOP[0][-2], SOP[4][-1]])])
    SOP_high_post14 = np.array([np.mean([SOP[2][-1], SOP[6][-2]]), np.mean([SOP[2][-2], SOP[6][-1]])])
    SOP_low_pretest = np.array([np.mean([SOP[1][1], SOP[3][1], SOP[5][0], SOP[7][0]]), np.mean([SOP[1][0], SOP[3][0], SOP[5][1], SOP[7][1]])])
    SOP_low_post6 = np.array([np.mean([SOP[1][-1], SOP[5][-2]]), np.mean([SOP[1][-2], SOP[5][-1]])])
    SOP_low_post14 = np.array([np.mean([SOP[3][-1], SOP[7][-2]]), np.mean([SOP[3][-2], SOP[7][-1]])])
    
    SOP_high = np.concatenate([SOP_high_pretest, [np.nan], SOP_high_post6, [np.nan], SOP_high_post14])
    SOP_low = np.concatenate([SOP_low_pretest, [np.nan], SOP_low_post6, [np.nan], SOP_low_post14])
    
    MTS_FB_high_pretest = np.array([np.mean([MTS_FB[0][1], MTS_FB[2][1], MTS_FB[4][0], MTS_FB[6][0]]), np.mean([MTS_FB[0][0], MTS_FB[2][0], MTS_FB[4][1], MTS_FB[6][1]])])
    MTS_FB_high_post6 = np.array([np.mean([MTS_FB[0][-1], MTS_FB[4][-2]]), np.mean([MTS_FB[0][-2], MTS_FB[4][-1]])])
    MTS_FB_high_post14 = np.array([np.mean([MTS_FB[2][-1], MTS_FB[6][-2]]), np.mean([MTS_FB[2][-2], MTS_FB[6][-1]])])
    MTS_FB_low_pretest = np.array([np.mean([MTS_FB[1][1], MTS_FB[3][1], MTS_FB[5][0], MTS_FB[7][0]]), np.mean([MTS_FB[1][0], MTS_FB[3][0], MTS_FB[5][1], MTS_FB[7][1]])])
    MTS_FB_low_post6 = np.array([np.mean([MTS_FB[1][-1], MTS_FB[5][-2]]), np.mean([MTS_FB[1][-2], MTS_FB[5][-1]])])
    MTS_FB_low_post14 = np.array([np.mean([MTS_FB[3][-1], MTS_FB[7][-2]]), np.mean([MTS_FB[3][-2], MTS_FB[7][-1]])])
    
    MTS_FB_high = np.concatenate([MTS_FB_high_pretest, [np.nan], MTS_FB_high_post6, [np.nan], MTS_FB_high_post14])
    MTS_FB_low = np.concatenate([MTS_FB_low_pretest, [np.nan], MTS_FB_low_post6, [np.nan], MTS_FB_low_post14])

    MTS_FF_high_pretest = np.array([np.mean([MTS_FF[0][1], MTS_FF[2][1], MTS_FF[4][0], MTS_FF[6][0]]), np.mean([MTS_FF[0][0], MTS_FF[2][0], MTS_FF[4][1], MTS_FF[6][1]])])
    MTS_FF_high_post6 = np.array([np.mean([MTS_FF[0][-1], MTS_FF[4][-2]]), np.mean([MTS_FF[0][-2], MTS_FF[4][-1]])])
    MTS_FF_high_post14 = np.array([np.mean([MTS_FF[2][-1], MTS_FF[6][-2]]), np.mean([MTS_FF[2][-2], MTS_FF[6][-1]])])
    MTS_FF_low_pretest = np.array([np.mean([MTS_FF[1][1], MTS_FF[3][1], MTS_FF[5][0], MTS_FF[7][0]]), np.mean([MTS_FF[1][0], MTS_FF[3][0], MTS_FF[5][1], MTS_FF[7][1]])])
    MTS_FF_low_post6 = np.array([np.mean([MTS_FF[1][-1], MTS_FF[5][-2]]), np.mean([MTS_FF[1][-2], MTS_FF[5][-1]])])
    MTS_FF_low_post14 = np.array([np.mean([MTS_FF[3][-1], MTS_FF[7][-2]]), np.mean([MTS_FF[3][-2], MTS_FF[7][-1]])])
    
    MTS_FF_high = np.concatenate([MTS_FF_high_pretest, [np.nan], MTS_FF_high_post6, [np.nan], MTS_FF_high_post14])
    MTS_FF_low = np.concatenate([MTS_FF_low_pretest, [np.nan], MTS_FF_low_post6, [np.nan], MTS_FF_low_post14])

    high = [davis_wagner[0], SOP_high, MTS_FB_high, MTS_FF_high]
    low = [davis_wagner[1], SOP_low, MTS_FB_low, MTS_FF_low]

    k = 0
    for i in range(2):
        for j in range(2):
            ax[i, j].plot([1,3,4,5,7,8,9,11],high[k],"o--", label="Habituation with high intensity")
            ax[i, j].plot([1,3,4,5,7,8,9,11],low[k],"^-", label="Habituation with low intensity")
            
            ax[i, j].set_xticks([1,3, 5, 7, 9, 11])
            ax[i, j].set_xticklabels(3*["Low","High"])
            ax[i, j].tick_params(axis='x', which='both', bottom=False, top=False)
            ax[i, j].tick_params(axis='y', direction='in')

            post6_lab = "6" if k != 0 else "300" 
            post14_lab = "14" if k !=0 else "700"
            
            prim = ax[i, j].secondary_xaxis(location=0)
            prim.set_xticks([2, 6, 10], labels=["Pre-test",f"Post-test {post6_lab} hab.\ntrials", f"Post-test {post14_lab} hab.\ntrials"])
            prim.tick_params('x', length=20)
            
            sec = ax[i, j].secondary_xaxis(location=0)
            sec.set_xticks([0, 4, 8, 12], labels=[])
            sec.tick_params('x', length=45)

            ax[i,j].set_xlim(0,12)
            ax[i,j].set_ylim(0,1)
            if j == 0:
                ax[i,j].set_ylabel("Response")
            k += 1

    ax[0,0].set_title("A) Davis & Wagner, 1968")
    ax[0,0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=1)
    
    ax[0,1].set_title("B) SOP")
    ax[1,0].set_title("C) MTS-Feedback")
    ax[1,1].set_title("D) MTS-Feedforward")

    fig12.suptitle("Figure 12",fontsize=16)
    plt.tight_layout()
    plt.show()

            
def fig13(f13B, f13C, f13D, trials, time):
    print("Plotting Figure 13")
    
    fig13, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    count = trial_count(trials,time)

    james_hughes = james_hughes_13()
    
    SOP = np.array([np.mean([find_max_response(f13B[i][0].A1, count)[1] for i in range(2)], axis=0),
                    np.mean([find_max_response(f13B[i][0].A1, count)[1] for i in range(2, 4)], axis=0)])

    MTS_FB = np.array([np.mean([find_max_response(f13C[i].X[2], count)[1] for i in range(2)], axis=0),
                       np.mean([find_max_response(f13C[i].X[2], count)[1] for i in range(2, 4)], axis=0)])

    MTS_FF = np.array([np.mean([find_max_response(f13D[i].X[2], count)[1] for i in range(2)], axis=0),
                       np.mean([find_max_response(f13D[i].X[2], count)[1] for i in range(2, 4)], axis=0)])

    data = [james_hughes, SOP, MTS_FB, MTS_FF]

    k = 0
    for i in range(2):
        for j in range(2):
            d1 = np.concatenate([data[k][0][:8],[np.nan],data[k][0][-4:]])
            d2 = np.concatenate([data[k][1][:8],[np.nan],data[k][1][-4:]])

            ax[i, j].plot([1,3,5,7,9,11,13,15,16,17,19,21,23], d1, "o-", label="Low Intensity")
            ax[i, j].plot([1,3,5,7,9,11,13,15,16,17,19,21,23], d2,"o--", label="High Intensity")
            
            ax[i, j].set_xticks([i for i in range(1,24,2)])
            ax[i, j].set_xticklabels([i for i in range(1,13)])
            ax[i, j].tick_params(axis='x', which='both', bottom=False, top=False)

            prim = ax[i, j].secondary_xaxis(location=0)
            prim.set_xticks([i for i in range(0,25,2)], labels=[])

            ylabel = "Response"
            ylim = (0, 8) if k == 0 else (0, 1)
            
            ax[i,j].set_xlim(0,24)
            if i == 1:
                ax[i,j].set_xlabel("Trial")
            
            ax[i,j].set_ylim(ylim)
            if j == 0:
                ax[i,j].set_ylabel(ylabel)

           
            k += 1

    ax[0,0].set_title("A) James & Hughes (1969)")
    ax[0,0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    ax[0,1].set_title("B (SOP)")
    ax[1,0].set_title("C (MTS-Feedback)")
    ax[1,1].set_title("D (MTS-Feedforward)")

    
    fig13.suptitle("Figure 13",fontsize=16)
    plt.tight_layout()
    plt.show()

def fig14(f14A, f14B, f14C, f14D, stimuli, time):
    print("Plotting Figure 14")
    
    fig14, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    count = [trial_count(stimuli[i],time) for i in range(18)]

    MTS_FB = []
    MTS_FF = []
    SOP_in = []
    SOP_out = []
    for i in range(18):
        MTS_FB.append(find_max_response(f14A[i].X[2], count[i])[1])
        MTS_FF.append(find_max_response(f14B[i].X[2], count[i])[1])
        SOP_in.append(find_max_response(f14C[i][0].A1, count[i])[1])
        SOP_out.append(find_max_response(f14D[i][0].A1, count[i])[1])

    prehabituation = 9*[1]
    
    habituation = [[MTS_FB[i][-2]/MTS_FB[i][0] for i in range(9)], [MTS_FF[i][-2]/MTS_FF[i][0] for i in range(9)], [SOP_in[i][-2]/SOP_in[i][0] for i in range(9)], [SOP_out[i][-2]/SOP_out[i][0] for i in range(9)]]

    ri1800 =[[MTS_FB[i][-1]/0.5 for i in range(9)], [MTS_FF[i][-1]/0.5 for i in range(9)], [SOP_in[i][-1]/0.5 for i in range(9)], [SOP_out[i][-1]/0.5 for i in range(9)]]

    ri3600 =[[MTS_FB[i+9][-1]/0.5 for i in range(9)], [MTS_FF[i+9][-1]/0.5 for i in range(9)], [SOP_in[i+9][-1]/0.5 for i in range(9)], [SOP_out[i+9][-1]/0.5 for i in range(9)]]

    k = 0
    for i in range(2):
        for j in range(2):
            ax[i, j].plot([1,3, 5, 7, 9, 11, 13, 15, 17], prehabituation, "o-", label = "Pre-habituation")
            ax[i, j].plot([1,3, 5, 7, 9, 11, 13, 15, 17], habituation[k], "o-", label = "Habituation")
            ax[i, j].plot([1,3, 5, 7, 9, 11, 13, 15, 17], ri1800[k], "^-", label = "Retention 1800")
            ax[i, j].plot([1,3, 5, 7, 9, 11, 13, 15, 17], ri3600[k], "x-", label = "Retention 3600")

            ax[i, j].set_xticks([1,3, 5, 7, 9, 11, 13, 15, 17])
            ax[i, j].set_xticklabels([i/10 for i in range(1,10)])
            ax[i, j].tick_params(axis='x', which='both', bottom=False, top=False)
            ax[i, j].tick_params(axis='y', direction='in')



            ax[i,j].set_xlim(0,18)
            ax[i,j].set_ylim(-0.1,1.1)
            if i == 1:
                ax[i,j].set_xlabel("Intensity (p1 or X1)")
            if j == 0:
                ax[i,j].set_ylabel("Standarized Response")
            
            prim = ax[i, j].secondary_xaxis(location=0)
            prim.set_xticks([0,2,4, 6, 8, 10, 12, 14, 16, 18], labels=[])
            k += 1
    
    ax[0,0].set_title("A) MTS-Feedback")
    ax[0,0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    
    ax[0,1].set_title("B) MTS-Feedforward")
    ax[1,0].set_title("C) SOP (retention in-context)")
    ax[1,1].set_title("D) SOP (retention out-context)")

    fig14.suptitle("Figure 14",fontsize=16)
    
    plt.tight_layout()
    plt.show()

def fig15():
    print("Plotting Figure 15")
    
    fig15, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    SOP = correlation_SOP()

    k = 0
    for i in range(2):
        for j in range(2):
            ax[i, j].imshow(SOP[k], cmap='gray', vmin=0, vmax=1)
            norm = mcolors.Normalize(vmin=0, vmax=1)

            for x in range(9):
                for y in range(9):
                    bg_color = plt.cm.gray(norm(SOP[k][x, y]))[0]
                    if SOP[k][x, y] >= 0.45 and SOP[k][x, y] <= 0.55:
                        text_color = (abs(1-bg_color)+0.18,abs(1-bg_color)+0.18,abs(1-bg_color)+0.18,1)
                    else:
                        text_color = (abs(1-bg_color),abs(1-bg_color),abs(1-bg_color),1)

                    ax[i,j].text(y, x, '{:.2f}'.format(SOP[k][x, y]), ha='center', va='center', color=text_color)


            if i == 1:
                ax[i, j].set_xticks([i for i in range(9)])
                ax[i, j].set_xticklabels([i/10 for i in range(1,10)])
                ax[i,j].set_xlabel("pd1")
            else:
                ax[i, j].set_xticks([i for i in range(9)], labels=[])

            if j == 0:
                ax[i, j].set_yticks([i for i in range(9)])
                ax[i, j].set_yticklabels([i for i in range(9,0,-1)])
                ax[i,j].set_ylabel("pd1/pd2")
            else:
                ax[i, j].set_yticks([i for i in range(9)], labels=[])
            
            ax[i, j].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
            
            k +=1

    ax[0,0].set_title("A) Rankin & Broster (1992)")
    ax[0,1].set_title("B) Davis (1970)")
    ax[1,0].set_title("C) Davis & Wagner (1968)")
    ax[1,1].set_title("D) James & Hughes (1969)")

    fig15.suptitle("Figure 15",fontsize=16)
    
    plt.tight_layout()
    plt.show()

def fig16():
    print("Plotting Figure 16")
    
    fig16, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    MTS = correlation_MTS()

    k = 0
    for i in range(2):
        for j in range(2):
            ax[i, j].imshow(MTS[k], cmap='gray', vmin=0, vmax=1)
            norm = mcolors.Normalize(vmin=0, vmax=1)

            for x in range(9):
                for y in range(9):
                    bg_color = plt.cm.gray(norm(MTS[k][x, y]))[0]
                    if MTS[k][x, y] >= 0.45 and MTS[k][x, y] <= 0.55:
                        text_color = (abs(1-bg_color)+0.18,abs(1-bg_color)+0.18,abs(1-bg_color)+0.18,1)
                    else:
                        text_color = (abs(1-bg_color),abs(1-bg_color),abs(1-bg_color),1)

                    ax[i,j].text(y, x, '{:.2f}'.format(MTS[k][x, y]), ha='center', va='center', color=text_color)


            if i == 1:
                ax[i, j].set_xticks([i for i in range(9)])
                ax[i, j].set_xticklabels([i for i in [0.6, 1.2, 1.7, 2.3, 2.8, 3.4, 3.9, 4.5, 5.0]])
                ax[i,j].set_xlabel("$\\lambda_a$")
            else:
                ax[i, j].set_xticks([i for i in range(9)], labels=[])

            if j == 0:
                ax[i, j].set_yticks([i for i in range(9)])
                ax[i, j].set_yticklabels([i for i in range(9,0,-1)])
                ax[i,j].set_ylabel("$\\lambda_a/\\lambda_b$")
            else:
                ax[i, j].set_yticks([i for i in range(9)], labels=[])
            
            ax[i, j].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
            
            k +=1

    ax[0,0].set_title("A) Rankin & Broster (1992)")
    ax[0,1].set_title("B) Davis (1970)")
    ax[1,0].set_title("C) Davis & Wagner (1968)")
    ax[1,1].set_title("D) James & Hughes (1969)")

    fig16.suptitle("Figure 16",fontsize=16)
    
    plt.tight_layout()
    plt.show()

def fig17(fb, ff, sop, stimuli, time):
    print("Plotting Figure 17")
    
    fig17, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    count = [trial_count(stimuli[i],time) for i in range(81)]
    
    MTS_FB = []
    MTS_FF = []
    SOP = []
    for i in range(81):
        MTS_FB.append(find_max_response(fb[i].X[2], count[i])[1])
        MTS_FF.append(find_max_response(ff[i].X[2], count[i])[1])
        SOP.append(find_max_response(sop[i][0].A1, count[i])[1])

    habituation = [[MTS_FB[i][-2]/MTS_FB[i][0] for i in range(81)], [MTS_FF[i][-2]/MTS_FF[i][0] for i in range(81)], [SOP[i][-2]/SOP[i][0] for i in range(81)]]

    for i in range(9):
        if i != 8: 
            ax[0].plot([1,3, 5, 7, 9, 11, 13, 15, 17], habituation[0][9*i:9*(i+1)],  "o-")
            ax[0].plot([1,3, 5, 7, 9, 11, 13, 15, 17], habituation[1][9*i:9*(i+1)],  "o-")
        else:
            ax[0].plot([1,3, 5, 7, 9, 11, 13, 15, 17], habituation[0][9*i:9*(i+1)],  "o-", label = "MTS-FB: X1 = 0.1 - 0.9")
            ax[0].plot([1,3, 5, 7, 9, 11, 13, 15, 17], habituation[1][9*i:9*(i+1)],  "o-", label = "MTS-FF: X1 = 0.1 - 0.9")
        ax[1].plot([1,3, 5, 7, 9, 11, 13, 15, 17], habituation[2][9*i:9*(i+1)],  "o-", label = f"p1 = {(i+1)/10}")

    for i in range(2):
        ax[i].set_xticks([1,3, 5, 7, 9, 11, 13, 15, 17])
        ax[i].set_xticklabels([2**i for i in range(1,10)])
        ax[i].tick_params(axis='x', which='both', bottom=False, top=False)
        ax[i].tick_params(axis='y', direction='in')
        
                   
        ax[i].set_xlim(0,18)
        ax[i].set_ylim(0,1.4)
        ax[i].set_xlabel("ISI")
        ax[i].set_ylabel("Standarized Response")
        
        prim = ax[i].secondary_xaxis(location=0)
        prim.set_xticks([0,2,4, 6, 8, 10, 12, 14, 16, 18], labels=[])

    ax[0].set_title("A) MTS")
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    
    ax[1].set_title("B) SOP")
    ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.31), ncol=3)
    
    fig17.suptitle("Figure 17",fontsize=16)
    
    plt.show()

def fig18(f18A, f18B, f18C, f18D, stimuli, time):
    print("Plotting Figure 18")
    
    fig18, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    count = [trial_count(stimuli[i],time) for i in range(81)]

    MTS_FB = []
    MTS_FF = []
    SOP_in = []
    SOP_out = []
    for i in range(81):
        MTS_FB.append(find_max_response(f18A[i].X[2], count[i])[1])
        MTS_FF.append(find_max_response(f18B[i].X[2], count[i])[1])
        SOP_in.append(find_max_response(f18C[i][0].A1, count[i])[1])
        SOP_out.append(find_max_response(f18D[i][0].A1, count[i])[1])

    ri =[[MTS_FB[i][-1] for i in range(81)], [MTS_FF[i][-1] for i in range(81)], [SOP_in[i][-1] for i in range(81)], [SOP_out[i][-1] for i in range(81)]]


    k = 0
    for i in range(2):
        for j in range(2):
            for z in range(9):
                lab = "X1" if k <= 1 else "p1"
                ax[i, j].plot([1,3, 5, 7, 9, 11, 13, 15, 17], ri[k][9*z:9*(z+1)], "o-", label = f"{lab} = {(z+1)/10}")
           
            ax[i, j].set_xticks([1,3, 5, 7, 9, 11, 13, 15, 17])
            ax[i, j].set_xticklabels([2**i for i in range(1,10)])
            ax[i, j].tick_params(axis='x', which='both', bottom=False, top=False)
            ax[i, j].tick_params(axis='y', direction='in')

            ax[i,j].set_xlim(0,18)
            ax[i,j].set_ylim(0,0.6)
            
            if i == 1:
                ax[i,j].set_xlabel("ISI")
            if j == 0:
                ax[i,j].set_ylabel("Response")
            
            prim = ax[i, j].secondary_xaxis(location=0)
            prim.set_xticks([0,2,4, 6, 8, 10, 12, 14, 16, 18], labels=[])
            k += 1
    
    ax[0,0].set_title("A) MTS-Feedback")
    ax[0,0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=5)
    
    ax[0,1].set_title("B) MTS-Feedforward")
    ax[1,0].set_title("C) SOP (retention in-context)")
    ax[1,1].set_title("D) SOP (retention out-context)")

    fig18.suptitle("Figure 18",fontsize=16)
    
    plt.tight_layout()
    plt.show()


def trial_count(s,time):
    count = []
    c = 0
    for i in time:
        if i in s:
            c += 1
        count.append(c)
    return count

def find_max_response(response, sample):
    indices = np.where(np.diff(sample) != 0)[0] + 1 #find indices where the number of sample change
    indices = np.insert(indices, 0, 0) #add zero to the first term

    max = np.maximum.reduceat(response, indices)
    max_indices = [start + np.argmax(response[start:end]) for start, end in zip(indices, indices[1:])]
    max_indices.append(indices[-1] + np.argmax(response[indices[-1]:]))

    if max[0] == 0:
        max = max[1::]
        max_indices = max_indices[1::]

    return max_indices, max

def rankin_and_broster_7():
    return np.array([[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                      np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                      np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                      np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                      np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
                      0.208229426,0.854114713,0.864089776,1.168329177],
                     [1,0.64516129,0.505806452,0.48516129,0.218064516,0.143225806,
                      0.091612903,0.197419355,0.141935484,0.290322581,0.064516129,0.238709677,
                      0.072258065,0.089032258,0.211612903,0.105806452,0.064516129,0.28,
                      0.089032258,0.101935484,0,0,0,0.038709677,
                      0.180645161,0.067096774,0.025806452,0.038709677,0.07483871,0.078709677, np.nan,
                      0.049875312,0.66957606,0.603491272,0.673316708],
                     [1,0.882692308,0.913461538,0.769230769,0.480769231,0.386538462,
                      0.467307692,0.569230769,0.386538462,0.430769231,0.448076923,0.430769231,
                      0.313461538,0.298076923,0.236538462,0.346153846,0.223076923,0.163461538,
                      0.409615385,0.351923077,0.186538462,0.367307692,0.438461538,0.475,
                      0.380769231,0.123076923,0.251923077,0.309615385,0.182692308,0.413461538, np.nan,
                      0.106927711,0.438253012,0.677710843,0.713855422],
                     [1,0.955671447,1.159061278,0.731421121,0.683181226,0.82398957,
                      0.479791395,0.507170795,0.521512386,0.610169492,0.752281617,0.462842243,
                      0.66232073,0.524119948,0.286831812,0.555410691,0.610169492,0.482398957,
                      0.293350717,0.199478488,0.544980443,0.31029987,0.113428944,0.249022164,
                      0.348109518,0.426336375,0.224250326,0.319426336,0.220338983,0.290743155, np.nan,
                      0.287650602,0.399096386,0.637048193,0.469879518]])

def rankin_and_broster_8():
    return np.array([[0, 0.209375, 0.85625, 0.85, 1.15625],
                     [0, 0.0125, 0.621875, 0.596875, 0.640625],
                     [0, 0.009375, 0.34375, 0.546875, 0.603125],
                     [0, 0.00625, 0.103125, 0.375, 0.146875]])

def davis_9():
    return [
        [0.539,0.3398,0.3859], #isi2
        [0.537,0.17205,0.32225],#isi16
        np.array([[0.276, 0.180, 0.157, 0.134, 0.139, 0.128, 0.146, 0.153, 0.122, 0.132], 
                 [0.583, 0.527, 0.516, 0.486,  0.525, 0.473, 0.437, 0.440, 0.445, 0.458]])]

def davis_wagner_12():
    return np.array([[0.54, 0.83, np.nan, 0.21, 0.46, np.nan, 0.2, 0.37],
                     [0.57, 0.83, np.nan, 0.38, 0.65, np.nan, 0.25, 0.58]])

def james_hughes_13():
    return np.array([[7.74, 6.84, 5.96, 4.94, 3.97, 3.3, 3.4, 2.1, 4.43, 3.265, 2.53, 2.84],
                    [7.9, 6.83, 5.96, 4.39, 4.96, 3.92, 4.11, 3.12, 2.47, 2.105, 1.175, 1.775]])

def correlation_SOP():
    rankin = np.array([[0.81, 0.80, 0.79, 0.79, 0.79, 0.78, 0.78, 0.78, 0.78],
                       [0.84, 0.83, 0.81, 0.81, 0.80, 0.80, 0.80, 0.80, 0.80],
                       [0.88, 0.85, 0.84, 0.83, 0.82, 0.82, 0.82, 0.82, 0.82],
                       [0.90, 0.88, 0.87, 0.85, 0.85, 0.85, 0.85, 0.86, 0.86],
                       [0.91, 0.90, 0.89, 0.88, 0.88, 0.88, 0.87, 0.86, 0.83],
                       [0.91, 0.89, 0.86, 0.85, 0.82, 0.77, 0.70, 0.62, 0.54],
                       [0.87, 0.80, 0.73, 0.66, 0.58, 0.52, 0.47, 0.43, 0.42],
                       [0.80, 0.66, 0.55, 0.49, 0.45, 0.43, 0.41, 0.41, 0.40],
                       [0.71, 0.55, 0.48, 0.44, 0.42, 0.41, 0.40, 0.39, 0.38]])

    davis = np.array([[0.60, 0.62, 0.63, 0.64, 0.65, 0.67, 0.70, 0.74, 0.81],
                      [0.62, 0.64, 0.67, 0.69, 0.73, 0.79, 0.87, 0.94, 0.95],
                      [0.66, 0.70, 0.74, 0.80, 0.88, 0.95, 0.94, 0.87, 0.79],
                      [0.73, 0.79, 0.87, 0.94, 0.93, 0.86, 0.79, 0.75, 0.72],
                      [0.83, 0.91, 0.94, 0.87, 0.79, 0.75, 0.73, 0.72, 0.72],
                      [0.86, 0.88, 0.81, 0.75, 0.73, 0.72, 0.72, 0.72, 0.72],
                      [0.74, 0.75, 0.73, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72],
                      [0.69, 0.71, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72],
                      [0.71, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72]])

    davis_wagner = np.array([[0.87, 0.84, 0.82, 0.82, 0.81, 0.81, 0.81, 0.81, 0.81],
                             [0.86, 0.83, 0.82, 0.82, 0.81, 0.81, 0.81, 0.81, 0.81],
                             [0.86, 0.83, 0.82, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81],
                             [0.86, 0.83, 0.82, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81],
                             [0.86, 0.83, 0.82, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81],
                             [0.86, 0.83, 0.82, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81],
                             [0.87, 0.83, 0.82, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81],
                             [0.88, 0.83, 0.82, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81],
                             [0.89, 0.83, 0.82, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81]])

    james_hughes = np.array([[0.75, 0.66, 0.62, 0.60, 0.60, 0.60, 0.61, 0.62, 0.64],
                             [0.76, 0.68, 0.65, 0.64, 0.64, 0.66, 0.69, 0.73, 0.78],
                             [0.78, 0.71, 0.69, 0.70, 0.73, 0.77, 0.81, 0.85, 0.86],
                             [0.81, 0.77, 0.77, 0.79, 0.83, 0.85, 0.86, 0.86, 0.86],
                             [0.85, 0.83, 0.84, 0.86, 0.87, 0.86, 0.86, 0.85, 0.85],
                             [0.88, 0.87, 0.87, 0.86, 0.86, 0.85, 0.85, 0.85, 0.85],
                             [0.88, 0.87, 0.86, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
                             [0.87, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
                             [0.86, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]])

    return [rankin, davis, davis_wagner, james_hughes]

def correlation_MTS():
    rankin = np.array([[0.77, 0.76, 0.75, 0.73, 0.74, 0.75, 0.73, 0.65, 0.61],
                       [0.77, 0.77, 0.76, 0.73, 0.74, 0.76, 0.75, 0.69, 0.66],
                       [0.77, 0.77, 0.76, 0.74, 0.75, 0.77, 0.76, 0.73, 0.71],
                       [0.77, 0.77, 0.77, 0.75, 0.75, 0.78, 0.79, 0.78, 0.76],
                       [0.77, 0.78, 0.78, 0.76, 0.76, 0.80, 0.82, 0.81, 0.81],
                       [0.77, 0.78, 0.78, 0.77, 0.77, 0.83, 0.85, 0.84, 0.85],
                       [0.77, 0.79, 0.79, 0.79, 0.80, 0.86, 0.91, 0.89, 0.89],
                       [0.78, 0.78, 0.77, 0.78, 0.80, 0.85, 0.92, 0.91, 0.85],
                       [0.78, 0.68, 0.59, 0.58, 0.62, 0.58, 0.53, 0.52, 0.54]])


    davis = np.array([[0.83, 0.87, 0.91, 0.85, 0.75, 0.63, 0.59, 0.60, 0.61],
                      [0.83, 0.86, 0.90, 0.86, 0.79, 0.66, 0.60, 0.59, 0.60],
                      [0.82, 0.85, 0.90, 0.88, 0.82, 0.70, 0.62, 0.59, 0.60],
                      [0.82, 0.84, 0.89, 0.89, 0.85, 0.74, 0.65, 0.60, 0.59],
                      [0.81, 0.82, 0.87, 0.90, 0.86, 0.79, 0.71, 0.62, 0.59],
                      [0.80, 0.78, 0.84, 0.90, 0.88, 0.84, 0.77, 0.68, 0.62],
                      [0.77, 0.72, 0.77, 0.87, 0.90, 0.88, 0.82, 0.70, 0.61],
                      [0.73, 0.56, 0.57, 0.71, 0.80, 0.80, 0.67, 0.48, 0.43],
                      [0.58, 0.19, 0.10, 0.28, 0.48, 0.31, 0.12, 0.14, 0.24]])


    davis_wagner = np.array([[0.81, 0.81, 0.83, 0.89, 0.87, 0.93, 0.92, 0.89, 0.84],
                             [0.81, 0.81, 0.83, 0.89, 0.87, 0.93, 0.93, 0.91, 0.85],
                             [0.81, 0.81, 0.83, 0.89, 0.87, 0.92, 0.94, 0.92, 0.88],
                             [0.81, 0.81, 0.83, 0.90, 0.89, 0.91, 0.94, 0.94, 0.91],
                             [0.81, 0.81, 0.83, 0.89, 0.90, 0.91, 0.94, 0.95, 0.93],
                             [0.81, 0.81, 0.83, 0.89, 0.92, 0.92, 0.93, 0.95, 0.95],
                             [0.81, 0.81, 0.83, 0.89, 0.92, 0.93, 0.94, 0.96, 0.97],
                             [0.81, 0.81, 0.83, 0.89, 0.92, 0.91, 0.92, 0.94, 0.94],
                             [0.81, 0.81, 0.82, 0.83, 0.83, 0.82, 0.83, 0.83, 0.83]])


    james_hughes = np.array([[0.53, 0.53, 0.54, 0.60, 0.79, 0.92, 0.91, 0.88, 0.86],
                             [0.53, 0.53, 0.54, 0.61, 0.79, 0.92, 0.92, 0.90, 0.88],
                             [0.53, 0.53, 0.55, 0.62, 0.78, 0.91, 0.92, 0.91, 0.90],
                             [0.53, 0.53, 0.55, 0.63, 0.77, 0.89, 0.92, 0.92, 0.92],
                             [0.53, 0.53, 0.55, 0.64, 0.77, 0.87, 0.90, 0.91, 0.91],
                             [0.53, 0.53, 0.55, 0.66, 0.74, 0.85, 0.86, 0.87, 0.87],
                             [0.53, 0.53, 0.56, 0.68, 0.77, 0.82, 0.83, 0.83, 0.81],
                             [0.53, 0.53, 0.57, 0.71, 0.79, 0.83, 0.84, 0.83, 0.81],
                             [0.53, 0.53, 0.57, 0.72, 0.80, 0.83, 0.78, 0.74, 0.77]])

    return [rankin, davis, davis_wagner, james_hughes]


