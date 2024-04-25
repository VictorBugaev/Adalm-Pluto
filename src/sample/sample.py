import numpy as np
import math
import matplotlib.pyplot as plt
import scipy

DEBUG = True
DEBUG_OFDM = False
DEBUG_OFDM_DE = False

#Строку символов в бинарный вид
def data_to_byte(data):
    bin_str = ''.join(format(ord(i), '08b') for i in data)
    return np.array(list(map(int, list(bin_str))))
def encode_QPSK(data, mode):
    ampl = 2**1
    if (len(data) % 2 != 0):
        print("QPSK:\nError, check data length", len(data))
        raise "error"
    else:
        sample = [] # массив комплексных чисел
        N = 2
        for i in range(0, len(data), N):
            b2i = data[i]
            b2i1 = data[i+1]
            real = (1 - N * b2i) / np.sqrt(N)
            imag = (1 - N * b2i1) / np.sqrt(N)
            sample.append(complex(real, imag))
        sample = np.asarray(sample)
        sample = sample * ampl
        return sample
def duplication_sample(data, N):
    dup_buff = []
    
    for i in data:
        for i2 in range(N):
            dup_buff.append(i)
    return dup_buff

def formula(value, k):
    return (k - 2 * value)    
def recurse_formula(arr_values, n, pos):
    if(len(arr_values) < 2):
        return 1
    return ( 2**n - (1 - 2 * arr_values[pos]) * recurse_formula(arr_values[2:], n-1, pos))


def calc_coeff(n):
    if(n < 1):
        return 2
    return 2 + 4 * calc_coeff(n-1)

def encode_QAM(data_bit, N):#[0, 1, 0, 1, ....], уровень QAM
    ampl = 2**14
    ad = int(np.log2(N))
    if (len(data_bit) % 2 != 0):
        print("QPSK:\nError, check bit_mass length", len(data_bit))
        raise "error"
        return
    sample = [] # массив комплексных чисел
    k1 = calc_coeff(int(np.log2(N))/2)
    #print("k1 = ", k1)
    for i in range(0, len(data_bit), int(np.log2(N))):
        sr = data_bit[i:i+int(np.log2(N))]
        sr = list(reversed(sr))
        d1 = formula(sr[0], 1)
        d2 = recurse_formula(sr[2:], int(np.log2(N))/2, 0)
        d1i = formula(sr[1], 1)
        d2i = recurse_formula(sr[2:], int(np.log2(N))/2, 1)
        dd = (d1 * d2) /np.sqrt(k1) + ((d1i * d2i) / np.sqrt(k1)) * 1j
        sample.append(dd)
    return sample



#Данные, количество поднесущих, защитный префикс, pilot, шаг расположения pilot, защитный интревал
def OFDM_modulator(data, Nb, N_interval, pilot, Rs, Nz):    
    Rs += 1
    ofdms = np.array([])
    ofdm_indexes = []
    step = Nb - Nz * 2
    Nrs = int(step / Rs)
    if(step % Rs != 0):
        Nrs += 1
    step_data = step - Nrs
    if(DEBUG_OFDM):
        print("Длинна входных данных: ", len(data))
        print("Rs = ", Rs, "N_prefix = ", N_interval)
        print("step = ", step)
        print("Nrs = ", Nrs)
        print("step_data = ", step_data)
    for i in range(step):
        if(i % Rs == 0):
            ofdm_indexes.append(i)
    count_ofdm = 0
    
    for i in range(0, len(data), step_data):
        part = data[i:i + step_data]
        if(len(part) < step_data):
            if(DEBUG_OFDM):
                print("Добивание:", step_data - len(part))
            ending =  np.zeros(step_data - len(part), dtype="complex_")
            #ending += 14 + 14j
            part = np.concatenate([part, ending])
        if(DEBUG_OFDM):
            print("len part = ", len(part))
        ofdm = np.zeros(step, dtype="complex_")
        if(DEBUG_OFDM):
            print("len do Nz:", ofdm.size)
        i2 = 0
        for i3 in range(ofdm.size):
            if(i3 % Rs == 0):
                ofdm[i3] = pilot
                if(DEBUG_OFDM):
                    print(i3, end=", ")
                
            else:
                ofdm[i3] = part[i2]
                i2 += 1
        if(DEBUG_OFDM):
            print()
            print(ofdm)
        interval_zeros = np.zeros(Nz, dtype="complex_")
        ofdm = np.concatenate([interval_zeros, ofdm, interval_zeros])
        if(DEBUG_OFDM):    
            print("len ofdm complete", len(ofdm))
        ofdm = np.fft.ifft(ofdm)
        if(DEBUG_OFDM):
            print("len ofdm to time:", len(ofdm))
        ofdm = np.concatenate([ofdm[len(ofdm) - N_interval:], ofdm])
        if(DEBUG_OFDM):    
            print("len prefix + ofdm: ", len(ofdm))
            
        # plt.figure(100 + (i * 2), figsize=(10, 10))
        # plt.subplot(2, 2, 1)
        # plt.plot(ofdm)
        # plt.subplot(2, 2, 2)
        # plt.plot(abs(np.fft.fft(ofdm,int(1e6))))
        
        ofdms = np.concatenate([ofdms, ofdm])
        count_ofdm += 1
    if(DEBUG_OFDM):
        print(ofdm_indexes)
        print("count ofdm = ", count_ofdm)
        print("len ofdms = ", len(ofdms))
    param = [step, step_data, Nrs]
    return [ofdms, count_ofdm, ofdm_indexes, param]
        
#  оценка АЧХ
def assessment_FR(Rrx, Rtx, indexes, len_inter):
    H = Rrx / Rtx
    # plt.figure(53 + s, figsize=(10, 10))
    # plt.subplot(2, 2, 1)
    # plt.plot(H)
    indexes = np.array(indexes)
    ts = np.linspace(0, len_inter, len_inter)
    Heq = np.interp(ts, indexes, H)
    # plt.subplot(2, 2, 2)
    # plt.plot(Heq)
    #print("Heq ", len(Heq), "  ",Heq)
    return Heq
    
##  ofdms_argv = [count_ofdm, ofdm_indexes, param]
def OFDM_demodulator(ofdms, ofdms_argv, Nb, N_interval, pilot, Rs, Nz):
    if(DEBUG_OFDM_DE):
        print(len(ofdms_argv))
        print("Nb = ", Nb, "N_prefix = ", N_interval)
    step = Nb + N_interval
    count_ofdm = 0
    data = np.array([])
    for i in range(0, len(ofdms), step):
        if(DEBUG_OFDM_DE):
            print("i = ", i)
        ofdm= ofdms[i : i + step]
        if(DEBUG_OFDM_DE):
            print("len ofdm = ", len(ofdm), "[", i, ":", i + step, "]")
        ofdm = ofdm[N_interval:]
        if(DEBUG_OFDM_DE):
            print("len ofdm, delete prefix:", len(ofdm))
        
        ofdm = np.fft.fft(ofdm)
        ofdm = ofdm[Nz:len(ofdm)-Nz]
        if(DEBUG_OFDM_DE):
            print("len ofdm, delete zeros: ", len(ofdm))
        Rtx = np.array([pilot for i in range(len(ofdms_argv[1]))])
        Rrx = []
        for i2 in range(0, len(ofdm), Rs+1):
            if(DEBUG_OFDM_DE):
                print(i2, end=" ")
            Rrx.append(ofdm[i2])
        if(DEBUG_OFDM_DE):
            print()     
        Rrx = np.array(Rrx)
        if(DEBUG_OFDM_DE):
            print(Rrx)
            print("Оценка АЧХ")
        Heq = assessment_FR(Rrx, Rtx, ofdms_argv[1], len(ofdm))
        
        ofdm = ofdm / Heq
        if(DEBUG_OFDM_DE):
            print("После АЧХ")
            Rrx = []
            for i2 in range(0, len(ofdm), Rs+1):
                print(i2, end=" ")
                Rrx.append(ofdm[i2])
            print()   
            Rrx = np.array(Rrx)
            print(Rrx)
        de_ofdm = []
        i3 = 0
        for i2 in range(len(ofdm)):
            #print("i2 = ", i2, " i3 = ", i3)
            if(i3 < len(ofdms_argv[1]) and i2 == ofdms_argv[1][i3]):
                i3 += 1
            else:
                #print("add ", i2)
                de_ofdm.append(ofdm[i2])
        de_ofdm = np.array(de_ofdm)
        data = np.concatenate([data, de_ofdm])
        count_ofdm += 1
        if(count_ofdm == ofdms_argv[0]):
            break
    return data
##  ofdms_argv = [count_ofdm, ofdm_indexes, param]
def OFDM_demodulator_NO_FR(ofdms, ofdms_argv, Nb, N_interval, pilot, Rs, Nz):
    if(DEBUG_OFDM_DE):
        print(len(ofdms_argv))
        print("Nb = ", Nb, "N_prefix = ", N_interval)
    step = Nb + N_interval
    count_ofdm = 0
    data = np.array([])
    for i in range(0, len(ofdms), step):
        if(DEBUG_OFDM_DE):
            print("i = ", i)
        ofdm= ofdms[i : i + step]
        if(DEBUG_OFDM_DE):
            print("len ofdm = ", len(ofdm), "[", i, ":", i + step, "]")
        ofdm = ofdm[N_interval:]
        if(DEBUG_OFDM_DE):
            print("len ofdm, delete prefix:", len(ofdm))
        
        ofdm = np.fft.fft(ofdm)
        ofdm = ofdm[Nz:len(ofdm)-Nz]
        if(DEBUG_OFDM_DE):
            print("len ofdm, delete zeros: ", len(ofdm))
        Rtx = np.array([pilot for i in range(len(ofdms_argv[1]))])
        Rrx = []
        for i2 in range(0, len(ofdm), Rs+1):
            if(DEBUG_OFDM_DE):
                print(i2, end=" ")
            Rrx.append(ofdm[i2])
        if(DEBUG_OFDM_DE):
            print()   
        Rrx = np.array(Rrx)
        if(DEBUG_OFDM_DE):
            print(Rrx)
        de_ofdm = []
        i3 = 0
        for i2 in range(len(ofdm)):
            #print("i2 = ", i2, " i3 = ", i3)
            if(i3 < len(ofdms_argv[1]) and i2 == ofdms_argv[1][i3]):
                i3 += 1
            else:
                #print("add ", i2)
                de_ofdm.append(ofdm[i2])
        de_ofdm = np.array(de_ofdm)
        data = np.concatenate([data, de_ofdm])
        count_ofdm += 1
        if(count_ofdm == ofdms_argv[0]):
            break
    return data

def norm_corr1(x, y):
    x_norm = (x - np.mean(x)) / np.std(x)
    y_norm = (y - np.mean(y)) / np.std(y)
    corrR = np.vdot(x_norm.real, y_norm.real) / (np.linalg.norm(x_norm.real) * np.linalg.norm(y_norm.real))
    corrI = np.vdot(x_norm.imag, y_norm.imag) / (np.linalg.norm(x_norm.imag) * np.linalg.norm(y_norm.imag))
    
    return max(corrR, corrI)    
def norm_corr(x,y):
    #x_normalized = (cp1 - np.mean(cp1)) / np.std(cp1)
    #y_normalized = (cp2 - np.mean(cp2)) / np.std(cp2)

    c_real = np.vdot(x.real, y.real) / (np.linalg.norm(x.real) * np.linalg.norm(y.real))
    c_imag = np.vdot(x.imag, y.imag) / (np.linalg.norm(x.imag) * np.linalg.norm(y.imag))
    
    return c_real+1j*c_imag

def norm_corr3(data, seq):
    sum_c = 0
    for i in range(len(seq)):
        sum_c += data[i] * seq[i]
    
    a = data[:len(seq)]
    #return sum_c / math.sqrt( sum(seq) * sum(data[step : step + len(seq)]) )
    return sum_c / math.sqrt( sum(seq * seq) * sum(a * a ))
  

def correlat_ofdm(rx_ofdm, cp,num_carrier):
    max = 0
    rx1 = rx_ofdm
    cor = []
    cor_max = []
    index = -1
    for j in range(len(rx1)):
        corr_sum =abs(norm_corr(rx1[:cp],np.conjugate(rx1[num_carrier:num_carrier+cp])))
        #print(corr_sum)
        cor.append(corr_sum)
        if corr_sum > max and (corr_sum.imag > 0.9 or corr_sum.real > 0.9):
            cor_max.append(corr_sum)
            max = corr_sum
            #print(np.round(max))
            index = j
        rx1= np.roll(rx1,-1)

    #cor = np.asarray(cor)
    #ic(cor_max)
    #plt.figure(3)
    #plt.plot(cor.real)
    #plt.plot(cor.imag)
    #print("ind",index)
    #return (index - (cp+num_carrier))
    return index

def del_prefix_while(data, Nb, N_interval):
    out_data = np.array([])
    print("Длинна входных данных",len(data))
    for i in range(N_interval, len(data), Nb):
        #print(i)
        #out_data += data[i:i + Nb]
        out_data = np.concatenate([out_data, data[i:i + Nb]])
        
    return out_data

#=======================================================================================================================

#Не преминяется для QAM16 
def TED_loop_filter(data): #ted loop filter 
    BnTs = 0.01 
    Nsps = 10
    C = np.sqrt(2)
    Kp = 1
    teta = ((BnTs)/(Nsps))/(C + 1/(4*C))
    K1 = (-4*C*teta)/((1+2*C*teta+teta**2)*Kp)
    K2 = (-4*teta**2)/((1+2*C*teta+teta**2)*Kp)
    print("K1 = ", K1)
    print("K2 = ", K2)
    #K1_2 = (1/Kp)*((((4*C)/(Nsps**2))*((BnTs/(C + (1/4*C)))**2))/(1 + ((2 * C)/Nsps)*(BnTs/(C + (1/(4*C))))+(BnTs/(Nsps*(C+(1/4*C))))**2))
    err = np.zeros(len(data)//10, dtype = "complex_")
    data = np.roll(data,-0)
    nsp = 10
    p1 = 0
    p2 = 0
    n = 0
    mass_cool_inex = []
    mass_id = []
    for ns in range(0,len(data)-(2*nsp),nsp):
        #real = (data.real[ns+n] - data.real[nsp+ns+n]) * data.real[n+(nsp)//2+ns]
        #imag = (data.imag[ns+n] - data.imag[nsp+ns+n]) * data.imag[n+(nsp)//2+ns]
        real = (data.real[nsp+ns+n] - data.real[ns+n]) * data.real[n + (nsp)//2+ns]
        imag = (data.imag[nsp+ns+n] - data.imag[ns+n] ) * data.imag[n + (nsp)//2+ns]
        err[ns//nsp] = np.mean(real + imag)
        #err[ns//nsp] = np.mean((np.conj(data[nsp+ns+n]) - np.conj(data[ns+n]))*(data[n + (nsp)//2+ns])) 
        error = err.real[ns//nsp]
        p1 = error * K1
        p2 = p2 + p1 + error * K2
        #print(ns ," p2 = ",p2)  
        while(p2 > 1):
            #print(ns ," p2 = ",p2)
            p2 = p2 - 1
        while(p2 < -1):
            #print(ns ," p2 = ",p2)
            p2 = p2 + 1
        
        n = round(p2*10)  
        n1 = n+ns+nsp   
        mass_cool_inex.append(n1)
        mass_id.append(n)

    #mass_cool_inex = [math.ceil(mass_cool_inex[i]) for i in range(len(mass_cool_inex))]
    mass1 = np.asarray(mass_cool_inex)
    mass = np.asarray(mass_id)
    #plt.figure(10, figsize=(10, 10))
    #plt.subplot(2,1,1)
    #plt.plot(err) 
    #plt.subplot(2,1,2)
    #plt.plot(mass)   
    
    return mass1
def PLL(conv):
    mu = 1  
    theta = 1
    phase_error = np.zeros(len(conv))  
    output_signal = np.zeros(len(conv), dtype=np.complex128)

    for n in range(len(conv)):
        theta_hat = np.angle(conv[n]) 
        #print(theta_hat)
        phase_error[n] = theta_hat - theta  
        output_signal[n] = conv[n] * np.exp(-1j * theta)  
        theta = theta + mu * phase_error[n]  
    return output_signal



def gardner_TED(data):
    error = 0
    tau = 2
    t1 = 1
    errors = [0 for i in range(len(data))]
    for i in range(1, len(data)):
        t1 = i
        t2 = t1 + tau
        errors[i] = (data.real[i-1]) % N
    