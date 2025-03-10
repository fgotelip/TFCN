  j=1
    for i in range(N):
        print("i=",i)
        if i<N/2:
            w[i] = (b-a)*j/(2*N)
            print(w[i])
            t[i] = a + j*w[i]/2
            print(t[i])
            j+=1
        else: # i>=N/2
            j=j-1
            w[i] = (b-a)*j/(2*N)
            print(w[i])
            t[i] = -1*(a + j*w[i]/2)
            print(t[i])


    if N%2==0:
        t[N//2] = (b-a)/2